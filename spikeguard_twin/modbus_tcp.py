"""A tiny dependency-free Modbus TCP server (function 0x03, read holding registers) and client.

Why hand-rolled: pymodbus changed its server API between major versions, and a 60-line server
you fully understand is easier to document than a library API that may have moved.
The REAL SpikeGuard node speaks Modbus RTU over RS-485 with the SAME register map, so only the
transport changes (use pymodbus ModbusSerialClient on the Pi for that)."""
import socketserver
import struct
import threading
import time

from common.schemas import MachineState, ZONES, ZONE_CODE

# Register map (the contract). All registers are unsigned 16-bit.
REG_LOOP_MA_X100 = 0     # loop current in mA x 100
REG_ZONE = 1             # 0 healthy, 1 warning, 2 critical, 3 fault
REG_SPIKE_RATE_X10 = 2   # spike rate in Hz x 10
REG_HEALTH_X100 = 3      # health index x 100
REG_HEARTBEAT = 4        # increments every update (lets the client detect a frozen node)
REG_SOURCE = 5           # 0 simulated, 1 hardware
N_REGS = 6


class RegisterBank:
    def __init__(self, n=N_REGS):
        self._regs = [0] * n
        self._lock = threading.Lock()

    def read(self, addr, qty):
        with self._lock:
            return self._regs[addr:addr + qty]

    def publish(self, st: MachineState):
        with self._lock:
            self._regs[REG_LOOP_MA_X100] = int(round(st.loop_ma * 100)) & 0xFFFF
            self._regs[REG_ZONE] = ZONE_CODE[st.zone]
            self._regs[REG_SPIKE_RATE_X10] = int(round(st.spike_rate_hz * 10)) & 0xFFFF
            self._regs[REG_HEALTH_X100] = int(round(st.health_index * 100)) & 0xFFFF
            self._regs[REG_HEARTBEAT] = (self._regs[REG_HEARTBEAT] + 1) & 0xFFFF
            self._regs[REG_SOURCE] = 0 if st.source == "simulated" else 1


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        bank = self.server.bank
        while True:
            hdr = _recv_exact(self.request, 7)                # MBAP header
            if hdr is None:
                return
            tid, pid, length, unit = struct.unpack(">HHHB", hdr)
            pdu = _recv_exact(self.request, length - 1)
            if pdu is None:
                return
            fc = pdu[0]
            if fc == 3 and len(pdu) >= 5:
                addr, qty = struct.unpack(">HH", pdu[1:5])
                if qty < 1 or addr + qty > N_REGS:
                    resp = struct.pack(">BB", fc | 0x80, 2)     # illegal data address
                else:
                    vals = bank.read(addr, qty)
                    resp = struct.pack(">BB", fc, qty * 2) + struct.pack(f">{qty}H", *vals)
            else:
                resp = struct.pack(">BB", fc | 0x80, 1)         # illegal function
            self.request.sendall(struct.pack(">HHHB", tid, 0, len(resp) + 1, unit) + resp)


class ModbusServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host="127.0.0.1", port=5020, bank=None):
        super().__init__((host, port), _Handler)
        self.bank = bank or RegisterBank()

    def start(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self


def read_holding(host, port, addr=0, qty=N_REGS, unit=1, timeout=1.0):
    """Minimal Modbus TCP client: returns a list of register values, or raises OSError."""
    import socket
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(struct.pack(">HHHBBHH", 1, 0, 6, unit, 3, addr, qty))
        hdr = _recv_exact(s, 7)
        if hdr is None:
            raise OSError("no reply")
        _, _, length, _ = struct.unpack(">HHHB", hdr)
        pdu = _recv_exact(s, length - 1)
        if pdu is None or pdu[0] != 3:
            raise OSError(f"modbus exception: {pdu!r}")
        return list(struct.unpack(f">{pdu[1] // 2}H", pdu[2:2 + pdu[1]]))


class MachineStateReader:
    """Polls the node and turns raw registers into a MachineState, or None when stale/unreachable.
    'Stale' = heartbeat has not changed for stale_after_s (the node froze or the link died)."""

    def __init__(self, host="127.0.0.1", port=5020, stale_after_s=3.0):
        self.host, self.port, self.stale_after_s = host, port, stale_after_s
        self._last_hb = None
        self._last_change = time.time()

    def read(self):
        try:
            r = read_holding(self.host, self.port)
        except OSError:
            return None
        now = time.time()
        if r[REG_HEARTBEAT] == 0:                      # node has never published anything yet
            return None
        if r[REG_HEARTBEAT] != self._last_hb:
            self._last_hb, self._last_change = r[REG_HEARTBEAT], now
        if now - self._last_change > self.stale_after_s:
            return None
        zone = ZONES[r[REG_ZONE]] if r[REG_ZONE] < len(ZONES) else "fault"
        return MachineState(ts=now, loop_ma=r[REG_LOOP_MA_X100] / 100.0, zone=zone,
                            spike_rate_hz=r[REG_SPIKE_RATE_X10] / 10.0,
                            health_index=r[REG_HEALTH_X100] / 100.0,
                            source="simulated" if r[REG_SOURCE] == 0 else "hardware")
