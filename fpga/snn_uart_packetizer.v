// 12-Byte Binary UART Packetizer — 460800 8N1 @ 27 MHz                  [FIXED]
// Frame: [0xA5][FLAGS][SPIKES_LE][PEAK_V_LE][DELTA_LE][WIN_US_LE][XOR_CRC][0x5A]
//
// FIXES vs Phase-1 version
//  1. Baud 115200 -> 460800. 12 B x 10 bit = 120 bit = 1.042 ms @115200, which is
//     LONGER than the 1 ms window, so every other frame was being dropped.
//     At 460800 a frame takes ~0.26 ms (CYCLES_PER_BIT = 27e6/460800 = 58.59 -> 59, +0.7 %).
//  2. Removed the extra per-byte state (was 11 bit-times per byte instead of 10).
//  3. 'overrun' pulses if a trigger arrives while still transmitting, so a dropped
//     frame can never go unnoticed again (the testbench asserts it stays 0).
module snn_uart_packetizer #(
    parameter [15:0] CYCLES_PER_BIT = 16'd59   // 27,000,000 / 460,800 = 58.59
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        send_trigger,           // 1-cycle pulse, once per 1 ms window
    input  wire [2:0]  trig_flags,             // {current_spike, vibration, acoustic}
    input  wire [15:0] spike_count,
    input  wire [15:0] peak_v_q88,
    input  wire [15:0] delta_rate,
    output reg         uart_tx,
    output reg         tx_busy,
    output reg         overrun
);
    reg [7:0]  packet [0:11];
    reg [3:0]  byte_idx;
    reg [3:0]  bit_idx;      // 0 start | 1..8 data | 9 stop | 10 stop-hold / next byte
    reg [15:0] baud_cnt;

    wire [7:0] flags_byte = {5'b00000, trig_flags};
    wire [7:0] crc_calc = flags_byte ^
                          spike_count[7:0] ^ spike_count[15:8] ^
                          peak_v_q88[7:0]  ^ peak_v_q88[15:8]  ^
                          delta_rate[7:0]  ^ delta_rate[15:8]  ^
                          8'hE8 ^ 8'h03;                    // window = 1000 us (0x03E8)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx  <= 1'b1;
            tx_busy  <= 1'b0;
            overrun  <= 1'b0;
            byte_idx <= 4'd0;
            bit_idx  <= 4'd0;
            baud_cnt <= 16'd0;
        end else begin
            overrun <= 1'b0;
            if (!tx_busy) begin
                uart_tx <= 1'b1;
                if (send_trigger) begin
                    packet[0]  <= 8'hA5;
                    packet[1]  <= flags_byte;
                    packet[2]  <= spike_count[7:0];
                    packet[3]  <= spike_count[15:8];
                    packet[4]  <= peak_v_q88[7:0];
                    packet[5]  <= peak_v_q88[15:8];
                    packet[6]  <= delta_rate[7:0];
                    packet[7]  <= delta_rate[15:8];
                    packet[8]  <= 8'hE8;
                    packet[9]  <= 8'h03;
                    packet[10] <= crc_calc;
                    packet[11] <= 8'h5A;
                    tx_busy  <= 1'b1;
                    byte_idx <= 4'd0;
                    bit_idx  <= 4'd0;
                    baud_cnt <= 16'd0;
                end
            end else begin
                if (send_trigger) overrun <= 1'b1;       // a frame would have been lost
                if (baud_cnt < CYCLES_PER_BIT - 16'd1) begin
                    baud_cnt <= baud_cnt + 16'd1;
                end else begin
                    baud_cnt <= 16'd0;
                    if (bit_idx == 4'd0) begin
                        uart_tx <= 1'b0;                  // start bit
                        bit_idx <= 4'd1;
                    end else if (bit_idx <= 4'd8) begin
                        uart_tx <= packet[byte_idx][bit_idx - 4'd1];   // LSB first
                        bit_idx <= bit_idx + 4'd1;
                    end else if (bit_idx == 4'd9) begin
                        uart_tx <= 1'b1;                  // stop bit
                        bit_idx <= 4'd10;
                    end else begin                        // bit_idx == 10: stop bit fully held
                        if (byte_idx < 4'd11) begin
                            byte_idx <= byte_idx + 4'd1;
                            uart_tx  <= 1'b0;             // next byte's start bit, no gap
                            bit_idx  <= 4'd1;
                        end else begin
                            tx_busy <= 1'b0;              // frame complete
                        end
                    end
                end
            end
        end
    end
endmodule
