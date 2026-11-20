`timescale 1ns/1ps
// Self-checking testbench for spikeguard_top (Icarus Verilog / Verilator-friendly Verilog-2001)
//  * drives realistic comparator scenarios (idle, steady channels, isolated glitch, short/long bursts)
//  * decodes the 460800 8N1 UART stream bit-by-bit exactly like a receiver would
//  * checks framing (SOF/EOF), XOR-CRC, 1000 us window field, hw_irq <-> spike_count agreement,
//    spike-count ranges, and that NO frame is ever dropped (overrun == 0)
//  * writes sim/uart_frames.csv  -> replayed by edge_pipeline_runner.py (Stage 0 demo source)
//  Run:  scripts/run_rtl_sim.sh      (add +vcd for a waveform: sim/spikeguard.vcd)
module tb_spikeguard_top;
    localparam integer CPB = 59;                 // must match UART_CYCLES_PER_BIT

    reg        clk = 1'b0;
    reg        rst_n = 1'b0;
    reg  [2:0] trig_flags = 3'b000;
    wire       uart_tx, hw_irq;

    spikeguard_top dut (.clk(clk), .rst_n(rst_n), .trig_flags(trig_flags),
                        .uart_tx(uart_tx), .hw_irq(hw_irq));

    always #18.5185 clk = ~clk;                   // 27 MHz

    // ---------------- scenario bookkeeping ----------------
    reg [8*24-1:0] cur_scen  = "boot";
    reg [8*24-1:0] tick_scen = "boot";            // scenario that owned the window that just closed
    reg [8*24-1:0] last_sc   = "none";
    always @(posedge dut.window_tick) tick_scen <= cur_scen;

    integer fd, errors, overruns, idx_in_scen;
    integer st_n, st_min, st_max, st_peak_max;
    reg     st_any_irq;
    always @(posedge clk) if (dut.tx_overrun) begin overruns = overruns + 1; end

    // ---------------- UART receiver model ----------------
    reg [7:0] fb [0:11];
    reg [7:0] rxb;
    integer   n, k, spikes, peakv, delta, winus;
    reg [7:0] crc;
    reg       irq_at_sof;

    task uart_rx_byte;
        begin
            @(negedge uart_tx);                           // start bit
            repeat (CPB + CPB/2) @(posedge clk);          // middle of data bit 0
            for (n = 0; n < 8; n = n + 1) begin
                rxb[n] = uart_tx;
                repeat (CPB) @(posedge clk);
            end
            if (uart_tx !== 1'b1) begin
                errors = errors + 1; $display("ERROR: stop bit not high");
            end
        end
    endtask

    initial begin : monitor
        forever begin
            uart_rx_byte; fb[0] = rxb;
            irq_at_sof = hw_irq;
            for (k = 1; k < 12; k = k + 1) begin uart_rx_byte; fb[k] = rxb; end

            crc = 8'h00;
            for (k = 1; k <= 9; k = k + 1) crc = crc ^ fb[k];
            spikes = {fb[3], fb[2]};  peakv = {fb[5], fb[4]};
            delta  = {fb[7], fb[6]};  winus = {fb[9], fb[8]};

            if (fb[0] !== 8'hA5 || fb[11] !== 8'h5A) begin errors = errors + 1; $display("ERROR: bad SOF/EOF"); end
            if (crc !== fb[10])                      begin errors = errors + 1; $display("ERROR: CRC mismatch"); end
            if (winus !== 1000)                      begin errors = errors + 1; $display("ERROR: window field %0d", winus); end
            if (irq_at_sof !== (spikes >= 15))       begin errors = errors + 1;
                $display("ERROR: hw_irq=%0d but spike_count=%0d (frame/IRQ misaligned)", irq_at_sof, spikes); end

            if (tick_scen !== last_sc) begin idx_in_scen = 0; last_sc = tick_scen; end
            else idx_in_scen = idx_in_scen + 1;

            $fdisplay(fd, "%0s,%0d,%0d,%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
                      tick_scen, idx_in_scen, irq_at_sof,
                      fb[0],fb[1],fb[2],fb[3],fb[4],fb[5],fb[6],fb[7],fb[8],fb[9],fb[10],fb[11]);

            if (idx_in_scen >= 1 && tick_scen === cur_scen) begin   // skip 1st window (state carried over) and frames of the previous scenario
                st_n = st_n + 1;
                if (spikes < st_min) st_min = spikes;
                if (spikes > st_max) st_max = spikes;
                if (peakv  > st_peak_max) st_peak_max = peakv;
                if (irq_at_sof) st_any_irq = 1'b1;
            end
        end
    end

    // ---------------- scenario helpers ----------------
    task stats_reset;
        begin st_n = 0; st_min = 1000000; st_max = 0; st_peak_max = 0; st_any_irq = 1'b0; end
    endtask

    task begin_scen; input [8*24-1:0] name; input [2:0] flags;
        begin
            @(posedge dut.window_tick); #1;
            cur_scen = name; trig_flags = flags; stats_reset;
        end
    endtask

    task check_steady;
        input [8*24-1:0] name; input integer lo; input integer hi; input exp_irq;
        begin
            #400000;                                       // let the last frame finish (0.4 ms)
            $display("%-14s frames=%0d spikes=[%0d..%0d] peak_raw=%0d irq=%0d  (expect %0d..%0d, irq=%0d)",
                     name, st_n, st_min, st_max, st_peak_max, st_any_irq, lo, hi, exp_irq);
            if (st_n < 4)                      begin errors = errors + 1; $display("  FAIL: frames dropped"); end
            if (st_min < lo || st_max > hi)    begin errors = errors + 1; $display("  FAIL: spike range"); end
            if (st_any_irq !== exp_irq)        begin errors = errors + 1; $display("  FAIL: hw_irq"); end
        end
    endtask

    task steady; input [8*24-1:0] name; input [2:0] flags; input integer lo; input integer hi; input exp_irq;
        begin
            begin_scen(name, flags);
            repeat (5) @(posedge dut.window_tick);
            check_steady(name, lo, hi, exp_irq);
        end
    endtask

    task burst; input [8*24-1:0] name; input [2:0] flags; input integer us;
                input integer max_lo; input integer max_hi; input exp_irq;
        begin
            begin_scen(name, 3'b000);
            repeat (2) @(posedge dut.window_tick);
            #50000; trig_flags = flags; #(us * 1000); trig_flags = 3'b000;
            repeat (2) @(posedge dut.window_tick);
            #400000;
            $display("%-14s frames=%0d max_spikes=%0d peak_raw=%0d irq=%0d  (expect max %0d..%0d, irq=%0d)",
                     name, st_n, st_max, st_peak_max, st_any_irq, max_lo, max_hi, exp_irq);
            if (st_max < max_lo || st_max > max_hi) begin errors = errors + 1; $display("  FAIL: spike range"); end
            if (st_any_irq !== exp_irq)             begin errors = errors + 1; $display("  FAIL: hw_irq"); end
        end
    endtask

    // ---------------- main ----------------
    initial begin
        errors = 0; overruns = 0; idx_in_scen = 0; stats_reset;
        fd = $fopen("sim/uart_frames.csv", "w");
        $fdisplay(fd, "scenario,idx,hw_irq,frame_hex");
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/spikeguard.vcd");
            $dumpvars(1, tb_spikeguard_top);
            $dumpvars(0, dut.u_lif);
        end

        repeat (10) @(posedge clk);  rst_n = 1'b1;

        $display("---- SpikeGuard Stage-0 RTL regression (27 MHz, 100 kHz LIF, 1 ms window, 460800 baud) ----");
        steady("idle",         3'b000,  0,  0, 1'b0);
        steady("acoustic",     3'b001,  9, 12, 1'b0);   // sustained acoustic alone stays below IRQ (15)
        steady("vibration",    3'b010, 18, 21, 1'b1);
        steady("current",      3'b100, 23, 26, 1'b1);
        steady("all_channels", 3'b111, 48, 50, 1'b1);   // physical maximum = 50 spikes / ms

        // isolated 10 us glitch: leaky integration must NOT fire or raise the IRQ
        begin_scen("glitch_10us", 3'b000);
        repeat (2) @(posedge dut.window_tick);
        #300000; trig_flags = 3'b100; #10000; trig_flags = 3'b000;
        repeat (2) @(posedge dut.window_tick);
        #400000;
        $display("%-14s frames=%0d max_spikes=%0d peak_raw=%0d irq=%0d  (expect 0 spikes, irq=0, peak>0)",
                 "glitch_10us", st_n, st_max, st_peak_max, st_any_irq);
        if (st_max != 0 || st_any_irq || st_peak_max == 0) begin errors = errors + 1; $display("  FAIL: glitch"); end

        burst("burst_0p3ms", 3'b010, 300, 4, 8, 1'b0);   // short vibration burst: seen, below IRQ
        burst("burst_0p9ms", 3'b010, 900, 15, 21, 1'b1); // sustained burst: raises hw_irq

        $display("UART overruns (dropped frames): %0d", overruns);
        if (overruns != 0) errors = errors + 1;
        $fclose(fd);
        if (errors == 0) $display("RESULT: PASS"); else $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
