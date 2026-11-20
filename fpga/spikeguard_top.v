// Top-Level SpikeGuard Stage 0 Hardware Anomaly Trigger                  [FIXED]
// Clock 27 MHz | LIF sample 100 kHz | telemetry window 1 ms | UART 460800 8N1
//
// FIXES vs Phase-1 version
//  1. 2-FF synchronizer on the asynchronous comparator inputs (metastability / EMI).
//  2. The UART frame now carries the SAME window that raised hw_irq. Before, the
//     packetizer sampled *_latched in the very cycle they were being updated, so the
//     frame lagged hw_irq by one window. send_trigger is now delayed by one clock.
//  3. Packetizer baud/state fixes (see snn_uart_packetizer.v).
//
// Physical limits (100 LIF samples per 1 ms window), useful for the report:
//   acoustic only ~11 spikes/ms | vibration only ~20 | current only ~25 | all three = 50 (max)
module spikeguard_top #(
    parameter [15:0] UART_CYCLES_PER_BIT = 16'd59,
    parameter [15:0] IRQ_SPIKE_THRESHOLD = 16'd15
)(
    input  wire       clk,          // 27 MHz onboard oscillator
    input  wire       rst_n,        // active-low reset
    input  wire [2:0] trig_flags,   // [0] acoustic, [1] vibration, [2] current spike
    output wire       uart_tx,      // 460800 8N1 telemetry
    output reg        hw_irq        // active-high anomaly interrupt
);
    // 0. input synchronizer (fix #1)
    reg [2:0] trig_meta, trig_sync;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin trig_meta <= 3'b000; trig_sync <= 3'b000; end
        else        begin trig_meta <= trig_flags; trig_sync <= trig_meta; end
    end

    // 1. sample tick (270 cycles = 100 kHz) and 1 ms window (27,000 cycles)
    reg [8:0]  sample_div;
    reg [14:0] window_cnt;
    reg        sample_ce, window_tick, window_tick_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_div <= 9'd0;  sample_ce <= 1'b0;
            window_cnt <= 15'd0; window_tick <= 1'b0; window_tick_d <= 1'b0;
        end else begin
            window_tick_d <= window_tick;
            if (sample_div >= 9'd269) begin sample_div <= 9'd0; sample_ce <= 1'b1; end
            else begin sample_div <= sample_div + 9'd1; sample_ce <= 1'b0; end

            if (window_cnt >= 15'd26999) begin window_cnt <= 15'd0; window_tick <= 1'b1; end
            else begin window_cnt <= window_cnt + 15'd1; window_tick <= 1'b0; end
        end
    end

    // 2. weighted synaptic current (Q8.8)
    wire signed [15:0] i_synaptic =
        (trig_sync[0] ? 16'sh0C00 : 16'sh0000) +   // acoustic +12.0
        (trig_sync[1] ? 16'sh1200 : 16'sh0000) +   // vibration +18.0
        (trig_sync[2] ? 16'sh1800 : 16'sh0000);    // current   +24.0

    wire        lif_spike;
    wire [15:0] lif_vm;

    lif_neuron_core u_lif (
        .clk(clk), .rst_n(rst_n), .sample_ce(sample_ce),
        .i_in(i_synaptic), .spike_out(lif_spike), .v_membrane(lif_vm)
    );

    // 3. 1 ms window accumulators
    reg [2:0]  prev_flags;
    reg [15:0] spike_acc,  spike_latched;
    reg [15:0] peak_v_acc, peak_v_latched;
    reg [15:0] delta_acc,  delta_latched;
    reg [2:0]  flags_latched;
    wire [2:0] flag_toggles = trig_sync ^ prev_flags;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_flags <= 3'b000;
            spike_acc <= 16'd0;  peak_v_acc <= 16'd0;  delta_acc <= 16'd0;
            spike_latched <= 16'd0; peak_v_latched <= 16'd0; delta_latched <= 16'd0;
            flags_latched <= 3'b000; hw_irq <= 1'b0;
        end else begin
            prev_flags <= trig_sync;
            if (window_tick) begin
                spike_latched  <= spike_acc;
                peak_v_latched <= peak_v_acc;
                delta_latched  <= delta_acc;
                flags_latched  <= trig_sync;
                hw_irq         <= (spike_acc >= IRQ_SPIKE_THRESHOLD);
                spike_acc <= 16'd0; peak_v_acc <= 16'd0; delta_acc <= 16'd0;
            end else begin
                if (lif_spike)           spike_acc  <= spike_acc + 16'd1;
                if (lif_vm > peak_v_acc) peak_v_acc <= lif_vm;
                if (|flag_toggles)       delta_acc  <= delta_acc + 16'd1;
            end
        end
    end

    // 4. UART packetizer — triggered ONE clock after the latches update (fix #2)
    wire tx_busy, tx_overrun;
    snn_uart_packetizer #(.CYCLES_PER_BIT(UART_CYCLES_PER_BIT)) u_uart (
        .clk(clk), .rst_n(rst_n),
        .send_trigger(window_tick_d),
        .trig_flags(flags_latched),
        .spike_count(spike_latched),
        .peak_v_q88(peak_v_latched),
        .delta_rate(delta_latched),
        .uart_tx(uart_tx), .tx_busy(tx_busy), .overrun(tx_overrun)
    );
endmodule
