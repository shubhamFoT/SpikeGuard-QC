// Top-Level SpikeGuard Stage 0 Hardware Anomaly Trigger
// Clock: 27 MHz | LIF Sample Rate: 100 kHz | UART Telemetry Window: 1 ms
module spikeguard_top (
    input  wire       clk,        // 27 MHz onboard oscillator
    input  wire       rst_n,      // Active-low reset button
    input  wire [2:0] trig_flags, // [0]: Acoustic, [1]: Vibration, [2]: Current Spike
    output wire       uart_tx,    // 115200 8N1 telemetry to Arm SoC
    output reg        hw_irq      // Active-high anomaly interrupt to Arm SoC GPIO
);

    // 1. Sample tick (100 kHz = 270 cycles @ 27 MHz) & 1 ms window (27,000 cycles)
    reg [8:0]  sample_div;
    reg [14:0] window_cnt;
    reg        sample_ce;
    reg        window_tick;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_div  <= 9'd0;
            sample_ce   <= 1'b0;
            window_cnt  <= 15'd0;
            window_tick <= 1'b0;
        end else begin
            if (sample_div >= 9'd269) begin
                sample_div <= 9'd0;
                sample_ce  <= 1'b1;
            end else begin
                sample_div <= sample_div + 1'b1;
                sample_ce  <= 1'b0;
            end

            if (window_cnt >= 15'd26999) begin
                window_cnt  <= 15'd0;
                window_tick <= 1'b1;
            end else begin
                window_cnt  <= window_cnt + 1'b1;
                window_tick <= 1'b0;
            end
        end
    end

    // 2. Weighted delta input current (Q8.8) from comparator channels
    wire signed [15:0] i_synaptic =
        (trig_flags[0] ? 16'sh0C00 : 16'sh0000) + // Acoustic weight (+12.0)
        (trig_flags[1] ? 16'sh1200 : 16'sh0000) + // Vibration weight (+18.0)
        (trig_flags[2] ? 16'sh1800 : 16'sh0000);  // Current spike weight (+24.0)

    wire        lif_spike;
    wire [15:0] lif_vm;

    lif_neuron_core u_lif (
        .clk        (clk),
        .rst_n      (rst_n),
        .sample_ce  (sample_ce),
        .i_in       (i_synaptic),
        .spike_out  (lif_spike),
        .v_membrane (lif_vm)
    );

    // 3. 1 ms window accumulators & edge-toggle counter
    reg [2:0]  prev_flags;
    reg [15:0] spike_acc, spike_latched;
    reg [15:0] peak_v_acc, peak_v_latched;
    reg [15:0] delta_acc, delta_latched;
    reg [2:0]  flags_latched;

    wire [2:0] flag_toggles = trig_flags ^ prev_flags;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_flags     <= 3'b000;
            spike_acc      <= 16'd0;
            peak_v_acc     <= 16'd0;
            delta_acc      <= 16'd0;
            spike_latched  <= 16'd0;
            peak_v_latched <= 16'd0;
            delta_latched  <= 16'd0;
            flags_latched  <= 3'b000;
            hw_irq         <= 1'b0;
        end else begin
            prev_flags <= trig_flags;

            if (window_tick) begin
                spike_latched  <= spike_acc;
                peak_v_latched <= peak_v_acc;
                delta_latched  <= delta_acc;
                flags_latched  <= trig_flags;
                hw_irq         <= (spike_acc >= 16'd15); // Fire Arm interrupt on burst

                spike_acc      <= 16'd0;
                peak_v_acc     <= 16'd0;
                delta_acc      <= 16'd0;
            end else begin
                if (lif_spike)
                    spike_acc <= spike_acc + 1'b1;
                if (lif_vm > peak_v_acc)
                    peak_v_acc <= lif_vm;
                if (|flag_toggles)
                    delta_acc <= delta_acc + 1'b1;
            end
        end
    end

    // 4. UART 12-Byte Packetizer
    wire tx_busy;
    snn_uart_packetizer u_uart (
        .clk          (clk),
        .rst_n        (rst_n),
        .send_trigger (window_tick),
        .trig_flags   (flags_latched),
        .spike_count  (spike_latched),
        .peak_v_q88   (peak_v_latched),
        .delta_rate   (delta_latched),
        .uart_tx      (uart_tx),
        .tx_busy      (tx_busy)
    );

endmodule