// Synthesizable Leaky Integrate-and-Fire (LIF) Neuron Core (Q8.8 Fixed-Point)
// Target: Gowin GW1NR-9 (Tang Nano 9K)
module lif_neuron_core #(
    parameter signed [15:0] V_THRESHOLD = 16'sh4000, // 64.0 in Q8.8
    parameter signed [15:0] V_RESET     = 16'sh0000
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               sample_ce,    // 100 kHz sample enable tick
    input  wire signed [15:0] i_in,         // Weighted delta input current (Q8.8)
    output reg                spike_out,    // 1-cycle firing pulse
    output reg  [15:0]        v_membrane    // Peak membrane potential (Q8.8)
);

    reg signed [15:0] v_m;

    // Leaky integration: v_next = v_m - (v_m / 8) + i_in (17-bit to prevent overflow)
    wire signed [15:0] leak_sub = v_m >>> 3;
    wire signed [16:0] v_next   = v_m - leak_sub + i_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_m        <= 16'sh0000;
            spike_out  <= 1'b0;
            v_membrane <= 16'h0000;
        end else if (sample_ce) begin
            if (v_next >= V_THRESHOLD) begin
                spike_out  <= 1'b1;
                v_m        <= V_RESET;
                v_membrane <= v_next[15:0];
            end else begin
                spike_out  <= 1'b0;
                v_m        <= (v_next < 17'sd0) ? 16'sh0000 : v_next[15:0];
                v_membrane <= (v_next < 17'sd0) ? 16'h0000  : v_next[15:0];
            end
        end else begin
            spike_out <= 1'b0;
        end
    end

endmodule