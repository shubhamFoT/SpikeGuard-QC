// 12-Byte Binary UART Packetizer (115200 8N1 @ 27 MHz)
// Frame: [0xA5][FLAGS][SPIKES_LE][PEAK_V_LE][DELTA_LE][WIN_US_LE][XOR_CRC][0x5A]
module snn_uart_packetizer #(
    parameter [15:0] CYCLES_PER_BIT = 16'd234 // 27,000,000 / 115,200 = 234.375
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        send_trigger, // 1-cycle pulse every 1 ms
    input  wire [2:0]  trig_flags,   // {current_spike, vibration, acoustic}
    input  wire [15:0] spike_count,  // Total LIF spikes in 1 ms window
    input  wire [15:0] peak_v_q88,   // Peak membrane potential (Q8.8)
    input  wire [15:0] delta_rate,   // Delta-modulator toggle count
    output reg         uart_tx,      // Connect to Arm SoC UART RX
    output reg         tx_busy
);

    reg [7:0]  packet [0:11];
    reg [3:0]  byte_idx;
    reg [3:0]  bit_idx;
    reg [15:0] baud_cnt;

    wire [7:0] flags_byte = {5'b00000, trig_flags};
    wire [7:0] crc_calc   = flags_byte ^
                            spike_count[7:0] ^ spike_count[15:8] ^
                            peak_v_q88[7:0]  ^ peak_v_q88[15:8]  ^
                            delta_rate[7:0]  ^ delta_rate[15:8]  ^
                            8'hE8            ^ 8'h03; // Window = 1000 us (0x03E8)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx  <= 1'b1;
            tx_busy  <= 1'b0;
            byte_idx <= 4'd0;
            bit_idx  <= 4'd0;
            baud_cnt <= 16'd0;
        end else if (!tx_busy) begin
            uart_tx <= 1'b1;
            if (send_trigger) begin
                packet[0]  <= 8'hA5;              // SOF
                packet[1]  <= flags_byte;         // Trigger flags
                packet[2]  <= spike_count[7:0];   // Spike count LSB
                packet[3]  <= spike_count[15:8];  // Spike count MSB
                packet[4]  <= peak_v_q88[7:0];    // Peak Vm LSB
                packet[5]  <= peak_v_q88[15:8];   // Peak Vm MSB
                packet[6]  <= delta_rate[7:0];    // Delta rate LSB
                packet[7]  <= delta_rate[15:8];   // Delta rate MSB
                packet[8]  <= 8'hE8;              // 1000 us LSB
                packet[9]  <= 8'h03;              // 1000 us MSB
                packet[10] <= crc_calc;           // XOR CRC over bytes 1..9
                packet[11] <= 8'h5A;              // EOF

                tx_busy  <= 1'b1;
                byte_idx <= 4'd0;
                bit_idx  <= 4'd0;
                baud_cnt <= 16'd0;
            end
        end else begin
            if (baud_cnt < CYCLES_PER_BIT - 1'b1) begin
                baud_cnt <= baud_cnt + 1'b1;
            end else begin
                baud_cnt <= 16'd0;
                if (bit_idx == 4'd0) begin
                    uart_tx <= 1'b0; // Start bit
                    bit_idx <= 4'd1;
                end else if (bit_idx >= 4'd1 && bit_idx <= 4'd8) begin
                    uart_tx <= packet[byte_idx][bit_idx - 4'd1]; // LSB first
                    bit_idx <= bit_idx + 1'b1;
                end else if (bit_idx == 4'd9) begin
                    uart_tx <= 1'b1; // Stop bit
                    bit_idx <= 4'd10;
                end else begin
                    bit_idx <= 4'd0;
                    if (byte_idx < 4'd11) begin
                        byte_idx <= byte_idx + 1'b1;
                    end else begin
                        tx_busy <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
