typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

extern "C" __global__ void needleman_wunsch(
    float *out, uint8_t *words_flat, uint8_t *words_offsets, const uint32_t num_nodes)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Row & column (both range from 0 to n-1)
    // note that z is injected as a constant by the Rust code
    unsigned int row = floor(z - sqrtf(z * z - 2 * idx));
    unsigned int s = row * (z - row / 2);
    unsigned int col = row + idx - s;

    uint8_t *word1 = words_flat + words_offsets[row];
    uint8_t word1_length = words_offsets[row + 1] - words_offsets[row];
    uint8_t *word2 = words_flat + words_offsets[col];
    uint8_t word2_length = words_offsets[col + 1] - words_offsets[col];

    // dummy assignment
    out[0] = word1[0];
    out[1] = word1[1];
}
