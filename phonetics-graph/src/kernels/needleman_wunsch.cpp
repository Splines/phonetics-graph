typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

extern "C" __global__ void needleman_wunsch(
    float *out, uint8_t *words_flat, uint8_t *words_lengths, const uint32_t num_nodes)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Row & column (both range from 0 to n-1)
    // note that z is injected as a constant by the Rust code
    unsigned int row = floor(z - sqrtf(z * z - 2 * idx));
    unsigned int s = row * (z - row / 2);
    unsigned int col = row + idx - s;

    // out[idx] = row * 100 + col; // dummy
    // out[0] = words_flat[0];
    out[0] = num_nodes;
}