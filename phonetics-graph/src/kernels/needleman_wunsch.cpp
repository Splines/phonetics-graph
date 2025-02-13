extern "C" __global__ void needleman_wunsch(float *out)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Row & column (both range from 0 to n-1)
    // note that z is injected as a constant by the Rust code
    unsigned int row = floor(z - sqrtf(z * z - 2 * idx));
    unsigned int s = row * (z - row / 2);
    unsigned int col = row + idx - s;

    out[idx] = row * 100 + col; // dummy
}