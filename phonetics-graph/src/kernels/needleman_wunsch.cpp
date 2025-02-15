typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

static const int8_t GAP_PENALTY = -1;

/**
 * Calculates the distance between two words using the Needleman-Wunsch algorithm.
 */
__device__ int8_t calculateDistance(uint8_t *a, uint8_t a_length,
                                    uint8_t *b, uint8_t b_length,
                                    int8_t *score_matrix)
{
    // Populate matrix
    for (int i = 0; i <= a_length; ++i)
    {
        score_matrix[i * (b_length + 1)] = GAP_PENALTY * i;
    }
    for (int j = 0; j <= b_length; ++j)
    {
        score_matrix[j] = GAP_PENALTY * j;
    }

    // Calculate score using the Needleman-Wunsch algorithm
    for (int i = 1; i <= a_length; ++i)
    {
        for (int j = 1; j <= b_length; ++j)
        {
            int8_t cost = (a[i - 1] == b[j - 1]) ? 1 : -1; // default metric
            int8_t match_score = score_matrix[(i - 1) * (b_length + 1) + (j - 1)] + cost;
            int8_t delete_score = score_matrix[(i - 1) * (b_length + 1) + j] + GAP_PENALTY;
            int8_t insert_score = score_matrix[i * (b_length + 1) + (j - 1)] + GAP_PENALTY;
            score_matrix[i * (b_length + 1) + j] = max(max(match_score, delete_score), insert_score);
        }
    }
    return score_matrix[a_length * (b_length + 1) + b_length];
}

extern "C" __global__ void needleman_wunsch(
    int8_t *out, uint8_t *words_flat, uint32_t *words_offsets, const uint32_t out_size,
    int8_t *score_matrices, const uint32_t max_word_length)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size)
    {
        return;
    }

    // Row & column (both range from 0 to n-1)
    // (the variable "z" is injected as a constant by the Rust code)
    // row and column both refer to an actual word in the words_flat array
    unsigned int row = floor(z - sqrtf(z * z - 2 * idx));
    unsigned int s = row * (z - row / 2);
    unsigned int col = row + idx - s;

    uint8_t *word1 = words_flat + words_offsets[row];
    uint8_t word1_length = words_offsets[row + 1] - words_offsets[row];
    uint8_t *word2 = words_flat + words_offsets[col];
    uint8_t word2_length = words_offsets[col + 1] - words_offsets[col];

    int8_t *score_matrix = score_matrices + idx * (max_word_length + 1) * (max_word_length + 1);
    int8_t distance = calculateDistance(word1, word1_length, word2, word2_length, score_matrix);
    out[idx] = distance;
}
