typedef char i8;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long int u64;

static const i8 GAP_PENALTY = -1;

/**
 * Calculates the alignment score between two words using the Needleman–Wunsch algorithm.
 */
__device__ i8 calculateScore(u8 *a, u8 a_length, u8 *b, u8 b_length, i8 *score_matrix)
{
    for (int i = 0; i <= a_length; ++i)
    {
        score_matrix[i * (b_length + 1)] = GAP_PENALTY * i;
    }
    for (int j = 0; j <= b_length; ++j)
    {
        score_matrix[j] = GAP_PENALTY * j;
    }

    for (int i = 1; i <= a_length; ++i)
    {
        for (int j = 1; j <= b_length; ++j)
        {
            i8 cost = (a[i - 1] == b[j - 1]) ? 1 : -1; // default metric
            i8 match_score = score_matrix[(i - 1) * (b_length + 1) + (j - 1)] + cost;
            i8 delete_score = score_matrix[(i - 1) * (b_length + 1) + j] + GAP_PENALTY;
            i8 insert_score = score_matrix[i * (b_length + 1) + (j - 1)] + GAP_PENALTY;
            i8 new_score = max(max(match_score, delete_score), insert_score);
            score_matrix[i * (b_length + 1) + j] = new_score;
        }
    }

    return score_matrix[a_length * (b_length + 1) + b_length];
}

extern "C" __global__ void needleman_wunsch(
    i8 *out, u8 *words_flat, u64 *words_offsets,
    const u32 num_nodes, const u64 num_edges, const u32 max_word_length)
{
    extern __shared__ i8 shared_score_matrix[];

    u64 idx = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_edges)
    {
        return;
    }

    double z = num_nodes - 0.5;
    u64 row = floor(z - sqrt(z * z - 2 * idx));
    u64 col = 0.5 * row * row + row * (1.5 - num_nodes) + idx + 1;

    // just for debugging
    // if (row >= num_nodes || col >= num_nodes)
    // {
    //     printf("Invalid row or col index: row=%u, col=%u\n", row, col);
    //     return;
    // }

    u8 *word1 = words_flat + words_offsets[row];
    u8 word1_length = words_offsets[row + 1] - words_offsets[row];
    u8 *word2 = words_flat + words_offsets[col];
    u8 word2_length = words_offsets[col + 1] - words_offsets[col];

    i8 *score_matrix = shared_score_matrix + threadIdx.x * (max_word_length + 1) * (max_word_length + 1);
    i8 distance = calculateScore(word1, word1_length, word2, word2_length, score_matrix);
    out[idx] = distance;
}
