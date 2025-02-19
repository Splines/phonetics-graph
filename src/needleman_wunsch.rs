type SimilarityMatrix = Vec<Vec<i8>>;

pub fn calculate_score(
    a: &Vec<u8>, 
    b: &Vec<u8>,
    similarity_matrix: &SimilarityMatrix,
    gap_penalty: i8,
) -> i8 {
    let mut score_matrix = vec![vec![0; b.len() + 1]; a.len() + 1];
    for i in 0..a.len() + 1 {
        score_matrix[i][0] = gap_penalty * i as i8;
    }
    for j in 0..b.len() + 1 {
        score_matrix[0][j] = gap_penalty * j as i8;
    }
    for i in 1..a.len() + 1 {
        for j in 1..b.len() + 1 {
            let cost = similarity_matrix[a[i - 1] as usize][b[j - 1] as usize];
            let match_score = score_matrix[i - 1][j - 1] + cost;
            let delete_score = score_matrix[i - 1][j] + gap_penalty;
            let insert_score = score_matrix[i][j - 1] + gap_penalty;
            score_matrix[i][j] = match_score.max(delete_score).max(insert_score);
        }
    }
    score_matrix[a.len()][b.len()]
}
