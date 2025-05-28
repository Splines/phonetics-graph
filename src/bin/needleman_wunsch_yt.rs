pub fn calculate_score(a: &Vec<u8>, b: &Vec<u8>, gap_penalty: i8) -> i8 {
    let mut score_matrix = vec![vec![0; b.len() + 1]; a.len() + 1];

    for i in 0..a.len() + 1 {
        score_matrix[i][0] = gap_penalty * i as i8;
    }

    for j in 0..b.len() + 1 {
        score_matrix[0][j] = gap_penalty * j as i8;
    }

    for i in 1..a.len() + 1 {
        for j in 1..b.len() + 1 {
            let diag_score_delta = if a[i - 1] == b[j - 1] { 1 } else { -1 };
            let from_diagonal_score = score_matrix[i - 1][j - 1] + diag_score_delta;
            let from_left_score = score_matrix[i][j - 1] + gap_penalty;
            let from_top_score = score_matrix[i - 1][j] + gap_penalty;
            score_matrix[i][j] = from_diagonal_score.max(from_left_score).max(from_top_score);
        }
    }

    score_matrix[a.len()][b.len()]
}

fn main() {
    // Here just a simple example by hand since Unicode characters like ɑ̃ need
    // special treatment such that they are not split into multiple chars.
    // Of course, this is autoamted in the real code.
    let a = Vec::from([0, 1, 2, 3, 4, 5]); // "pɥisɑ̃s"
    let b = Vec::from([6, 1, 4, 5]); // "nɥɑ̃s"

    let gap_penalty = -2;
    let score = calculate_score(&a, &b, gap_penalty);
    println!("Score for {a:?} and {b:?}: {score}");
}
