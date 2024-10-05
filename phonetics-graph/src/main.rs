mod needleman_wunsch;

fn main() {
    let a = vec![0, 1, 2, 3];
    let b = vec![0, 1, 2, 3];

    let similarity_matrix = vec![
        vec![1, -1, -1, -1],
        vec![-1, 1, -1, -1],
        vec![-1, -1, 1, -1],
        vec![-1, -1, -1, 1],
    ];

    let score = needleman_wunsch::calculate_score(a, b, &similarity_matrix, -1);

    println!("Score: {}", score);
}
