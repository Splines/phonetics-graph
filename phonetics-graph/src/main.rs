use std::fs::File;
use std::io::{self, BufRead};

mod needleman_wunsch;

fn read_words_from_csv(file_path: &str) -> io::Result<Vec<Vec<u8>>> {
    let mut words = Vec::new();
    let file = File::open(file_path)?;
    for line in io::BufReader::new(file).lines() {
        let line = line?;
        let word: Vec<u8> = line
            .split(',')
            .map(|s| s.trim().parse().expect("Failed to parse integer"))
            .collect();
        words.push(word);
    }
    Ok(words)
}

fn main() {
    let words = read_words_from_csv("../data/ipa/fr_FR_words_symbols.csv")
        .expect("Failed to read words from CSV");
    let a = &words[58320];
    let b = &words[3634];

    println!("Word 1: {:?}", a);
    println!("Word 2: {:?}", b);

    // Create a dummy similarity matrix with 1 on the diagonal and -1 elsewhere
    let mut similarity_matrix = vec![vec![-1; 39]; 39];
    for i in 0..39 {
        similarity_matrix[i][i] = 1;
    }

    let score = needleman_wunsch::calculate_score(a, b, &similarity_matrix, -1);
    println!("Score: {}", score);
}
