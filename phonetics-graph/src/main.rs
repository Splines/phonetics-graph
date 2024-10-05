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

    // Create a dummy similarity matrix with 1 on the diagonal and -1 elsewhere
    let mut similarity_matrix = vec![vec![-1; 39]; 39];
    for i in 0..39 {
        similarity_matrix[i][i] = 1;
    }

    // Calculate score for every pair of words
    for i in 0..words.len() {
        let word1 = &words[i];
        println!("Word {i}: {word1:?}");
        for j in i + 1..words.len() {
            needleman_wunsch::calculate_score(word1, &words[j], &similarity_matrix, -1);
        }
    }
}
