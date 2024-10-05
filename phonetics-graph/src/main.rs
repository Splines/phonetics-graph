use rayon::prelude::*;
use rayon_progress::ProgressAdaptor;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

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

    let result = Arc::new(Mutex::new(Vec::new()));

    // Calculate score for every pair of words in parallel
    let it = ProgressAdaptor::new(0..1);
    let progress = it.items_processed();
    let total = it.len();

    rayon::spawn({
        let result_clone = Arc::clone(&result);
        move || {
            it.for_each(|i| {
                let word1 = &words[i];
                // println!("▶ Word {i}: {word1:?}");
                for j in i..words.len() {
                    let score =
                        needleman_wunsch::calculate_score(word1, &words[j], &similarity_matrix, -1);
                    let mut result_write = result_clone.lock().unwrap();
                    result_write.push((i, j, score));
                }
            });
        }
    });

    loop {
        let percent = (progress.get() * 100) / total;
        println!("🕖 Processing... {}% complete", percent);
        std::thread::sleep(std::time::Duration::from_secs(3));

        if progress.get() == total {
            println!("✅ Calculations done");
            break;
        }
    }

    // Write results to edges.csv
    let num_entries = result.lock().unwrap().len();
    println!("📝 Writing results to edges.csv (number of entries: {num_entries})");
    let output = result.lock().unwrap();
    let mut file = File::create("../data/ipa/graph-rust/edges.csv").expect("Failed to create edges.csv");
    writeln!(file, "source,target,weight").expect("Failed to write header to edges.csv");
    for (i, j, score) in output.iter() {
        writeln!(file, "{},{},{}", i, j, score).expect("Failed to write to edges.csv");
    }
    println!("✅ Done! Results written to edges.csv");
}
