use rayon::prelude::*;
use rayon_progress::ProgressAdaptor;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

mod needleman_wunsch;

static OUTPUT_FILE: &str = "../data/graph/final/edges.cpu.bin";
static MAX_WORDS: usize = 10_000;

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
    let words = read_words_from_csv("../data/graph/french-phonetics-integers.txt")
        .expect("Failed to read words from CSV");

    // Create a dummy similarity matrix with 1 on the diagonal and -1 elsewhere
    let mut similarity_matrix = vec![vec![-1; 39]; 39];
    for i in 0..39 {
        similarity_matrix[i][i] = 1;
    }

    let result = Arc::new(Mutex::new(Vec::new()));

    let it = ProgressAdaptor::new(0..MAX_WORDS);
    let progress = it.items_processed();
    let total = it.len();

    let start = std::time::Instant::now();

    rayon::spawn({
        let result_clone = Arc::clone(&result);
        move || {
            it.for_each(|i| {
                let word1 = &words[i];
                for j in (i + 1)..MAX_WORDS {
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
            println!("💻 Calculations done");
            break;
        }
    }

    let duration = start.elapsed();
    println!("⏱️  Time for calculations: {}s", duration.as_secs());

    // Sort the results
    println!("🔍 Sorting results...");
    let mut output = result.lock().unwrap();
    // Due to the parallel processing, the results are not sorted, e.g.
    // 0,1,score -> 0,3,score -> 0,2,score -> 1,3,score -> 1,1,score -> 1,2,score
    // We want to sort it like this:
    // 0,1,score -> 0,2,score -> 0,3,score -> 1,1,score -> 1,2,score -> 1,3,score
    output.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    println!("🔍 Results sorted");

    // Print highest and lowest score
    let highest_score = output.iter().max_by_key(|(_, _, score)| *score).unwrap();
    let lowest_score = output.iter().min_by_key(|(_, _, score)| *score).unwrap();
    println!("💠 Highest score: {highest_score:?}");
    println!("💠 Lowest score: {lowest_score:?}");

    // Extract weights and write to edges.bin
    let weights: Vec<u8> = output.iter().map(|&(_, _, weight)| weight as u8).collect();
    let num_entries = weights.len();
    println!("📝 Writing results to {OUTPUT_FILE} (#entries: {num_entries})");
    let mut file = File::create(OUTPUT_FILE).expect("Failed to create edges.bin");
    file.write_all(&weights)
        .expect("Failed to write to {OUTPUT_FILE}");
    println!("✅ Done! Results written to {OUTPUT_FILE}");
}
