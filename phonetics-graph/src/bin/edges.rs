use std::fs::{self, File};
use std::io::{self, BufRead};

use cudarc::driver::{LaunchAsync, LaunchConfig};

static KERNEL_FILE: &str = "./src/kernels/needleman_wunsch.cpp";
static MODULE_NAME: &str = "phonetics_module";
static KERNEL_NAME: &str = "needleman_wunsch";

static THRESHOLD: u8 = 100;

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

fn compute(words: Vec<Vec<u8>>) -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("Device name: {}", dev.name()?);

    let num_nodes: usize = 5;
    let num_adjacency_matrix_elements: usize = (num_nodes * (num_nodes + 1)) / 2;
    println!("num_adjacency_matrix_elements: {num_adjacency_matrix_elements}");

    let words_flat: Vec<u8> = words.iter().flat_map(|w| w.iter()).copied().collect();
    let words_offsets: Vec<usize> = words
        .iter()
        .scan(0, |acc, w| {
            let start = *acc;
            *acc += w.len();
            Some(start)
        })
        .collect();
    println!("{:?}", words_offsets);

    let words_flat_device = dev.htod_copy(words_flat)?;
    let words_offsets_device = dev.htod_copy(words_offsets)?;

    println!("Allocating buffers");
    let mut out = dev.alloc_zeros::<f32>(num_adjacency_matrix_elements)?;

    let mut kernel_code = fs::read_to_string(KERNEL_FILE).unwrap();
    kernel_code = format!("static const float z = {num_nodes}.5;\n") + &kernel_code;
    let ptx = cudarc::nvrtc::compile_ptx(kernel_code)?;

    println!("Loading PTX");
    dev.load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME])?;
    let kernel = dev.get_func(MODULE_NAME, KERNEL_NAME).unwrap();
    let cfg = LaunchConfig::for_num_elems(num_adjacency_matrix_elements as u32);

    println!("Launching kernel");
    unsafe {
        kernel.launch(
            cfg,
            (
                &mut out,
                &words_flat_device,
                &words_offsets_device,
                words.len() as u32,
            ),
        )
    }?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    assert_eq!(out_host.len(), num_adjacency_matrix_elements);
    println!("{:?}", out_host);

    println!("Done");
    Ok(())
}

fn main() {
    let words = read_words_from_csv("../data/graph/french-phonetics-integers.txt")
        .expect("Failed to read words from CSV");
    println!("Num words: {}", words.len());

    // only consider the first THRESHOLD words
    let words_shortened: Vec<Vec<u8>> = words.into_iter().take(THRESHOLD as usize).collect();
    assert_eq!(words_shortened.len(), THRESHOLD as usize);

    if let Err(e) = compute(words_shortened) {
        eprintln!("Error: {}", e);
    }
}
