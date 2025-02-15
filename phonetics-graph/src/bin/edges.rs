use std::fs::{self, File};
use std::io::{self, BufRead, Write};

use cudarc::driver::{LaunchAsync, LaunchConfig};

static KERNEL_FILE: &str = "./src/kernels/needleman_wunsch.cpp";
static MODULE_NAME: &str = "phonetics_module";
static KERNEL_NAME: &str = "needleman_wunsch";

static THRESHOLD: u32 = 611000;
// static THRESHOLD: u32 = 611786;

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

    let num_nodes: u32 = words.len().try_into().unwrap();
    let num_adjacency_matrix_elements: u32 = (num_nodes * (num_nodes + 1)) / 2;
    println!("num_adjacency_matrix_elements: {num_adjacency_matrix_elements}");

    let words_flat: Vec<u8> = words.iter().flat_map(|w| w.iter()).copied().collect();
    let mut words_offsets: Vec<u32> = words
        .iter()
        .scan(0, |acc, w| {
            let start = *acc;
            *acc += w.len();
            Some(start as u32)
        })
        .collect();
    words_offsets.push(words_flat.len() as u32); // add the last offset
    println!("Num words: {}", words.len());
    assert!(words_offsets.len() == words.len() + 1);

    let words_flat_device = dev.htod_copy(words_flat)?;
    let words_offsets_device = dev.htod_copy(words_offsets)?;

    println!("Allocating buffers");
    let mut out = dev.alloc_zeros::<i8>(num_adjacency_matrix_elements.try_into().unwrap())?;

    let max_word_length = words.iter().map(|w| w.len()).max().unwrap();
    println!("max_word_length: {}", max_word_length);

    // Launch config
    let max_shared_mem_bytes: u32 = dev
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
        )?
        .try_into()
        .unwrap();
    println!(
        "Max shared memory per block (in kB): {}",
        max_shared_mem_bytes / 1024
    );
    let mut block_size: u32 = 1;
    let mut shared_mem_size: u32 = 0;

    // adjust block size such that shared memory size of 48kB is not exceeded
    for bs in (1..=1024).rev() {
        let sm_size = bs
            * (max_word_length as u32 + 1)
            * (max_word_length as u32 + 1)
            * (std::mem::size_of::<i8>() as u32); // 1 byte

        if sm_size <= max_shared_mem_bytes {
            block_size = bs;
            shared_mem_size = sm_size;
            break;
        }
    }

    println!("Block size: {}", block_size);
    println!("Shared memory size (in bytes): {}", shared_mem_size);
    println!("Shared memory size (in kB): {}", shared_mem_size / 1024);

    let cfg = LaunchConfig {
        grid_dim: (
            (num_adjacency_matrix_elements + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0, // this is only dynamic shared memory size
    };

    let mut kernel_code = fs::read_to_string(KERNEL_FILE).unwrap();
    let z_assignment = format!("static const double z = {num_nodes}.5;\n");
    let shared_mem_assignment = format!("static const int shared_mem_size = {shared_mem_size};\n");
    kernel_code = z_assignment + &shared_mem_assignment + &kernel_code;
    let ptx = cudarc::nvrtc::compile_ptx(kernel_code)?;

    println!("Loading PTX");
    dev.load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME])?;
    let kernel = dev.get_func(MODULE_NAME, KERNEL_NAME).unwrap();

    println!("Launching kernel");
    let start = std::time::Instant::now();
    unsafe {
        kernel.launch(
            cfg,
            (
                &mut out,
                &words_flat_device,
                &words_offsets_device,
                num_nodes,
                num_adjacency_matrix_elements,
                max_word_length as u32,
            ),
        )
    }?;
    dev.synchronize()?;
    let duration = start.elapsed();
    println!("Kernel execution time: {:?}", duration);

    let out_host: Vec<i8> = dev.dtoh_sync_copy(&out)?;
    assert_eq!(
        out_host.len(),
        num_adjacency_matrix_elements.try_into().unwrap()
    );
    println!("{:?}", &out_host[..20]);
    println!("{:?}", &out_host[out_host.len() - 20..]);

    println!("Done");

    // Print highest and lowest score
    let highest_score = out_host.iter().max_by_key(|score| *score).unwrap();
    let lowest_score = out_host.iter().min_by_key(|score| *score).unwrap();
    println!("💠 Highest score: {highest_score:?}");
    println!("💠 Lowest score: {lowest_score:?}");

    // println!("📝 Writing results to edges.bin (#entries: {num_nodes})");
    // let mut file =
    //     File::create("../data/graph/edges-new-gpu.bin").expect("Failed to create edges.bin");
    // file.write_all(&out_host.iter().map(|&x| x as u8).collect::<Vec<u8>>())
    //     .expect("Failed to write to edges.bin");
    // println!("✅ Done! Results written to edges.bin");

    Ok(())
}

fn main() {
    let words = read_words_from_csv("../data/graph/french-phonetics-integers.txt")
        .expect("Failed to read words from CSV");
    println!("Num words: {}", words.len());

    // only consider the first THRESHOLD words
    let words_shortened: Vec<Vec<u8>> = words.into_iter().take(THRESHOLD as usize).collect();
    assert_eq!(words_shortened.len(), THRESHOLD as usize);
    println!("Threshold: {}", THRESHOLD);

    if let Err(e) = compute(words_shortened) {
        eprintln!("Error: {}", e);
    }
}
