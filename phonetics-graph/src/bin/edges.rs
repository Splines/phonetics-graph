use std::fs::{self, File};
use std::io::{self, BufRead, Write};

use cudarc::driver::{LaunchAsync, LaunchConfig};

static KERNEL_FILE: &str = "./src/kernels/needleman_wunsch.cpp";
static MODULE_NAME: &str = "phonetics_module";
static KERNEL_NAME: &str = "needleman_wunsch";

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

fn compute(mut words: Vec<Vec<u8>>) -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("Device name: {}", dev.name()?);

    // Available memory
    let dev_unsafe: cudarc::driver::sys::CUdevice = cudarc::driver::result::device::get(0)?;
    let total_memory = unsafe { cudarc::driver::result::device::total_mem(dev_unsafe) }?;
    println!("Total global memory: {} MB", total_memory / (1024 * 1024));

    // Max number of nodes to not exceed available memory
    let available_memory = 0.7 * total_memory as f64; // a bit less than total memory
    let score_size = std::mem::size_of::<i8>() as f64;
    let max_num_edges = (available_memory / score_size) as u64;
    let max_num_nodes = -0.5 + f64::sqrt(0.25 + 2.0 * max_num_edges as f64);
    let threshold: u32 = max_num_nodes as u32;
    words = words.into_iter().take(threshold as usize).collect();
    println!("Threshold (num words): {}", threshold);

    let num_nodes: u32 = words.len().try_into().unwrap();
    let num_adjacency_matrix_elements: u64 =
        (u64::from(num_nodes) * (u64::from(num_nodes) + 1)) / 2;
    println!("num_adjacency_matrix_elements: {num_adjacency_matrix_elements}");

    let words_flat: Vec<u8> = words.iter().flat_map(|w| w.iter()).copied().collect();
    let mut words_offsets: Vec<u64> = words
        .iter()
        .scan(0, |acc, w| {
            let start = *acc;
            *acc += w.len();
            Some(start.try_into().unwrap())
        })
        .collect();
    words_offsets.push(words_flat.len() as u64); // add the last offset
    println!("Num words: {}", words.len());
    assert!(words_offsets.len() == words.len() + 1);

    let words_flat_device = dev.htod_copy(words_flat)?;
    let words_offsets_device = dev.htod_copy(words_offsets)?;

    println!("Allocating buffers (for output writing)");
    println!(
        "Buffer size (in MB): {}",
        (num_adjacency_matrix_elements * std::mem::size_of::<i8>() as u64) / (1024 * 1024)
    );
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
    let grid_size: u32 = ((num_adjacency_matrix_elements + block_size as u64 - 1)
        / block_size as u64)
        .try_into()
        .expect("Failed to convert grid size to u32");
    let max_grid_size =
        dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?;
    println!("Max grid size: {max_grid_size}");
    if grid_size > max_grid_size.try_into().unwrap() {
        panic!("Grid size exceeds maximum grid size");
    }

    // assert that grid size and block size calculation are correct
    if (grid_size as u64 * block_size as u64) < num_adjacency_matrix_elements {
        panic!(
            "Grid size ({}) * block size ({}) is less than the number of adjacency matrix elements ({})",
            grid_size, block_size, num_adjacency_matrix_elements
        );
    }

    println!("Grid size: {grid_size}");
    println!("Shared memory size (in bytes): {}", shared_mem_size);
    println!("Shared memory size (in kB): {}", shared_mem_size / 1024);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem_size,
    };

    let mut kernel_code = fs::read_to_string(KERNEL_FILE).unwrap();
    kernel_code = format!("static const double z = {num_nodes}.5;\n") + &kernel_code;
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

    if let Err(e) = compute(words) {
        eprintln!("Error: {}", e);
    }
}
