use phonetics_graph::pretty::prettified_int;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::time::Duration;

type CudaDeviceArc = std::sync::Arc<cudarc::driver::CudaDevice>;
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

static DEVICE_ID: usize = 0;

static KERNEL_FILE: &str = "./src/kernels/needleman_wunsch.cpp";
static MODULE_NAME: &str = "phonetics_module";
static KERNEL_NAME: &str = "needleman_wunsch";

static INPUT_FILE: &str = "data/graph/french-phonetics-integers.txt";
static OUTPUT_FILE: &str = "data/graph/final/edges.gpu.bin";
static TIMING_FILE: &str = "data/graph/final/timing.csv";

/// Reads words from a CSV file and returns a vector of vectors of u8.
/// Each inner vector represents a word.
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

/// Initializes the device and prints the device name.
fn initialize_device() -> CudaDeviceArc {
    let dev = cudarc::driver::CudaDevice::new(DEVICE_ID).expect("Failed to initialize device");
    println!(
        "Device name: {}",
        dev.name().expect("Failed to get device name")
    );
    dev
}

/// Returns the available memory on the device in bytes.
fn total_available_memory() -> usize {
    let dev_unsafe: cudarc::driver::sys::CUdevice =
        cudarc::driver::result::device::get(DEVICE_ID as i32).expect("Failed to get device");
    let total_memory = unsafe { cudarc::driver::result::device::total_mem(dev_unsafe) }
        .expect("Failed to retrieve total memory of device");
    let available = total_memory as f64 * 0.85;
    println!("Available memory: {} MB", available / (1024.0 * 1024.0));
    available as usize
}

/// Returns the number of undirected edges in a fully connected graph with
/// `num_nodes` nodes.
///
/// The formula is `num_nodes * (num_nodes + 1) / 2`.
fn num_edges(num_nodes: u32) -> u64 {
    (u64::from(num_nodes) * (u64::from(num_nodes) - 1)) / 2
}

/// Calculates the threshold for the maximum number of words that can be
/// processed given the available memory.
///
/// Note that this is just a rough estimate and the actual threshold may be
/// lower due to other factors. We only take a percentage of the total memory
/// available on the device, see `total_available_memory`.
fn num_words_in_one_run(available_memory: usize) -> u32 {
    let score_size = std::mem::size_of::<i8>() as f64;
    let max_num_edges = (available_memory as f64 / score_size) as u64;
    let max_num_nodes = 0.5 + f64::sqrt(0.25 + 2.0 * max_num_edges as f64);
    max_num_nodes as u32
}

/// Prepares the data to be used in the kernel:
/// - Flattens the words into a single vector.
/// - Creates a vector of offsets for each word.
/// - Calculates the maximum word length.
fn prepare_data(words: &Vec<Vec<u8>>) -> (Vec<u8>, Vec<u64>, u32) {
    let words_flat: Vec<u8> = words.iter().flat_map(|w| w.iter()).copied().collect();

    let mut words_offsets: Vec<u64> = words
        .iter()
        .scan(0, |acc, w| {
            let start = *acc;
            *acc += w.len();
            Some(start.try_into().unwrap())
        })
        .collect();
    words_offsets.push(words_flat.len() as u64); // last offset
    assert!(words_offsets.len() == words.len() + 1);

    let max_word_length = words.iter().map(|w| w.len()).max().unwrap();
    println!("Max word length: {max_word_length}");

    (words_flat, words_offsets, max_word_length as u32)
}

/// Configures the kernel launch parameters:
/// - Block size
/// - Grid size
/// - Shared memory size
fn configure_kernel(device: &CudaDeviceArc, num_edges: u64, max_word_length: u32) -> LaunchConfig {
    let max_shared_mem_bytes: u32 = device
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
        )
        .expect("Failed to retrieve attribute 'shared memory per block' from device")
        .try_into()
        .unwrap();

    // block size (best to maximize shared memory usage)
    let mut block_size: u32 = 1;
    let mut shared_mem_size: u32 = 0;

    for bs in (1..=1024).rev() {
        let sm_size =
            bs * (max_word_length + 1) * (max_word_length + 1) * (std::mem::size_of::<i8>() as u32);
        if sm_size <= max_shared_mem_bytes {
            block_size = bs;
            shared_mem_size = sm_size;
            break;
        }
    }

    // grid size
    let grid_size: u32 = ((num_edges + block_size as u64 - 1) / block_size as u64)
        .try_into()
        .expect("Failed to convert grid size to u32");
    let max_grid_size = device
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
        .expect("Failed to retrieve attribute 'max grid dim x' from device");
    if grid_size > max_grid_size.try_into().unwrap() {
        panic!("Grid size exceeds maximum grid size");
    }

    println!("🦷 Launch configuration");
    println!("- Block size: {}", block_size);
    println!("- Shared memory size: {} kB", shared_mem_size / 1024);
    println!("- Grid size: {:}", prettified_int(grid_size as u64));

    LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem_size,
    }
}

/// Compiles the kernel code to PTX and launches the kernel with the given
/// launch configuration.
fn compile_and_launch_kernel(
    device: &CudaDeviceArc,
    launch_config: LaunchConfig,
    words_flat_device: CudaSlice<u8>,
    words_offsets_device: CudaSlice<u64>,
    num_nodes: u32,
    num_edges: u64,
    max_word_length: u32,
    out: &mut CudaSlice<i8>,
) -> Duration {
    let kernel_code = fs::read_to_string(KERNEL_FILE).unwrap();

    println!("💫 Compile PTX");
    let ptx = cudarc::nvrtc::compile_ptx(kernel_code).expect("Failed to compile PTX");
    device
        .load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME])
        .expect("Failed to load PTX");
    let kernel = device.get_func(MODULE_NAME, KERNEL_NAME).unwrap();

    println!("🎈 Launching kernel");
    let start = std::time::Instant::now();
    unsafe {
        kernel.launch(
            launch_config,
            (
                out,
                &words_flat_device,
                &words_offsets_device,
                num_nodes,
                num_edges,
                max_word_length,
            ),
        )
    }
    .expect("Failed to launch kernel");
    device.synchronize().expect("Failed to synchronize device");
    start.elapsed()
}

fn compute(device: &CudaDeviceArc, words: Vec<Vec<u8>>) -> (Vec<i8>, Duration) {
    let num_nodes: u32 = words.len().try_into().unwrap();
    let num_edges = num_edges(num_nodes);
    println!("Num nodes: {num_nodes}");
    println!(
        "Num edges: {:} (including self-loops)",
        prettified_int(num_edges)
    );

    let (words_flat, words_offsets, max_word_length) = prepare_data(&words);
    let words_flat_device = device
        .htod_copy(words_flat)
        .expect("Failed to copy flattened words to device");
    let words_offsets_device = device
        .htod_copy(words_offsets)
        .expect("Failed to copy word offsets to device");

    println!("📭 Allocating buffers (for output writing)");
    let out_size_in_mb = (num_edges * std::mem::size_of::<i8>() as u64) / (1024 * 1024);
    println!("Buffer size (in MB): {out_size_in_mb}");
    let mut out = device
        .alloc_zeros::<i8>(num_edges.try_into().unwrap())
        .expect("Failed to allocate buffer");

    let launch_config = configure_kernel(&device, num_edges, max_word_length);
    let duration = compile_and_launch_kernel(
        &device,
        launch_config,
        words_flat_device,
        words_offsets_device,
        num_nodes,
        num_edges,
        max_word_length,
        &mut out,
    );
    println!("Kernel execution time: {:?}", duration);

    let out_host: Vec<i8> = device
        .dtoh_sync_copy(&out)
        .expect("Failed to copy buffer back from device to host");

    (out_host, duration)
}

fn analyze(results: &Vec<i8>) {
    println!("{:?}", &results[..20]);
    println!("{:?}", &results[results.len() - 20..]);

    let highest_score = results.iter().max().unwrap();
    let lowest_score = results.iter().min().unwrap();
    println!("💠 Highest score: {highest_score:?}");
    println!("💠 Lowest score: {lowest_score:?}");
}

fn save(results: &Vec<i8>) {
    assert!(results[..20].iter().any(|&x| x != 0));
    assert!(results[results.len() - 20..].iter().any(|&x| x != 0));

    let mut file = File::create(OUTPUT_FILE).expect("Failed to create {OUTPUT_FILE}");
    // weights are actually i8, but we convert them to u8 for writing
    let weights: Vec<u8> = results.iter().map(|&x| x as u8).collect();
    file.write_all(&weights)
        .expect("Failed to write to {OUTPUT_FILE}");
    println!("✅ Done! Results written to {OUTPUT_FILE}");
}

fn main() {
    // return main_time();

    let mut words = read_words_from_csv(INPUT_FILE).expect("Failed to read words from CSV");
    words.truncate(100_000);
    println!("Num total available words: {}", words.len());
    let num_edges = num_edges(words.len().try_into().unwrap());
    println!(
        "Storage needed for edges: {} MB",
        num_edges * std::mem::size_of::<i8>() as u64 / (1024 * 1024)
    );
    println!("Num edges: {:}", prettified_int(num_edges));

    let device = initialize_device();
    let available_memory = total_available_memory();
    let num_words_in_one_run = num_words_in_one_run(available_memory);
    println!("Num of words that could fit in one run: {num_words_in_one_run}");

    let (results, _duration) = compute(&device, words);
    drop(device);

    println!("\n🌟 Analyzing");
    analyze(&results);
    println!("\n📝 Saving");
    save(&results);
}

////////////////////////////////////////////////////////////////////////////////
/// 🕑 Timing
////////////////////////////////////////////////////////////////////////////////

fn main_time() {
    let num_words_list = vec![10_000, 50_000, 100_000];

    let mut csv_file = File::create(TIMING_FILE).expect("Failed to create CSV file");
    writeln!(csv_file, "num_words,duration_mean,duration_variance")
        .expect("Failed to write header");

    for &num_words in &num_words_list {
        let mut words = read_words_from_csv(INPUT_FILE).expect("Failed to read words from CSV");
        words.truncate(num_words);

        println!("Num total available words: {}", words.len());
        let num_edges = num_edges(words.len().try_into().unwrap());
        println!(
            "Storage needed for edges: {} MB",
            num_edges * std::mem::size_of::<i8>() as u64 / (1024 * 1024)
        );
        println!("Num edges: {:}", prettified_int(num_edges));

        let num_runs = 12;
        let durations = (0..num_runs)
            .map(|i| {
                println!();
                println!("---------------------------------");
                println!("Run: {:}/{num_runs} (num_words: {num_words})", i + 1);
                println!("---------------------------------");
                let device = initialize_device();
                let (_results, duration) = compute(&device, words.clone());
                drop(device);
                std::thread::sleep(std::time::Duration::from_secs(2));
                duration
            })
            .collect::<Vec<_>>();

        let mean_duration: Duration = durations.iter().sum::<Duration>() / num_runs as u32;
        let variance: f64 = durations
            .iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - mean_duration.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / (num_runs as f64 - 1.0);

        println!("🕑 Durations for {} words: {:?}", num_words, durations);
        println!("📊 Mean duration: {:?}", mean_duration);
        println!("📊 Variance: {:?}", variance);

        writeln!(
            csv_file,
            "{},{:.6},{:.6}",
            num_words,
            mean_duration.as_secs_f64(),
            variance
        )
        .expect("Failed to write to CSV file");
    }
}
