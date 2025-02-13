use std::fs;

use cudarc::driver::{LaunchAsync, LaunchConfig};

static KERNEL_FILE: &str = "./src/kernels/needleman_wunsch.cpp";
static MODULE_NAME: &str = "phonetics_module";
static KERNEL_NAME: &str = "needleman_wunsch";

fn compute() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("Device name: {}", dev.name()?);

    let num_nodes: usize = 5;
    let num_adjacency_matrix_elements: usize = (num_nodes * (num_nodes + 1)) / 2;
    println!(
        "num_adjacency_matrix_elements: {}",
        num_adjacency_matrix_elements
    );

    println!("Allocating buffers");
    let inp = dev.htod_copy(vec![1.0f32; num_adjacency_matrix_elements])?;
    let mut out = dev.alloc_zeros::<f32>(num_adjacency_matrix_elements)?;

    let mut kernel_code = fs::read_to_string(KERNEL_FILE).unwrap();
    kernel_code = format!("static const float z = {num_nodes}.5;\n") + &kernel_code;
    let ptx = cudarc::nvrtc::compile_ptx(kernel_code)?;

    println!("Loading PTX");
    dev.load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME])?;
    let kernel = dev.get_func(MODULE_NAME, KERNEL_NAME).unwrap();
    let cfg = LaunchConfig::for_num_elems(num_adjacency_matrix_elements as u32);

    println!("Launching kernel");
    unsafe { kernel.launch(cfg, (&mut out, &inp, 42usize)) }?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    assert_eq!(out_host.len(), num_adjacency_matrix_elements);
    println!("{:?}", out_host);

    println!("Done");
    Ok(())
}

fn main() {
    if let Err(e) = compute() {
        eprintln!("Error: {}", e);
    }
}
