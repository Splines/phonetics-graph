use std::fs;

use cudarc::driver::{LaunchAsync, LaunchConfig};

static KERNEL_FILE: &str = "./src/kernels/sin_kernel.cpp";

fn getting_started() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("Device name: {}", dev.name()?);

    println!("Allocating buffers");
    let inp = dev.htod_copy(vec![1.0f32; 100])?;
    let mut out = dev.alloc_zeros::<f32>(100)?;

    let kernel_code = fs::read_to_string(KERNEL_FILE).unwrap();
    assert!(kernel_code.contains("sin_kernel"), "Kernel code does not contain 'sin_kernel'");
    let ptx = cudarc::nvrtc::compile_ptx(kernel_code)?;

    println!("Loading PTX");
    dev.load_ptx(ptx, "my_module", &["sin_kernel"])?;

    println!("Retrieving kernel");

    let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(100);

    println!("Launching kernel");
    unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100usize)) }?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    assert_eq!(out_host.len(), 100);

    println!("Done");
    Ok(())
}

fn main() {
    if let Err(e) = getting_started() {
        eprintln!("Error: {}", e);
    }
}
