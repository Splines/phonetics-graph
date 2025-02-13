use cudarc::driver::{LaunchAsync, LaunchConfig};

fn getting_started() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    println!("Device name: {}", dev.name()?);

    println!("Allocating buffers");
    let inp = dev.htod_copy(vec![1.0f32; 10000000])?;
    let mut out = dev.alloc_zeros::<f32>(10000000)?;

    let ptx = cudarc::nvrtc::compile_ptx(
"extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}"
)?;

    println!("Loading PTX");
    dev.load_ptx(ptx, "my_module", &["sin_kernel"])?;

    println!("Retrieving kernel");

    let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(10000000);

    println!("Launching kernel");
    unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 10000000usize)) }?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    assert_eq!(out_host.len(), 10000000);

    println!("Done");
    Ok(())
}

fn main() {
    if let Err(e) = getting_started() {
        eprintln!("Error: {}", e);
    }
}
