## Rust Cuda libraries
- [Rust Cuda Reboot](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot) (Blog Post) & [Issue](https://github.com/Rust-GPU/Rust-CUDA/issues/130)
- [Rust-GPU Getting Started](https://rust-gpu.github.io/Rust-CUDA/guide/getting_started.html)
- [**cudarc**](https://github.com/coreylowman/cudarc?tab=readme-ov-file)

## Cuda in general
- [Cuda C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#)
- [Cuda in WSL (Git Gist)](https://gist.github.com/Ayke/5f37ebdb84c758f57d7a3c8b847648bb) | [Enable Cuda on WSL2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) | [Nvidia: Getting Started with CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2)

## Run

```bash
RUST_BACKTRACE=full cargo run --bin edges --profile release
```
