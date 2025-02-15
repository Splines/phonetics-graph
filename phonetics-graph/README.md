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

## Expected results

```bash
first 20: [2, 0, -2, -2, -2, -2, 0, -2, -2, -2, 2, -2, -2, 0, -2, -2, -2, 1, 0, -3]

# for 6000 elements total:
last 20: [-5, -4, -5, -5, -2, -4, -6, -4, -4, -4, 8, -8, -5, -8, -4, -4, -4, 7, -6, -5]
```


## Timing (Preliminary results)

Fully-connected graphs with undirected edges.

$$\text{num edges} = \binom{n}{2} = \frac{n(n-1)}{2}$$

**CPU**:

- 10,000 nodes: 12s
- 15,000 nodes: 24s
- 20,000 nodes: 42s
- 30,000 nodes: 93s
- 35,000 nodes: 132s
- 50,000 nodes: couldn't test due to memory constraints (RAM)

**GPU (CUDA)**:

- 10,000 nodes: 173.2ms
- 20,000 nodes: 438ms
- 30,000 nodes 1.09s
- 35,000 nodes: 1.73s
- 100,000 nodes: 2.57s
- 200,000 nodes: 3.80s
- 300,000 nodes: 11.33s
- 350,000 nodes: 5.88s
- 400,000 nodes: 2.64s
- 500,000 nodes: 2.21s
- 611,786 nodes: 1.38s


## Diff

Command from [here](https://superuser.com/a/968863/):

```bash
cmp --silent ../data/graph/edges-new.bin ../data/graph/edges-new-gpu.bin # check error code (!)
diff <(xxd ../data/graph/edges-new.bin) <(xxd ../data/graph/edges-new-gpu.bin)
```
