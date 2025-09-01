> [!warning]
> This includes preliminary results and various notes for myself. Not necesarilly well structured or explained.


Fully-connected graphs with undirected edges.

$$\text{num edges} = \binom{n}{2} = \frac{n(n-1)}{2}$$


## Rust Cuda libraries
- [Rust Cuda Reboot](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot) (Blog Post) & [Issue](https://github.com/Rust-GPU/Rust-CUDA/issues/130)
- [Rust-GPU Getting Started](https://rust-gpu.github.io/Rust-CUDA/guide/getting_started.html)
- [**cudarc**](https://github.com/coreylowman/cudarc) (used here)


## Cuda in general
- [Cuda C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#)
- [Cuda in WSL (Git Gist)](https://gist.github.com/Ayke/5f37ebdb84c758f57d7a3c8b847648bb) | [Enable Cuda on WSL2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) | [Nvidia: Getting Started with CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2)


## Run

See the main Readme for better installation instructions.

```bash
$ RUST_BACKTRACE=full cargo run --bin edges --profile release
```


## Expected results

```bash
first 20: [2, 0, -2, -2, -2, -2, 0, -2, -2, -2, 2, -2, -2, 0, -2, -2, -2, 1, 0, -3]

# for 6000 elements total:
last 20: [-5, -4, -5, -5, -2, -4, -6, -4, -4, -4, 8, -8, -5, -8, -4, -4, -4, 7, -6, -5]
```


## Timing

These are preliminary results. See the paper for the official results.

| Number of nodes | Time CPU | Time GPU |
|-|-|-|
| 10,000 | 12s | 173.2ms |
| 15,000 | 24s | - |
| 20,000 | 42s | 438ms |
| 30,000 | 93s | 1.09s |
| 35,000 | 132s | 1.73s |
| 40,000 | 159s (cannot sort anymore) | - |
| 50,000 | couldn't test due to memory constraints (RAM) | - |
| 100,000 | - | 2.57s |
| 200,000 | - | 3.80s |
| 300,000 | - | 11.33s (weird alignment issue?) |
| 350,000 | - | 5.88s |
| 400,000 | - | 2.64s |
| 500,000 | - | 2.21s |
| 611,786 | - | 1.38s |

1 to 46: 30 runs each<br>
47 to 51: 12 runs each


## Other useful stuff

**Diff two binary files.** Command from [here](https://superuser.com/a/968863/):

```bash
cmp --silent ../data/graph/final/edges.cpu.bin ../data/graph/final/edges.gpu.bin # check error code (!)
diff <(xxd ../data/graph/final/edges.cpu.bin) <(xxd ../data/graph/final/edges.gpu.bin)
```

**Show head of binary file.**

```bash
head -c 100 ../data/graph/edges-new-gpu.bin | od -t d1
```


## Graphs

We consider different ranges of edge weights. See the paper for more details.

| Weight range | Details |
|-|-|
| $[60, 100]$ | _puissant_ ego |
| $[40, 49]$ | _étirer_ ego |
| $[8, 10]$ | $30,185$ nodes ($30.19\%$),  $306,473$ edges (big view) |
| only $-35$ | does not give any good results |
| $[60, 100]$ | _emporter_ ego (based on $458,529$ edges, small graph: $449$ nodes, $5921$ edges) |

Rendering via [Gephi](https://gephi.org/): node opacity 20, edge opacity 30
