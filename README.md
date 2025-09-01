# Parallel Needleman–Wunsch on CUDA to measure word similarity based on phonetic transcriptions

> [!warning]
> This is a repository accompanying a research paper on arxiv, which was published on September 1, 2025. I don't plan to maintain this repository in the long-term. Use at your own risk.

| [Read Paper](./paper/main.pdf) (soon on arXiv) | [Watch video on YouTube](https://youtu.be/xbcpnItE3_4) |
:---------------------|------------------------:
| <img src="https://github.com/user-attachments/assets/2263d1ad-4aec-42f6-bcf3-af48674b2ccc" height="300" /> | <img src="https://github.com/user-attachments/assets/a3e7d647-480f-44cd-b6b8-2191a31a3feb" height="300" />  |


## 📜 Abstract

We present a method to calculate the **similarity between words based on their phonetic transcription (their pronunciation) using the Needleman–Wunsch algorithm**. We implement this algorithm in Rust and parallelize it on both CPU and GPU to handle large datasets efficiently. The GPU implementation leverages CUDA and the cudarc Rust library to achieve significant performance improvements.

We validate our approach by constructing a fully-connected graph where nodes represent words and edges have weights according to the similarity between the words. This graph is then analyzed using clustering algorithms to identify groups of phonetically similar words. Our results demonstrate the feasibility and effectiveness of the proposed method in analyzing the phonetic structure of languages. It might be easily expanded to other languages.


## 🎈 Run

**Preprocess data**. See the data section in this Readme for more details. Also see the Python virual env section below.

```bash
$ python3 python/0-data-preparation/<script-up-to-number-2>.py
```

Run the parallelized **Rust GPU** implementation. The code was tested on a consumer NVIDIA GeForce GTX 1060 6GB. It might not work as intended on other GPUs, although the algorithms were designed to be general enough.

```bash
$ cargo run --bin gpu --release
$ cargo run --bin make_graph --release
```

Run the parallelized, but slow Rust CPU implementation.

```bash
$ cargo run --release
```

Plot some outputs. Create a top-level `eval/` folder first for the scripts to work.

```bash
$ python3 python/evaluation/<script>.py
```


## Python virtual env

It is advised to run the Python scripts in a virtual environment.

```bash
python -m venv projectname
source projectname/bin/activate
(venv) $ pip install -r requirements.txt
```


## 💾 Data

This repository does _not_ contain the data used in the paper. Instead, you can download it from the following sources (also see the paper for more details). For the Python scripts to run properly, create the folders `data/lists/` and `data/graph/`. Then fill them with the following files:

- `data/lists/french-words.txt`: [french-words](https://github.com/frodonh/french-words). French words with partial POS-tagging and relative frequencies. By [frodonh](https://github.com/frodonh). Note that this uses many different sources to construct the word list, see the repository for more details.
- `data/lists/french-phonetics.json`: [WikiPronunciationDict](https://github.com/DanielSWolf/wiki-pronunciation-dict/tree/main/dictionaries). Pronunciation dictionaries for several languages, based on Wiktionary data. By [Daniel Wolf](https://github.com/DanielSWolf).
