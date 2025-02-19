import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots

fig_size = (2.756, 1.703)
HIGHLIGHT_COLOR = "#FF4A6D"
plt.style.use(["science", "ieee"])

INPUT_FILE = "data/graph/final/edge_weights.csv"
OUTPUT_FILE = "eval/edge_weights.pdf"


def main():
    """
    Reads the edge_weights.csv file (weight,count) and plots the histogram
    of the weights.
    """
    data = np.genfromtxt(INPUT_FILE, delimiter=",", skip_header=1)
    weights = data[:, 0]
    counts = data[:, 1]

    print(f"Mean: {np.mean(weights)}")
    print(f"Median: {np.median(weights)}")
    print(f"Max: {np.max(weights)}")
    print(f"Min: {np.min(weights)}")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(fig_size[0], fig_size[1] * 2),
        sharex=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    ax1.bar(weights, counts, width=0.5, color=HIGHLIGHT_COLOR)
    ax1.set_ylabel("Count")

    ax2.bar(weights, counts, width=0.5, color=HIGHLIGHT_COLOR)
    ax2.set_xlabel("Edge Weight")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
