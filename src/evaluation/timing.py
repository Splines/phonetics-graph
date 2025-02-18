import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib

fig_size = (2.756, 1.703)
matplotlib.use("TkAgg")
plt.style.use(["science", "ieee"])


def main():
    data = np.genfromtxt("data/graph/final/timing-1-46.csv", delimiter=",", skip_header=1)
    num_edges = 0.5 * data[:, 0] * (data[:, 0] - 1)

    plt.figure(figsize=fig_size)
    # plt.errorbar(num_edges, data[:, 1], yerr=data[:, 2], capsize=3, ms=2, fmt="o")
    plt.plot(num_edges, data[:, 1], "o-", ms=1.5)

    plt.xlabel("Number of edges")
    plt.ylabel("Time (s)")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval/timing.pdf")


if __name__ == "__main__":
    main()
