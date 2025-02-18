import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
import matplotlib

# fig_size = (2.756, 1.703)
fig_size = (2.756, 1.85)
plt.style.use(["science", "ieee"])
# plt.rcParams.update({"text.usetex": True, "font.family": "lmodern"})


def calc_num_edges(num_nodes):
    return 0.5 * num_nodes * (num_nodes - 1)


def format_nodes(value, _):
    return f"{int(value // 1_000)}"  # Convert to thousands


def main():
    data = np.genfromtxt("data/graph/final/timing-1-46.csv", delimiter=",", skip_header=1)
    num_nodes = data[:, 0]
    num_edges = calc_num_edges(num_nodes)

    fig, ax1 = plt.subplots(figsize=fig_size)
    ax2 = ax1.twiny()

    ax1.plot(num_edges, data[:, 1], "o-", ms=1.5)
    ax1.set_xlabel("Number of edges")
    ax1.set_ylabel("Time (s)")

    ax2.set_xlabel("Number of nodes")
    tick_nodes = [10_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
    tick_edge_indices = [calc_num_edges(n) for n in tick_nodes]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tick_edge_indices)
    ax2.set_xticks([], minor=True)  # turn off minor ticks

    ax2.set_xticklabels([format_nodes(n, None) for n in tick_nodes])
    # ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_nodes))

    # Add "×10³" annotation at the end of the axis
    ax2.annotate(
        r"$\times 10^3$",
        xy=(0.95, 1.02),
        xycoords="axes fraction",
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    plt.tight_layout()
    plt.savefig("eval/timing.pdf")


if __name__ == "__main__":
    main()
