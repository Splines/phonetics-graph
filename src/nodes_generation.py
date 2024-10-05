import pickle
import csv
from data_model import WordIpaChars

WORDS_PICKLE_PATH = "./data/ipa/fr_FR.pkl"
NODES_CSV_PATH = "./data/ipa/graph-rust/nodes.csv"


def generate_and_save_nodes(words):
    # Go once through all nodes to initialize the set of nodes
    # and the lookup table for node ids. For words that have the same word.word,
    # treat them as different nodes (they can have different ipa values).
    # The key in the node_to_id dictionary is the word.word, followed by its ipa value.
    nodes = []
    node_to_id = {}
    node_id = 0
    for word in words:
        node_to_id[word.word] = node_id
        nodes.append((word.word, word.ipa))
        node_id += 1
    with open(NODES_CSV_PATH, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for node_id, (word, ipa) in enumerate(nodes):
            writer.writerow([node_id, f"{word} ({ipa})"])

    return node_to_id


def main():
    with open("./data/ipa/fr_FR.pkl", "rb") as f:
        words = pickle.load(f)

    generate_and_save_nodes(words)


if __name__ == "__main__":
    main()
