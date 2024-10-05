import pickle
import time
from needleman_wunsch import NeedlemanWunsch, SimilarityMatrix
from data_model import WordIpaChars
import csv

WORDS_PICKLE_PATH = "./data/ipa/fr_FR.pkl"
SIMILARITY_COSTS_PICKLE_PATH = "./data/ipa/fr_FR_similarity_costs.pkl"
NODES_CSV_PATH = "./data/ipa/graph/nodes.csv"
EDGES_CSV_PATH = "./data/ipa/graph/edges.csv"


def generate_nodes(words):
    # Go once through all nodes to initialize the set of nodes
    # and the lookup table for node ids. For words that have the same word.word,
    # treat them as different nodes (they can have different ipa values).
    # The key in the node_to_id dictionary is the word.word, followed by its ipa value.
    nodes = []
    node_to_id = {}
    node_id = 0
    for word in words:
        # node_to_id[f"{word.word}_{word.ipa}"] = node_id
        node_to_id[word.word] = node_id
        nodes.append((word.word, word.ipa))
        node_id += 1
    with open(NODES_CSV_PATH, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for node_id, (word, ipa) in enumerate(nodes):
            writer.writerow([node_id, f"{word} ({ipa})"])

    return node_to_id


def calculate_graph():
    # Words as nodes
    with open(WORDS_PICKLE_PATH, "rb") as f:
        words = pickle.load(f)
        node_to_id = generate_nodes(words)
        print(f"🌟 Nodes generated in {NODES_CSV_PATH} ({len(words)} nodes)")

    # Init algorithm with similarity matrix
    with open(SIMILARITY_COSTS_PICKLE_PATH, "rb") as f:
        french_similarity_costs = pickle.load(f)
    similarity_matrix = SimilarityMatrix(french_similarity_costs)
    algorithm = NeedlemanWunsch(similarity_matrix, -1)

    # Calculate the score for every possible combination of two words.
    # Take into account the symmetry, i.e. don't calculate the score for the same pair twice.
    edges = []
    for i, word in enumerate(words[3000:3010]):
        print(f"▶ Word: {word}")
        start_time = time.time()

        for j, word2 in enumerate(words):
            if j < i:  # Edges are undirected
                continue
            score = algorithm.calculate_score(word.ipa_chars, word2.ipa_chars)

            node_source, node_target = node_to_id[word.word], node_to_id[word2.word]
            edges.append((node_source, node_target, score))

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Time taken for word {i}: {elapsed_time_ms:.2f} ms")

    # Save edges
    # Normalize edges such that the most negative value is at 0
    print(f"🌟 Edges calculated ({len(edges)} edges)")
    min_score = min(edge[2] for edge in edges)
    normalized_edges = [
        (source, target, score + min_score) for source, target, score in edges
    ]
    print("🌟 Edges normalized")

    with open(EDGES_CSV_PATH, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for edge in normalized_edges:
            writer.writerow(edge)
    print(f"🌟 Edges saved to {EDGES_CSV_PATH}")


if __name__ == "__main__":
    calculate_graph()
