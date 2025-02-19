import pickle
from word_list_model import WordData
from tqdm import tqdm

NUM_WORDS = 30000


def used_phonetic_symbols() -> set[str]:
    """
    Returns the set of phonetic symbols used in the French phonetics list.

    This set is read from an already existing file which lists the symbols.
    """
    with open("./data/graph/french-phonetics-list.txt", "r", encoding="utf-8") as f:
        return set(line.rstrip() for line in f)


def possible_combinations(symbols: set[str]) -> set[tuple[str, str]]:
    """
    Returns all possible combinations of phonetic symbols (order does not matter).
    """
    combinations = set()
    for symbol1 in symbols:
        for symbol2 in symbols:
            if symbol1 != symbol2:
                if (symbol2, symbol1) not in combinations:
                    combinations.add((symbol1, symbol2))
    return combinations


def read_words() -> list[WordData]:
    with open("./data/lists/french.pkl", "rb") as f:
        return pickle.load(f)


def main():
    """
    Reads in the pickled french.pkl as a list of WordData objects.
    Randomly goes through all words and saves the words in categories
    """
    symbols = used_phonetic_symbols()
    print(symbols)
    num_possible_combinations = len(possible_combinations(symbols))

    # Read words
    words = read_words()
    print(f"Number of words: {len(words)}")
    words = words[:NUM_WORDS]
    print(f"Thereof, we consider {len(words)} words.")

    # Randomly iterate through the words. Inside this random iteration
    # do another random iteration to consider pairs. Then find the words
    # where only one phonetic symbol was changed to arrive at the other word.
    # Save these pairs by their pair of phonetic symbols, i.e. the phonetic
    # symbols that were changed.
    pairs_by_phonetic_change = {}

    for word1 in tqdm(words):
        for word2 in words:
            if word1 == word2:
                continue

            # Make sure length of phonetics differs by at most 1
            if abs(len(word1.phonetics) - len(word2.phonetics)) > 1:
                continue

            # Calculate differences
            phonetic_diff = [
                (p1, p2) for p1, p2 in zip(word1.phonetics, word2.phonetics) if p1 != p2
            ]

            if len(phonetic_diff) == 1:
                key = phonetic_diff[0]  # first pair of phonetic symbols that differ
                if key not in pairs_by_phonetic_change:
                    pairs_by_phonetic_change[key] = []
                pairs_by_phonetic_change[key].append((word1, word2))

    # Calculate how many phonetic symbols pairs were found out of num_possible_combinations
    num_found_combinations = len(pairs_by_phonetic_change)
    percentage_found = num_found_combinations / num_possible_combinations
    print(
        f"Found combinations: {num_found_combinations} / {num_possible_combinations}"
        + f" ({(percentage_found*100):.2f}%)"
    )

    # Save the pairs as CSV file
    with open(
        f"./data/graph/phonetic-change-pairs-sorted-{NUM_WORDS}.csv",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("phonetic1,phonetic2,word1,word2\n")
        for phonetic_change, pairs in pairs_by_phonetic_change.items():
            for word1, word2 in pairs:
                word1_repr = f"{word1.word} ({''.join(word1.phonetics)})"
                word2_repr = f"{word2.word} ({''.join(word2.phonetics)})"
                f.write(f"{phonetic_change[0]},{phonetic_change[1]},{word1_repr},{word2_repr}\n")


if __name__ == "__main__":
    main()
