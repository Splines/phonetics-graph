import pickle
from word_list_model import WordData


def calculate_average_frequency(word: WordData):
    freq_1 = word.frequency_lexique
    freq_2 = word.frequency_ngrams
    if freq_1 != -1.0 and freq_2 != -1.0:
        return (freq_1 + freq_2) / 2
    if freq_1 != -1.0:
        return freq_1
    if freq_2 != -1.0:
        return freq_2
    return 0.0


def used_phonetic_symbols(words: list[WordData]) -> set[str]:
    symbols = set()
    for word in words:
        symbols.update(word.phonetics)
    return symbols


def convert_words_to_phonetic_integers(
    words: list[WordData], phonetic_symbols: list[str]
) -> list[list[int]]:
    phonetic_symbol_to_int = {symbol: i for i, symbol in enumerate(phonetic_symbols)}
    res = []
    for word in words:
        representation = [phonetic_symbol_to_int[symbol] for symbol in word.phonetics]
        res.append(representation)
    return res


def main():
    with open("./data/lists/french.pkl", "rb") as f:
        words: list[WordData] = pickle.load(f)

    frequencies_average = [calculate_average_frequency(word) for word in words]
    words_sorted = sorted(zip(words, frequencies_average), key=lambda x: x[1], reverse=True)
    words_sorted_ = [word for word, _ in words_sorted]

    print(f"Number of words: {len(words)}")
    print()

    print("Top words by frequency:")
    for word, freq in words_sorted[:10]:
        print(f"{word.word} ({word.lemma}) - {freq:.2f}")

    phonetic_symbols = used_phonetic_symbols(words)
    print()
    print(f"Number of phonetic symbols: {len(phonetic_symbols)}")
    print("Phonetic symbols:")
    print(phonetic_symbols)

    # Construct node csv file with id,label
    with open("./data/graph/nodes.csv", "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for i, word in enumerate(words_sorted_):
            phonetic_representation = "".join(word.phonetics)
            f.write(f"{i},{word.word} ({phonetic_representation})\n")

    # Convert phonetic symbols to integers and store them
    words_phonetic_integers = convert_words_to_phonetic_integers(
        words_sorted_, list(phonetic_symbols)
    )
    with open("./data/graph/french-phonetics-integers.txt", "w", encoding="utf-8") as f:
        for representation in words_phonetic_integers:
            str_list = [str(i) for i in representation]
            f.write(",".join(str_list) + "\n")


if __name__ == "__main__":
    main()
