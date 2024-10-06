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


def main():
    with open("./data/lists/french.pkl", "rb") as f:
        words: list[WordData] = pickle.load(f)

    frequencies_average = [calculate_average_frequency(word) for word in words]
    words_sorted = sorted(
        zip(words, frequencies_average), key=lambda x: x[1], reverse=True
    )

    print(f"Number of words: {len(words)}")
    print("Top 100 words by frequency:")
    for word, freq in words_sorted[:100]:
        print(f"{word.word} ({word.lemma}) - {freq:.2f}")

    phonetic_symbols = used_phonetic_symbols(words)
    print(f"Number of phonetic symbols: {len(phonetic_symbols)}")
    print("Phonetic symbols:")
    print(phonetic_symbols)


if __name__ == "__main__":
    main()
