import pickle
from word_list_model import WordData


def convert_to_float_with_default(value: str, default: float) -> float:
    """
    Converts a string to a float, returning a default value if the conversion fails.
    """
    try:
        return float(value)
    except ValueError:
        return default


def process_phonetics(phonetics: str) -> list[str]:
    """
    Turns the whitespace-separated phonetic string into a list of phonetic symbols.

    Note that d followed by ʒ will be aggregated into dʒ.
    Likewise for t followed by ʃ, which will be aggregated into tʃ.

    Symbols like ɔ̃ that consist of multiple Unicode chars are not a problem here
    since we split at whitespaces.
    """
    symbols = phonetics.strip().split(" ")
    res = []
    for i, symbol in enumerate(symbols):
        if symbol == "d" and i + 1 < len(symbols) and symbols[i + 1] == "ʒ":
            res.append("dʒ")
        elif symbol == "t" and i + 1 < len(symbols) and symbols[i + 1] == "ʃ":
            res.append("tʃ")
        else:
            res.append(symbol)
    return symbols


def convert_line_to_word_data(line: list[str], word_pos_tags_dict) -> WordData:
    """
    Converts a line from the word list raw data to a WordData object.
    """
    word = line[0]
    lemma = line[2]
    frequency_lexique = convert_to_float_with_default(line[3], -1.0)
    frequency_google = convert_to_float_with_default(line[4], -1.0)
    phonetics = process_phonetics(line[5])

    if word in word_pos_tags_dict:
        pos_tags = word_pos_tags_dict[word]
    else:
        raise RuntimeError(f"Word {word} not found in word_dict. This should not happen.")

    return WordData(
        word=word,
        pos=pos_tags,
        lemma=lemma,
        frequency_lexique=frequency_lexique,
        frequency_ngrams=frequency_google,
        phonetics=phonetics,
    )


def main():
    """
    Converts the French word list from a text file to a Python pickle file
    containing a list of WordData objects.
    """
    with open("./data/lists/french.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.rstrip("\n").split("\t") for line in lines]

    output: list[WordData] = []

    # Some words may have multiple lines due to multiple associated POS-tags.
    # Collect all POS-tags for a word.
    word_pos_tags_dict = {}
    for line in lines:
        word, pos_tag = line[0], line[1]
        if word in word_pos_tags_dict:
            word_pos_tags_dict[word].append(pos_tag)
        else:
            word_pos_tags_dict[word] = [pos_tag]

    for line in lines:
        word_data = convert_line_to_word_data(line, word_pos_tags_dict)
        output.append(word_data)

    assert len(output) == len(lines)
    print(f"Number of words: {len(output)}")

    with open("./data/lists/french.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()
