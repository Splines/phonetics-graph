import csv
import pickle
from data_model import WordIpaChars

french_allowed_symbols = [
    "b",
    "d",
    "f",
    "g",
    "k",
    "l",
    "m",
    "n",
    "p",
    "s",
    "t",
    "v",
    "z",
    "ɲ",
    "ʁ",
    "ʃ",
    "ʒ",
    "dʒ",
    "tʃ",
    "ŋ",
    "j",
    "w",
    "ɥ",
    "a",
    "ɑ",
    "e",
    "i",
    "o",
    "u",
    "y",
    "ø",
    "œ",
    "ɔ",
    "ə",
    "ɛ",
    "ɑ̃",
    "ɔ̃",
    "ɛ̃",
    "œ̃",
]

# Open the file with the words and their phonetic transcription
# then convert these transcriptions to a list of integers.
# The integers are the ids of the phonetic symbols, determined by the position
# in the french_allowed_symbols array. Store the resulting file as csv
# just with the arrays of integers (no id column).


class Phonetics:

    @classmethod
    def split(cls, string, allowed_symbols) -> list[str]:
        """
        Extracts the phonetic symbols from a word by means of a small 1-look-ahead parser.
        """
        symbols = []

        i = 0
        while i < len(string):
            letter = string[i]

            # Ignore these symbols
            if letter in ["ː", ".", "ʼ", " "]:
                i += 1
                continue

            if letter not in allowed_symbols:
                raise ValueError(f"Invalid symbol '{letter}' in word '{string}'.")

            if letter == "\u0303":
                raise ValueError(
                    "Unicode 'Combining tilde' character should have been subsumed into the previous character."
                )

            if i + 1 < len(string):
                next_letter = string[i + 1]
                if letter in ["ɑ", "ɛ", "ɔ", "œ"] and next_letter == "\u0303":
                    symbols.append(letter + "\u0303")
                    i += 2
                    continue

                if letter == "d" and next_letter == "ʒ":
                    symbols.append("dʒ")
                    i += 2
                    continue

                if letter == "t" and next_letter == "ʃ":
                    symbols.append("tʃ")
                    i += 2
                    continue

            symbols.append(string[i])
            i += 1

        return symbols


with open("./data/ipa/fr_FR.pkl", "rb") as f:
    words = pickle.load(f)

res = []

for word in words:
    symbols_list = Phonetics.split(word.ipa, french_allowed_symbols)
    representation = [french_allowed_symbols.index(symbol) for symbol in symbols_list]
    res.append(representation)

with open("./data/ipa/fr_FR_words_symbols.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in res:
        writer.writerow(row)
