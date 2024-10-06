"""
Enriches a list of French words with their phonetic transcription.

french-words.txt is a high-quality list of French words from
https://github.com/frodonh/french-words
from the documentation:
# Structure of files
The list of words is in a tab-separated file with the following fields:
- word
- POS-tagging, using the Lexique.org nomenclature
- associated lemma
- frequency according to Lexique.org (measured in number of occurrences per million of words)
- frequency according to Google Ngrams (measured in number of occurrences per million of words)

Furthermore in french-phonetics.json, the phonetic transcription of many
French words is provided from
https://github.com/DanielSWolf/wiki-pronunciation-dict
From the documentation:
The spelling of each word is given as a string consisting entirely of the
graphemes listed in the corresponding metadata file (see the next section).
To simplify processing, phonemes are separated by space characters.
e.g. "tondeuse": ["t ɔ̃ d ø z"]
"""

import json
import csv


def convert_frequency_str_to_float(float_str: str) -> float:
    try:
        return float(float_str)
    except ValueError:
        raise ValueError(f"Invalid frequency '{float_str}'.")


with open("./data/lists/french-words.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]

with open("./data/lists/french-phonetics.json", "r", encoding="utf-8") as f:
    phonetics_json = json.load(f)

output = []
phonetics = []
words_with_alternative_phonetics = []

# Merge word list with phonetics
with open("./data/lists/missing-phonetics.txt", "w", encoding="utf-8") as log_file:
    for line in lines:
        word = line[0]
        if word in phonetics_json:
            phonetic_options = phonetics_json[word]
            if len(phonetic_options) > 1:
                words_with_alternative_phonetics.append((word, phonetic_options))
            # Just take the first phonetic transcription available
            phonetics.append(phonetic_options[0])
            output.append(line)
        else:
            log_file.write(f"{word}\n")

assert len(output) == len(phonetics)

# Small parser to construct a CSV file with phonetic transcriptions.
# Note that some fields may not be present, e.g. associated lemma
# or frequencies. In this case, the fields are filled with empty strings.
# Word, POS and phonetic transcription are always present.
# Note that one word can have multiple POS-tags. In this case, the word
# occurs multiple times in the CSV file.
with open("./data/lists/french-phonetics.csv", "w", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "word",
            "POS",
            "lemma",
            "frequency_1",
            "frequency_2",
            "phonetic_transcription",
        ]
    )
    for line in output:
        word = line[0]
        pos = line[1]

        # if the next word is a string, it is the lemma
        # if not, it could be a frequency (if is a number)
        # but there might be many combinations since lemma, frequency_1 and frequency_2 are optional
        # therefore, write a small parser that checks the type of the current word
        # to determine if it is a lemma or a frequency
        lemma = ""
        frequency_1 = ""
        frequency_2 = ""

        # The conversion in the following steps is only done to make sure
        # the string representation is really a number.
        # After that it is saved as a string in the CSV anyways.
        if len(line) > 2:
            print(line)
            if line[2].isalpha():
                lemma = line[2]
                if len(line) > 3:
                    frequency_1 = convert_frequency_str_to_float(line[3])
                if len(line) > 4:
                    frequency_2 = convert_frequency_str_to_float(line[4])
            elif line[2].isdigit():
                frequency_1 = convert_frequency_str_to_float(line[2])
                if len(line) > 3:
                    frequency_2 = convert_frequency_str_to_float(line[3])

        phonetic_transcription = phonetics.pop(0)
        csv_writer.writerow(
            [
                word,
                pos,
                lemma,
                frequency_1,
                frequency_2,
                phonetic_transcription,
            ]
        )


with open(
    "./data/lists/french-alternative-phonetics.txt", "w", encoding="utf-8"
) as out_file:
    for word, phonetic_options in words_with_alternative_phonetics:
        out_file.write(f"{word}\t{phonetic_options}\n")

percentage = len(output) / len(lines) * 100
print(
    f"Phonetic transcriptions found for {len(output)}/{len(lines)} words ({percentage:.2f}%)"
)
print(
    f"Words with alternative phonetic transcriptions: {len(words_with_alternative_phonetics)}"
)
