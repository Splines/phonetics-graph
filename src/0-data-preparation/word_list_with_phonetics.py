"""
Enriches a list of French words with their phonetic transcription.

french-words.txt is a high-quality list of French words from
https://github.com/DanielSWolf/wiki-pronunciation-dict
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

with open("./data/lists/french-words.txt", "r", encoding="utf-8") as f:
    word_lines = f.readlines()
    word_lines = [line.strip().split("\t") for line in word_lines]

with open("./data/lists/french-phonetics.json", "r", encoding="utf-8") as f:
    phonetics = json.load(f)

output = []

with open("./data/lists/missing-phonetics.txt", "w", encoding="utf-8") as log_file:
    for line in word_lines:
        word = line[0]
        if word in phonetics:
            phonetic = phonetics[word][0]
            line.append(phonetic)
            output.append(line)
        else:
            log_file.write(f"{word}\n")

with open("./data/lists/french.txt", "w", encoding="utf-8") as out_file:
    for line in output:
        out_file.write("\t".join(line) + "\n")

percentage = len(output) / len(word_lines) * 100
print(
    f"Phonetic transcriptions found for {len(output)}/{len(word_lines)} words ({percentage:.2f}%)"
)
