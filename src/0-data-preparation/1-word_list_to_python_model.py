import pickle
from word_list_model import WordData

with open("./data/lists/french.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [line.rstrip("\n").split("\t") for line in lines]

output: list[WordData] = []


def convert_to_float_with_default(value: str, default: float) -> float:
    try:
        return float(value)
    except ValueError:
        return default


# Some words may have multiple lines due to multiple associated POS-tags.
# Collect all POS-tags for a word.
word_dict = {}
for line in lines:
    word, pos_tag = line[0], line[1]
    if word in word_dict:
        word_dict[word].append(pos_tag)
    else:
        word_dict[word] = [pos_tag]

for line in lines:
    word = line[0]
    lemma = line[2]
    frequency_lexique = convert_to_float_with_default(line[3], -1.0)
    frequency_google = convert_to_float_with_default(line[4], -1.0)
    phonetics = line[5].strip().split(" ")

    if word in word_dict:
        pos_tags = word_dict[word]
    else:
        raise RuntimeError(
            f"Word {word} not found in word_dict. This should not happen."
        )

    output.append(
        WordData(
            word=word,
            pos=pos_tags,
            lemma=lemma,
            frequency_lexique=frequency_lexique,
            frequency_ngrams=frequency_google,
            phonetics=phonetics,
        )
    )

assert len(output) == len(lines)
print(f"Number of words: {len(output)}")

with open("./data/lists/french.pkl", "wb") as f:
    pickle.dump(output, f)
