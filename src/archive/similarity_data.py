# We use a customized phonetic similarity matrix to punish or reward
# certain phonetic substitutions.

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

# french_similarity_data = {
#     # "b"
#     ("b", "b"): 2,
#     ("b", "d"): -1,
#     ("b", "f"): -3,
#     ("b", "g"): -3,
#     ("b", "k"): -3,
#     ("b", "l"): -3,
#     ("b", "m"): -3,
#     ("b", "n"): -3,
#     ("b", "p"): -3,
#     ("b", "s"): -3,
#     ("b", "t"): -3,
#     ("b", "v"): -3,
#     ("b", "z"): -3,
#     ("b", "ɲ"): -3,
#     ("b", "ʁ"): -3,
#     ("b", "ʃ"): -3,
#     ("b", "ʒ"): -3,
#     ("b", "dʒ"): -3,
#     ("b", "tʃ"): -3,
#     ("b", "ŋ"): -3,
#     ("b", "j"): -3,
#     ("b", "w"): -3,
#     ("b", "ɥ"): -3,
#     ("b", "a"): -3,
#     ("b", "ɑ"): -3,
#     ("b", "e"): -3,
#     ("b", "i"): -3,
#     ("b", "o"): -3,
#     ("b", "u"): -3,
#     ("b", "y"): -3,
#     ("b", "ø"): -3,
#     ("b", "œ"): -3,
#     ("b", "ɔ"): -3,
#     ("b", "ə"): -3,
#     ("b", "ɛ"): -3,
#     ("b", "ɑ̃"): -3,
#     ("b", "ɔ̃"): -3,
#     ("b", "ɛ̃"): -3,
#     ("b", "œ̃"): -3,
# }

french_similarity_data = {}
for i, sym1 in enumerate(french_allowed_symbols):
    for sym2 in french_allowed_symbols[i:]:
        if sym1 == sym2:
            french_similarity_data[(sym1, sym2)] = 2
        else:
            french_similarity_data[(sym1, sym2)] = -3

# Pickle the data
import pickle

with open("./data/ipa/fr_FR_similarity_costs.pkl", "wb") as f:
    pickle.dump(french_similarity_data, f)

print(len(french_similarity_data))
