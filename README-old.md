## Phonetics Graph

This projects aims at defining a metric to measure the similarity between two words based on their phonetic transcription. This metric is calculated for every pair of words in a given list of words. We then process the results to generate a graph where the nodes are the words and the edges are the similarity between the words. Graph clustering algorithms are then applied to the graph to identify groups of words that are phonetically similar.

- International Phonetic Alphabet (IPA)
- https://github.com/open-dict-data/ipa-dict
- https://github.com/xinjli/ucla-phonetic-corpus
- https://archive.phonetics.ucla.edu/
- https://archive.phonetics.ucla.edu/Language/ipa-pop-up-2.html
- [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- https://mlcommons.org/datasets/multilingual-spoken-words/

### French list
- https://github.com/hbenbel/French-Dictionary
- https://openlexicon.fr/datasets-info/
- **https://github.com/frodonh/french-words**
- https://github.com/DanielSWolf/wiki-pronunciation-dict


### Community detection in large networks

Problem: edges file will be > 30GB (just the weights), so it doesn't fit in the RAM.

- https://github.com/Sotera/spark-distributed-louvain-modularity
- https://github.com/GraphChi/graphchi-cpp


### Python virtual env

```
$ python -m venv projectname
$ source projectname/bin/activate
(venv) $ pip install ipykernel
(venv) $ ipython kernel install --user --name=projectname
```


### To consider

- There can exist multiple possible pronunciations -> split for multiple nodes in the graph, e.g. "est (1)" and "est (2)"
- Omit the "ː" for lengthening of a sound
- Some characters are encoded as two Unicode characters, e.g. ɛ̃
- "d" needs lookahead since "dʒ" also exists
- "t" needs lookahead since "tʃ" also exists


### Similarity matrix construction (manually)

- 10,000: Found combinations: 670 / 741 (90.42%)
- 10,000: Found combinations: 476 / 741 (64.24%) (exact length match)
- 30,000: Found combinations: 1062 / 741 (143.32%)

-> higher than 100% due to not taking ordering into account, i.e.
b,d and d,b both exist in the resulting file, so divide percentage by 2
