## Phonetics Graph

This projects aims at defining a metric to measure the similarity between two words based on their phonetic transcription. This metric is calculated for every pair of words in a given list of words. We then process the results to generate a graph where the nodes are the words and the edges are the similarity between the words. Graph clustering algorithms are then applied to the graph to identify groups of words that are phonetically similar.

- International Phonetic Alphabet (IPA)
- https://github.com/open-dict-data/ipa-dict
- https://github.com/xinjli/ucla-phonetic-corpus
- https://archive.phonetics.ucla.edu/
- https://archive.phonetics.ucla.edu/Language/ipa-pop-up-2.html
- [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)


### Python virtual env

```
$ python -m venv projectname
$ source projectname/bin/activate
(venv) $ pip install ipykernel
(venv) $ ipython kernel install --user --name=projectname
```
