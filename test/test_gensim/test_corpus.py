
import os 

from gensim.corpora import *

from gensim.corpora.sharded_corpus import ShardedCorpus


corpus = [
    [(10, 5), (8, 3.9)],
    [(1, 4), (6, 3.5)],
    [(2, 2)],
    [(3, 4)],
]


save_dir = os.path.dirname(__file__)

MmCorpus.serialize(os.path.join(save_dir, "corpus.mm"), corpus=corpus.__iter__())

mm_corpus = MmCorpus(os.path.join(save_dir, "corpus.mm"))

print(mm_corpus)

# BleiCorpus.serialize(os.path.join(save_dir, "corpus.blei"), corpus)

# SvmLightCorpus.serialize(os.path.join(save_dir, "corpus.svmlight"), corpus)

# LowCorpus.serialize(os.path.join(save_dir, "corpus.low"), corpus)

# UciCorpus.serialize(os.path.join(save_dir, "corpus.uci"), corpus)

# MalletCorpus.serialize(os.path.join(save_dir, "corpus.mallet"), corpus)
