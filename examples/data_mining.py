# -*- coding:utf-8 -*-
# Author: lqxu

#%% import modules
from _prepare import *

import os 
import json 
import warnings

import numpy as np 
from gensim.corpora import Dictionary
from tmtoolkit.topicmod import model_stats

from core import LDA 

warnings.filterwarnings(action="ignore")
with warnings.catch_warnings(record=True) as _:
    import pyLDAvis  # 这里输出的警告只能捕获, 不能 filter 掉, 不知道是不是 bug

#%% global settings 
lda_model_dir = os.path.join(model_dir, f"lda_model")

dict_path = os.path.join(lda_model_dir, "dict.txt")
model_path = os.path.join(lda_model_dir, "model.{}.bin")
mining_path = os.path.join(lda_model_dir, "mining.{}.bin")

candidate_num_topics = [15, 50, ]


#%% load dictionary and build vocab array 
def build_vocab_array(dict_path: str) -> np.ndarray:
    dictionary: Dictionary = Dictionary.load_from_text(dict_path)
    
    vocab_list = [
        token for token, token_id in sorted(dictionary.token2id.items(), key=lambda item: item[1])
    ]
    
    return np.array(vocab_list)


vocab_array = build_vocab_array(dict_path)

#%% 加载挖掘数据

all_mining_data = []

for num_topics in candidate_num_topics:
    all_mining_data.append(
        LDA.load_mining_data(mining_path.format(num_topics))
    )


# %% 输出 salient 词语

# 从效果来看, num_topics 影响不是很大
for num_topics, mining_data in zip(candidate_num_topics, all_mining_data):

    print("word saliency in LDA model with {} topics".format(num_topics))
    print(model_stats.most_salient_words(
        vocab=vocab_array, 
        topic_word_distrib=mining_data["topic_word_dists"],
        doc_topic_distrib=mining_data["doc_topic_dists"], 
        doc_lengths=mining_data["doc_lengths"], 
        n=50
    ).tolist())

# %% 输出 distinct 词语

# 从效果来看, num_topics 影响还是很大的
for num_topics, mining_data in zip(candidate_num_topics, all_mining_data):

    print("word distinctiveness in LDA model with {} topics".format(num_topics))
    print(model_stats.most_distinct_words(
        vocab=vocab_array, 
        topic_word_distrib=mining_data["topic_word_dists"],
        doc_topic_distrib=mining_data["doc_topic_dists"], 
        doc_lengths=mining_data["doc_lengths"], 
        n=50
    ).tolist())

# %% 输出可能性大的词语

# 从效果来看, num_topics 影响非常小
for num_topics, mining_data in zip(candidate_num_topics, all_mining_data):

    print("word probable in LDA model with {} topics".format(num_topics))
    print(model_stats.most_probable_words(
        vocab=vocab_array, 
        topic_word_distrib=mining_data["topic_word_dists"],
        doc_topic_distrib=mining_data["doc_topic_dists"], 
        doc_lengths=mining_data["doc_lengths"], 
        n=50
    ).tolist())

# %%

for num_topics, mining_data in zip(candidate_num_topics, all_mining_data):
    vis_obj = pyLDAvis.prepare(
        topic_term_dists=mining_data["topic_word_dists"], 
        doc_topic_dists=mining_data["doc_topic_dists"], 
        doc_lengths=mining_data["doc_lengths"], 
        vocab=vocab_array, 
        # term frequency 实际上已经是弃用参数了, 只要传入一个 shape 满足要求的数组即可
        term_frequency=np.ones_like(vocab_array),
    )
    
    pyLDAvis.save_html(vis_obj, f"./output/vis/lda_{num_topics}.html")
    
# %%

for num_topics, mining_data in zip(candidate_num_topics, all_mining_data):
    
    word_weights = model_stats.word_distinctiveness(
        topic_word_distrib=mining_data["topic_word_dists"], 
        p_t=model_stats.marginal_topic_distrib(doc_topic_distrib=mining_data["doc_topic_dists"], doc_lengths=mining_data["doc_lengths"])
        # p_t=np.full(shape=num_topics, fill_value=1. / num_topics)
    )

    word_weights = (word_weights - word_weights.min()) / (word_weights.max() - word_weights.min())

    word_weights_dict = {word: word_weight for word, word_weight in zip(vocab_array, word_weights)}

    with open(f"./weights/lda_{num_topics}_word_weights.json", "w", encoding="utf-8") as writer:
        json.dump(word_weights_dict, writer, ensure_ascii=False, indent=0)

# %%
