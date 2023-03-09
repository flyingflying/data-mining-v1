# -*- coding:utf-8 -*-
# Author: lqxu

""" 将 jionlp 中的 idf 权重和 word distinctiveness 权重初始化 """

import os 
import json
from itertools import chain

import jionlp 

import numpy as np


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    return data


def load_topic_weights():
    # 加载数据
    data_dir = os.path.join(
        os.path.dirname(jionlp.__file__), "dictionary"
    )
    word_topic_dists_dict = load_json(os.path.join(data_dir, "topic_word_weight.json"))
    topic_word_dists_dict = load_json(os.path.join(data_dir, "word_topic_weight.json"))

    # 构建词表
    vocab1 = set(word_topic_dists_dict.keys())
    vocab2 = set(chain(*[value.keys() for value in topic_word_dists_dict.values()]))
    vocab = sorted(vocab1.union(vocab2))
    token2id = {token: id_ for id_, token in enumerate(vocab)}

    # 构建矩阵 (能用矩阵运算的都建议使用矩阵运算)
    n_topics = len(topic_word_dists_dict)
    vocab_size = len(vocab)

    word_topic_dists = np.zeros(shape=(vocab_size, n_topics), dtype=np.float64)
    topic_word_dists = np.zeros(shape=(n_topics, vocab_size), dtype=np.float64)

    for token, topic_dists_dict in word_topic_dists_dict.items():
        token_id = token2id[token]
        topic_dists = [topic_dists_dict.get(str(topic_id), 0.0) for topic_id in range(n_topics)]
        word_topic_dists[token_id] = topic_dists

    for topic, word_dists_dict in topic_word_dists_dict.items():
        topic_id = int(topic)
        for token, prob in word_dists_dict.items():
            token_id = token2id[token]
            topic_word_dists[topic_id][token_id] = prob

    # 不可能为 1, jionlp 的作者数据没有保存全
    print(word_topic_dists.sum(axis=1).min())
    print(topic_word_dists.sum(axis=1).min())

    return token2id, word_topic_dists, topic_word_dists


def load_word_weights():
    """ 加载根据 topic 计算出来的词权重, 只需要用到 p(t|w), 上面的过程有一些繁琐了 """
    data_dir = os.path.join(
        os.path.dirname(jionlp.__file__), "dictionary"
    )
    word_topic_dists_dict = load_json(os.path.join(data_dir, "topic_word_weight.json"))

    vocab_list = sorted(set(word_topic_dists_dict.keys()))
    vocab_dict = {token: id_ for id_, token in enumerate(vocab_list)}

    vocab_size = len(vocab_list)
    n_topics = 100  # 这里是作者加载 word_topic_weight.json 文件的作用

    word_topic_dists = np.zeros(shape=(vocab_size, n_topics), dtype=np.float64)

    for token, topic_dists_dict in word_topic_dists_dict.items():
        token_id = vocab_dict[token]
        # 如果缺失, 使用 1e-5 作为概率的默认值
        topic_dists = [topic_dists_dict.get(str(topic_id), 1e-5) for topic_id in range(n_topics)]
        word_topic_dists[token_id] = topic_dists

    # 假设 topic 的概率分布是均匀分布
    topic_dists = np.full(fill_value=1. / n_topics, shape=n_topics,)
    # 这里实际上计算的是每一个 word 的区分度
    # 一般情况下都是用 np.log 函数, 不知道为什么 jionlp 的作者使用 np.log2
    word_distinctiveness = (word_topic_dists * (np.log2(word_topic_dists) - np.log2(topic_dists))).sum(axis=1)

    min_value = word_distinctiveness.min()
    max_value = word_distinctiveness.max()
    word_weights = (word_distinctiveness - min_value) / (max_value - min_value)

    word_weights_dict = {word: weight for word, weight in zip(vocab_list, word_weights)}
    word_weights_dict = {word: weight for word, weight in sorted(word_weights_dict.items(), key=lambda item: item[1])}
    return word_weights_dict


def load_idf_weights():
    """ 加载每一个词的 idf 权重 """
    idf = jionlp.idf_loader()
    return idf


if __name__ == '__main__':
    # with open("weights/default_idf.json", "w", encoding="utf-8") as writer:
    #     json.dump(load_idf_weights(), writer, ensure_ascii=False, indent=0)

    # with open("weights/default_word_weights.json", "w", encoding="utf-8") as writer:
    #     json.dump(load_word_weights(), writer, ensure_ascii=False, indent=0)
    
    with open("weights/pos_combine_weights.json", "w", encoding="utf-8") as writer:
        with open(os.path.join(os.path.dirname(jionlp.__file__), "algorithm/keyphrase/pos_combine_weights.json"), "r", encoding="utf-8") as reader:
            json.dump(json.load(reader), writer, ensure_ascii=False, indent=0)
