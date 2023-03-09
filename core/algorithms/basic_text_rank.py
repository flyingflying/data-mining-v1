# -*- coding:utf-8 -*-
# Author: lqxu

""" 
实现 TextRank, 从旧代码改编过来的, 旧代码是我在读何晗的 《自然语言处理入门》 时写的。
"""

from typing import * 

import math 

import numpy as np 

from .basic_page_rank import page_rank


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def text_rank_for_summary(tokens_list: List[List[str]], bm25_k1 = 1.5, bm25_b = 0.75, page_rank_d = 0.85, min_diff = 0.001, max_iter = 200):
    """
    TextRank 算法提取摘要
    
    从 HanLP v1 中复现的方法, 复现的主要原因是 HanLP 中的摘要提取分句分的太细了, 导致摘要的可读性非常差 !!! 

    Args:
        tokens_list (List[List[str]]): 分句并分词好的文档
        bm25_k1 (float, optional): BM25 算法中的 k1 参数. Defaults to 1.5, ES 中设置的是 1.2, 一般取值在 [1.2, 2], TF 的饱和度
        bm25_b (float, optional): BM25 算法中的 b 参数. Defaults to 0.75. 文本长度的平滑因子
        page_rank_d (float, optional): Page Rank 算法中的平滑因子. Defaults to 0.85.
        min_diff (float, optional): 两次迭代之间的最小差值, 用于提前结束迭代. Defaults to 0.001.
        max_iter (int, optional): 最大的迭代次数. Defaults to 200.
    """
    
    # ## step1: 根据 BM25 算法计算句子之间的权重值
    
    # 语料库的总文档数
    n_sentences = len(tokens_list)

    # 语料库文档的平均长度
    avg_dl = sum(len(tokens) for tokens in tokens_list) / n_sentences
    
    # 将 sentence 从 tokens 的形式转化成 tf_dict 的形式
    tf_dicts = [Counter(tokens) for tokens in tokens_list]

    # 计算语料库每一个词的 df 值
    token_df_dict = Counter(token for tokens in tokens_list for token in set(tokens))  # 两层循环

    # 计算语料库中每一个词的 idf 值
    token_idf_dict = {
        token: math.log(n_sentences - df + 0.5) - math.log(df + 0.5)
        for token, df in token_df_dict.items()
    }

    weights = np.zeros(shape=(n_sentences, n_sentences), dtype=np.float64)

    """
    BM25 公式是: BM25(query, document): 计算 query 中每一个 token 和 document 的相关性, 然后加起来作为 query 和 document 的相关性
    
    在 Text Rank 算法中, A -> B 的边权重值是 BM25(A, B), 也就是说 A 是 query, B 是 document 
    
    对应下面的代码中, idx / tokens 表示 query, idj / tf_dict 表示 document。注意 tokens 和 tf_dict 都是句子的某一种表示 !!! 
    
    TODO: 优化代码样式, 将 Node 概念单独拿出来, 句子的三种表示形式: text / tokens / tf_dict 都放在数据类中
    
    TODO: BM25 的计算可以使用向量化编程的方式吗 ? 
    """

    for idx, tokens in enumerate(tokens_list):
        for idj, tf_dict in enumerate(tf_dicts):
            if idx == idj:  # 没有自循环
                continue

            weight = 0.0
            dl = len(tokens)
            for token in tokens:  # 遍历查询语句中的词语
                if token not in tf_dict:
                    continue

                tf = tf_dict[token]
                numerator = token_idf_dict[token] * tf * (bm25_k1 + 1)  # 分子
                denominator = (tf + bm25_k1 * (1 - bm25_b + bm25_b * (dl / avg_dl)))  # 分母
                weight += (numerator / denominator)
            weights[idx, idj] = weight
    
    scores = page_rank(weights, is_text_rank=True, damping_factor=page_rank_d, max_iter=max_iter, min_diff=min_diff)
    
    return np.argsort(scores).tolist()[::-1]


def text_rank_for_keyword(tokens: List[str], window_size: int = 4, page_rank_d = 0.85, min_diff = 0.001, max_iter = 200):
    id2token = list(set(tokens))
    token2id = {token: index for index, token in enumerate(id2token)}

    weights = np.zeros(shape=(len(id2token), len(id2token)), dtype=np.float32)
    
    for idx, center_token in enumerate(tokens):
        for context_token in tokens[idx+1: idx+window_size+1]:
            if center_token == context_token:
                continue
            
            center_token_id = token2id[center_token]
            context_token_id = token2id[context_token]
            
            # 这里是无向图, 不是有向图, 因此要加双向边
            # 这里的权重就是 1.0, 和 page rank 中保持一致, 不是 word co-current 的数量
            weights[center_token_id][context_token_id] = 1.0
            weights[context_token_id][center_token_id] = 1.0
    
    init_score = _sigmoid(weights.sum(axis=1))
    
    scores = page_rank(weights, init_score=init_score, is_text_rank=True, damping_factor=page_rank_d, max_iter=max_iter, min_diff=min_diff)
    
    results = [id2token[idx] for idx in np.argsort(scores)[::-1].tolist()]
    
    return results
