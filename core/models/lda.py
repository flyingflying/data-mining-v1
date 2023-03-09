# -*- coding:utf-8 -*-
# Author: lqxu

"""
对 GenSim 库中的 LdaModel 用类似 sklearn 的方式进行封装, 并添加大量的注释, 更加方便使用和理解

gensim version: 4.3.0
"""

import time 
from typing import * 

import numpy as np 

from tqdm import tqdm

from gensim.models import LdaModel
from gensim import matutils


class LDA:
    
    def __init__(
        self, 
        # 基础参数
        num_topics=100,            # "主题" 的个数
        id2word=None,              # 字典, id -> token
        chunksize=2000,            # 每一次读取的数据量, 和内存大小是相关的
        
        # reference: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#training 
        passes=1,                  # 和 epoches 一个意思, 表示数据集迭代的次数
        iterations=50,             # 每一个 epoch 内, 单个文档的迭代次数
        eval_every=10,             # 更新多少次计算 log perplexity, 其计算是非常缓慢的, 在 logging 设置成 debug 模式时, 会输出文档收敛的数量
        
        # 超参数
        update_every=1,            # 0: batch learning, > 1: online learning
        alpha="auto",              # auto 表示: 自动学习
        eta="auto",                # auto 表示: 自动学习
        decay=0.5,                 # 
        offset=1.0,                # 
        gamma_threshold=0.001,     # 
        
        # 其它参数
        minimum_probability=0.0,   # 低于 max(minimum_probability, 1e-8) 的概率为 0, 方便构建稀疏向量
        random_state=None,         # 随机数种子, 主要作用是为了复现
        callbacks=None,            # 回调函数, 用于记录训练时 metrics 值的大小, 使用 visdom 库进行可视化
        dtype=np.float32,          # 数组的类型, 默认 float32
        model=None                 # 模型参数, 用于 load 初始化用的
    ) -> None:

        """
        初始化 LDA 模型, 需要注意的是:
        1. 原版的代码是支持分布式运算的, 采用 Pyro4 库, 配置 ns_conf 参数即可, 这里将这个功能禁止了。目前没有这个资源探索分布式运算, 以后有资源会进行尝试。
        2. 部分参数的含义在上面的注释中已经写了, 剩下的参数等完全理解 LDA 模型后再来补。
        """

        if model is not None:
            self.model = model
            self.is_trained = True
            return 

        self.model = LdaModel(
            num_topics=num_topics, id2word=id2word, chunksize=chunksize, 
            passes=passes, iterations=iterations, eval_every=eval_every, 
            update_every=update_every, alpha=alpha, eta=eta, decay=decay, offset=offset, gamma_threshold=gamma_threshold, 
            minimum_probability=minimum_probability, random_state=random_state, callbacks=callbacks, dtype=dtype
        )
        
        self.is_trained = False
    
    def fit(self, corpus):
        """ 训练 LDA 模型, 需要注意的是这里并没有对数据集进行 shuffle 操作, 需要传入前自行 shuffle。"""

        if self.is_trained:
            raise ValueError("模型已经训练完成了, 如果还要更新, 请调用 partial_fit 函数")

        # 从 gensim 中拷贝出来的
        use_numpy = self.model.dispatcher is not None
        start = time.time()
        self.model.update(corpus, chunks_as_numpy=use_numpy)
        self.model.add_lifecycle_event(
            event_name="created",
            msg=f"trained {self.model} in {time.time() - start:.2f}s",
        )

        self.is_trained = True
    
    def partial_fit(self, corpus, passes=None):
        """ 局部更新, 这里的 corpus 只允许一个, 不允许多个 """
        
        use_numpy = self.model.dispatcher is not None
        self.model.update(corpus, passes=passes, chunks_as_numpy=use_numpy)
        self.is_trained = True

    def transform(self, corpus, memmap_path = None, n_docs = None, progress_bar = True):
        """ 将 BOW/TF 文本向量 转化成 主题文本向量 的形式, 注意这里返回的是 dense 矩阵, 不是 sparse 矩阵, 也不是 BOW 矩阵 """
        
        if not self.is_trained:
            raise RuntimeError("请先调用 fit 函数")
        
        if n_docs is None:
            if not hasattr(corpus, "__len__") and memmap_path is not None:
                raise ValueError("使用 memmap_path 时需要预先知道文档数, 请传入 n_docs 参数")
            
            if hasattr(corpus, "__len__"):
                n_docs = len(corpus)
            
        corpus = tqdm(corpus, desc=f"生成主题向量中", total=n_docs, disable=not progress_bar)
        
        if memmap_path is None:
            doc_topic_dists = matutils.corpus2dense(
                [self.model.get_document_topics(doc, minimum_probability=0.0) for doc in corpus], 
                num_terms=self.model.num_topics
            ).T
            doc_topic_dists = np.ascontiguousarray(doc_topic_dists)

        else:
            # reference: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html 
            doc_topic_dists = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(len(corpus), self.model.num_topics))
            # TODO: 这里应该独立成一个函数, 以减少循环的层数
            for doc_idx, doc_bow in enumerate(corpus):
                for topic_idx, topic_weight in self.model.get_document_topics(doc_bow, minimum_probability=0.0):
                    doc_topic_dists[doc_idx, topic_idx] = topic_weight
            doc_topic_dists.flush()
        
        return doc_topic_dists
    
    def fit_transform(self, corpus):
        """ 将 fit 方法和 transform 方法合并起来 """
        
        if not isinstance(corpus, Iterable):
            raise ValueError("这里的 corpus 需要进行多次迭代, 必须是 Iterable 对象, 不能是 Iterator 对象")
        
        self.fit(corpus=corpus)
        return self.transform(corpus=corpus)
    
    def score(self, corpus, topn=20):
        """ 计算模型的 u_mass 连贯性分数, 对于每一对词 (w^*, w), 计算 log p(w|w^*) 值 """

        if not self.is_trained:
            raise RuntimeError("请先调用 fit 函数")

        num_topics = self.model.num_topics
        topic_scores = self.model.top_topics(corpus=corpus, coherence="u_mass", topn=topn)
        metric = sum([topic_score for topic_idx, topic_score in topic_scores]) / num_topics
        return metric
    
    def save(self, file_path: str):
        """ 保存模型 """

        if not self.is_trained:
            raise RuntimeError("请先调用 fit 函数")
        self.model.save(file_path)
    
    def save_mining_data(self, file_path: str, corpus):
        """ 保存数据挖掘所需要的三个矩阵 """
        
        n_docs, n_topics, vocab_size = len(corpus), self.model.num_topics, self.model.num_terms

        # 主题-词语 概率分布
        topic_word_dists = self.model.get_topics()
        assert topic_word_dists.shape == (n_topics, vocab_size)

        # 文档-主题 概率分布
        doc_topic_dists = self.transform(corpus=corpus)

        # 文档 长度
        doc_lengths = np.array([sum([token[1] for token in doc]) for doc in corpus])
        assert doc_lengths.shape == (n_docs, )
        
        # TODO: doc_topic_dists 支持使用 memmap 的方式进行存储
        np.savez(file_path, topic_word_dists=topic_word_dists, doc_topic_dists=doc_topic_dists, doc_lengths=doc_lengths)
    
    @staticmethod
    def load_mining_data(file_path: str):
        """ 加载数据挖掘的相关矩阵 """
        
        file_obj = np.load(f"{file_path}.npz")
        
        return {
            "topic_word_dists": file_obj["topic_word_dists"], 
            "doc_topic_dists": file_obj["doc_topic_dists"], 
            "doc_lengths": file_obj["doc_lengths"]
        }
    
    @classmethod
    def load(cls, file_path: str):
        """ 加载模型 """

        self = cls(model=LdaModel.load(file_path))
        self.is_trained = True
        return self
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f"LDA(num_topics={self.model.num_topics})"
