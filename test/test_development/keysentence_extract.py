# -*- coding:utf-8 -*-
# Author: lqxu
# 9.4 关键句提取

import math
import numpy as np
from typing import *
from collections import Counter


_BM25_K1 = 1.5  # 调节因子 k_1, 控制 TF 饱和度用的, HanLP 设置为 1.5, 一般默认是 1.2
_BM25_B = 0.75  # 调节因子 b, 控制文本长度影响用的, HanLP 和默认保持一致为 0.75
_TEXT_RANK_D = 0.85  # 平滑因子
_MIN_DIFF = 0.001  # 两次迭代参数最小差值


def extract_key_sentence(docs: List[List[str]], max_iter: int = 200) -> List[int]:
    # ## 语料库中的句子数
    docs_num = len(docs)
    # ## 语料库的平均长度
    avg_dl = sum(len(doc) for doc in docs) / docs_num
    # ## 将句子从 List[str] 的形式转化为 Bag-of-Words 的形式, 也就是计算每一个词的词频(TF)
    bow_docs = [Counter(doc) for doc in docs]
    # ## 统计词语的文档频率(DF)
    words_df = Counter(word for bow_doc in bow_docs for word in bow_doc)
    # ## 统计词语的逆文档频率(IDF)
    words_idf = {
        word: math.log(docs_num - df + 0.5) - math.log(df + 0.5)
        for word, df in words_df.items()
    }
    # ## 计算权重矩阵
    # 在提取关键句的时候, 我们认为每一个句子都是一个节点, 并且所有句子之间都是相连的, 在这种情况下, 我们用矩阵来表示图
    # 首先要注意的一点是, BM25 算法是衡量两个句子之间的相似度的, 但是 BM25(A, B) != BM25(B, A)
    # 根据 BM25 公式可以, 对于 BM25(A, B) 而言, B 是查询语句, 根据 TextRank 公式可知, BM25(A, B) 表示的是由 B 指向 A
    # 因此计算过程是由 查询文档Q 指向 被查询的文档 D
    # 我们将每一个节点的分数用列向量的形式来表示, 权重矩阵第 i 行第 j 列表示的是由 v_j 指向 v_i 的权重, v_j 是查询文档
    weights = np.zeros(shape=(docs_num, docs_num), dtype=np.float64)
    for idx in range(docs_num):
        for idj in range(docs_num):
            if idx == idj:  # 没有自循环
                continue
            weight = 0.0
            for word in docs[idx]:  # 遍历查询语句中的词语
                if word in bow_docs[idj]:  # 如果查询的词语在文档当中
                    dl = len(docs[idx])
                    tf = bow_docs[idx][word]
                    numerator = words_idf[word] * tf * (_BM25_K1 + 1)  # 分子
                    denominator = (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * (dl / avg_dl)))  # 分母
                    weight += (numerator / denominator)
            weights[idx, idj] = weight
    with np.errstate(divide="ignore", invalid="ignore"):  # 有可能会有 除0 问题或者 0除0 问题
        # 归一化, 我们需要对出链归一化, 这其实已经和之前 PageRank 当中矩阵的例子相对应了
        weights = weights / weights.sum(axis=0)  # 纵轴是 axis=0
        np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.ones(docs_num)
    for _ in range(max_iter):
        next_scores = (1 - _TEXT_RANK_D) + _TEXT_RANK_D * (weights @ scores)
        max_diff = np.max(next_scores - scores)
        scores = next_scores
        if max_diff < _MIN_DIFF:
            break
    return scores.argsort()[::-1]


if __name__ == '__main__':
    # ## 测试数据
    demo_text = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
                "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
                "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
                "严格地进行水资源论证和取水许可的批准。"
    # ## 第一步: 分句, 切分符号为: ，,。:：“”？?！!；;
    # 分句后的结果是:
    sentences = [
        '水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露',
        '根据刚刚完成了水资源管理制度的考核',
        '有部分省接近了红线的指标',
        '有部分省超过红线的指标',
        '对一些超过红线的地方',
        '陈明忠表示',
        '对一些取用水项目进行区域的限批',
        '严格地进行水资源论证和取水许可的批准'
    ]
    # ## 第二步: 分词 + 词性标注 + 去停用词
    # HanLP 标准分词器处理后的结果:
    test_docs_v1 = [
        ['水利部', '水资源', '司', '司长', '陈明忠', '9月', '国务院新闻办', '举行', '新闻', '发布会', '透露'],
        ['刚刚', '完成', '水资源', '管理制度', '考核'],
        ['部分', '省', '接近', '红线', '指标'],
        ['部分', '省', '超过', '红线', '指标'],
        ['超过', '红线', '地方'],
        ['陈明忠', '表示'],
        ['取', '用水', '项目', '进行', '区域', '限'],
        ['严格', '进行', '水资源', '论证', '取水', '许可', '批准'],
    ]
    test_docs_v2 = [
        ['水利部', '水资源', '司', '司长', '陈明忠', '9月', '国务院', '新闻办', '举行', '新闻', '发布会', '透露'],
        ['刚刚', '完成', '水资源', '管理', '制度', '考核'],
        ['部分', '省', '接近', '红线', '指标'],
        ['部分', '省', '超过', '红线', '指标'],
        ['超过', '红线', '地方'],
        ['陈明忠', '表示'],
        ['取', '用水', '项目', '进行', '区域', '限'],
        ['严格', '进行', '水资源', '论证', '取水', '许可', '批准'],
    ]
    # ## 第三步: 提取关键词
    # np.set_printoptions(linewidth=np.inf)
    # 不同的分词结果计算出来是不一样的, 在 HanLP 中如果自己训练模型了, 分词结果就很可能不一样
    print(
        [sentences[i] for i in extract_key_sentence(test_docs_v1)[:3]]
    )
    print(
        [sentences[i] for i in extract_key_sentence(test_docs_v2)[:3]]
    )
    
    import time 
    s = time.time()
    print(extract_key_sentence(test_docs_v1))
    print(time.time() - s)
    
    print(extract_key_sentence(test_docs_v2))
