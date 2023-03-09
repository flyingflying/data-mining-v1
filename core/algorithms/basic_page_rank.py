# -*- coding:utf-8 -*-
# Author: lqxu

""" 
实现 Page Rank 算法

这一版的实现方便扩展, 主要是为了和 TextRank, Salience Rank, Topical PageRank 适配

关于 Page Rank 算法的细节, 建议参考维基百科

给出一些中文资料:
   1. https://www.cnblogs.com/LittleHann/p/9969955.html 
   2. https://www.zybuluo.com/evilking/note/902585 
   3. https://zhuanlan.zhihu.com/p/500097430 

TODO: 学习随机过程, 尝试理解马尔可夫链的收敛性

注意 LexRank 和 Page Rank 是不一样的, 其通过提取特征向量作为最终的结点分数, 不能归为这一系列。
关于 LexRank, 参考: https://iq.opengenus.org/lexrank-text-summarization/ 
"""

import numpy as np 
from numpy import ndarray


def page_rank(
        edge_weights: ndarray, node_weights: ndarray = None, init_score: ndarray = None, is_text_rank: bool = False,
        damping_factor: float = 0.85, max_iter: int = 200, min_diff: float = 0.001
    ):

    """
    用 page rank 算法计算每一个结点的分数值, 不进行排序
    
    edge_weights[i, j] 表示从 i 结点指向 j 结点的权重值, 也就是说, 一行表示一个结点的所有出链
    一般情况下, 在 NLP 中, 构建的图是没有自循环的, 也就是 edge_weights 的对角线应该都是 0。
    如果是无向图, 那么结点之间应该是双向链接, 即 edge_weights 的上三角和下三角是对称的。
     

    Args:
        edge_weights (ndarray): 边与边之间的权重值, shape [n_nodes, n_nodes]
        node_weights (ndarray, optional): 结点的权重值, shape: [n_nodes, ]
        is_text_rank (bool, optional): 是否是 text rank 算法 (text rank 使用的是有偏差的公式, 详细见 page rank 的维基百科页面)
        damping_factor (float): _description_
        max_iter (int, optional): _description_. Defaults to 200.
        min_diff (float, optional): _description_. Defaults to 0.001.
    """

    n_nodes = edge_weights.shape[0]

    if node_weights is None and is_text_rank:
        node_weights = np.ones(shape=n_nodes)

    elif node_weights is None and not is_text_rank:
        node_weights = 1. / np.full(shape=n_nodes, fill_value=n_nodes)  # 默认均匀分布

    else:
        node_weights = node_weights / node_weights.sum()  # 概率标准化

    if node_weights.shape != (n_nodes, ) or edge_weights.shape != (n_nodes, n_nodes):
        raise ValueError("node_weights 或者 edge_weights 的 shape 有问题")
    if np.any(np.isnan(node_weights)) or np.any(np.isinf(node_weights)):
        raise ValueError("发现 nan 或者 inf, 请检查 node_weights 的值")
    if np.any(np.isnan(edge_weights)) or np.any(np.isinf(node_weights)):
        raise ValueError("发现 nan 或者 inf, 请检查 edge_weights 的值")

    # 自循环检测
    # np.all(np.abs(np.diag(edge_weights)) < 1e-8):

    with np.errstate(divide="ignore", invalid="ignore"):  # a / 0 = inf; 0 / 0 = nan
        # 需要注意的是, 概率标准化是在 **入链** 的权重上 (axis=0), 不是在 **出链** 的权重上 (axis=1)
        # 这里很容易出错 !!! 
        edge_weights = edge_weights / edge_weights.sum(axis=0)

        if is_text_rank:  # hanlp 中 text rank 的实现中, 没有入链的结点就让其没有入链 ... 
            np.nan_to_num(edge_weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:  # page rank 中对于没有 入链 的结点, 认为其和所有的结点相连
            np.nan_to_num(edge_weights, copy=False, nan=1. / node_weights, posinf=1. / node_weights, neginf=1. / node_weights)

    if init_score is None:
        prev_score = score = node_weights
    else:
        prev_score = score = init_score

    for _ in range(max_iter):
        score = (1 - damping_factor) * node_weights + damping_factor * (edge_weights @ score)

        if np.max(np.abs(score - prev_score)) < min_diff:
            break 

        prev_score = score

    return score
