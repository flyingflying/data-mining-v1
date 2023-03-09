# -*- coding:utf-8 -*-
# Author: lqxu
# 9.2.3 关键词抽取 -> TextRank

import math
import heapq
from typing import *
from collections import deque

__all__ = ["extract_keyword_by_textrank", ]

_DAMPING_FACTOR = 0.85  # 阻尼系数
_MIN_DIFF = 0.001  # 两次迭代之间的最小差值


"""
PageRank 核心思想介绍: 
1. PageRank 是一种对 web网页 计算分数的算法。其想法是用每一个网页的外链来计算分数。
   这里用符号 C 表示是某一网页, I 表示指向 C 的网页, 根据 PageRank 公式我们可以知道, 如果要网页 C 的分数尽可能地高, 那么:
      a. 网页 I 的数量要尽可能地多, 也就是指向网页 C 的数量要尽可能地多; [越优秀的网站指向其的链接就越多]
      b. 网页 I 指向其它网页的数量要尽可能地少; [资源站汇总了其它网站的信息, 要减少这种网站对评分的影响]
      c. 网页 I 的分数要尽可能地高。[优秀的网站往往指向优秀的网站]
2. PageRank 的计算:
   从上面可以看出, 如果网页 C 的评分是依赖于网页 I 的分数, 那么怎么去算分数呢, 这里用到了 马尔可夫过程的收敛性。
   我们假设网民浏览网页时会随机通过外链选择下一个网页, 并且这个过程仅仅依赖当前网页的分数, 不依赖之前所浏览过的网页。
   我们首先随机初始化每一个网页的评分, 并且将网页 I 的数量变成转移概率矩阵(假设访问下一个网页的概率是相同的, 那么转移概率就是 1/出链数量), 
   然后反复的用网页评分去乘以转移概率, 根据马尔可夫过程的收敛性, 最终网页的评分会趋近于某一定值, 然后结束计算, 即可得到每一个网页的分数。
3. PageRank 的问题: 
   1) spider traps(自循环节点) 和 dead ends(不存在外链的节点)问题: 解决办法是将公式平滑处理, 因此有了阻尼系数d。具体的数学分析就先不分析了。
   2) 没有过滤掉部分网页, 容易被外部的人操纵分数, 解决办法是使用 TrustRank 算法。

TextRank 核心思想介绍:
1. TextRank 实际上将 PageRank 算法中的 web网页 变成了词语, 也就是给每一个词语打分。
   认为某个中心词附近一定范围内的词都指向中心词, 然后就可以将句子变成一张图了。
   对于转移概率, 我看有的公式会给不同的边不同的权重, 目测是根据词频来计算的, HanLP 中采用的是和 PageRank 算法中一样的方式, 平等地对待每一个出链。
2. 用符号 C 表示某一个中心词, I 在中心词附近的词语, 也就是指向 C 的词语, 如果希望中心词 C 的分数尽可能地高, 那么:
   a. 中心词 C 附近的词要尽可能地丰富, 也就是 I 的种类要尽可能地多; [关键词附近的词语种类要丰富]
   b. I 附近地词要尽可能地不丰富, 也就是 I 指向的词要尽可能地少; [减少常用词对评分的影响]
   c. I 的分数要尽可能地高。 [关键词旁边往往也是关键词]
   
Reference: 
   1. https://www.cnblogs.com/LittleHann/p/9969955.html
   2. https://www.zybuluo.com/evilking/note/902585
"""


def _sigmoid(value: float) -> float:
    return 1. / (1. + math.exp(-value))


def extract_keyword_by_textrank(words: List[str], size: int, max_iter: int = 200) -> List[str]:
    # ## 构建词图, 这里使用的是链表的形式, 注意这里的 value 值表示指向 key 的节点集合, 而不是一般的表示由 key 指向的节点集合
    words_grapy: Dict[str, set] = {word: set() for word in words}
    pre_words_queue = deque()  # 中心词前面的词语队列
    for center_word in words:
        if len(pre_words_queue) > 4:  # 窗口半径设置为 4
            pre_words_queue.popleft()
        for pre_word in pre_words_queue:
            if pre_word == center_word:  # 避免spider traps(自循环)的情况发生
                continue
            words_grapy[center_word].add(pre_word)  # pre_word 指向 center_word
            words_grapy[pre_word].add(center_word)  # 关系是相互的, 这样就没必要再反向遍历了
        pre_words_queue.append(center_word)

    # ## ### 计算分数
    # ## 第一步: 初始化每一个节点的分数值, 可以采用随机数的方式, HanLP 中采用的是 sigmoid(节点的入边数)
    # 节点的入边 指的是 指向当前节点边的数量
    # 仔细想了一下, 这里用 sigmoid 很奇怪, 还不如初始化为1, 因为窗口半径设置为是4, 而 sigmoid 函数在值为 4 的时候就是 0.98+ 了
    cur_score: Dict[str, float] = {node: _sigmoid(len(from_nodes)) for node, from_nodes in words_grapy.items()}
    print(cur_score)
    # ## 第二步: 不断迭代更新每一个词语的分数, 直接按照公式来就可以了
    for _ in range(max_iter):
        max_diff = 0.0  # max_diff 用于记录 所有节点 当前值和更新后值之间 的差值 的最大值
        next_score: Dict[str, float] = {node: 1 - _DAMPING_FACTOR for node in words_grapy.keys()}
        for node, from_nodes in words_grapy.items():
            for from_node in from_nodes:
                if from_node == node or len(words_grapy[from_node]) == 0:
                    continue
                next_score[node] += (_DAMPING_FACTOR / len(words_grapy[from_node]) * cur_score[from_node])  # 公式计算
            max_diff = max(
                max_diff,
                abs(cur_score[node] - next_score[node])  # 更新后与更新前分数的差值
            )
        # 更新分数
        cur_score = next_score
        if max_diff < _MIN_DIFF:
            break

    # ## 根据分数进行排序
    ret = heapq.nlargest(size, cur_score.items(), key=lambda item: item[1])
    return [r[0] for r in ret]


if __name__ == '__main__':
    # TextRank 关键词提取的流程
    demo_text = "程序员(英文Programmer)是从事程序开发、维护的专业人员。" \
                "一般将程序员分为程序设计人员和程序编码人员，" \
                "但两者的界限并不非常清楚，特别是在中国。" \
                "软件从业人员分为初级程序员、高级程序员、系统" \
                "分析员和项目经理四大类。"
    # ## 第一步: 分词并去停用词
    # 注意使用 TextRank 时, 去停用词这一步非常重要, 不仅要根据停用词表过滤, 还需要根据词性去过滤, 如果想要效果好, 这一步需要下功夫
    # 这里直接将 HanLP 处理过后的 demo_text 放在这里, 可以有直观的认识和理解
    # HanLP v1 的默认分词器是 ViterbiSegment, 也就是 二元分词, 词性标注采用的是 HMM, 这样效率会比较高, 但是效果会不太好
    # 比方说默认分词器会将 "初级" 打上 区分词(b) 的标签, 而 "高级" 打上 形容词(a) 的标签, 结果就是会过滤掉 "初级" 保留 "高级"
    # 但是从我们的直观上理解, 至少这两个词的词性应该是相同的, 要么都保留, 要么都去掉
    segment_and_filter_results = [
        '程序员', '英文', 'Programmer', '从事', '程序', '开发', '维护', '专业', '人员', '程序员', '分为', '程序设计', '人员',
        '程序', '编码', '人员', '界限', '并不', '非常', '清楚', '特别是在', '中国', '软件', '从业人员', '分为',
        '程序员', '高级', '程序员', '系统分析员', '项目经理', '四大'
    ]
    # ## 第二步: 用 TextRank 算法提取关键词
    keywords = extract_keyword_by_textrank(segment_and_filter_results, 5)
    print(keywords)  # ['程序员', '分为', '程序', '人员', '软件']
    # 这里输出有"分为", "人员" 这样的词, 还需要继续过滤
    
    from core.algorithms import text_rank_for_keyword
    
    print(text_rank_for_keyword(segment_and_filter_results))
