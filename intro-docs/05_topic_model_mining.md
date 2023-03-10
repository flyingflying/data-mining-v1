
# LDA 数据挖掘和可视化

[TOC]

## 1. 简介

首先, 我们统一下术语:

+ term 表示词语, 常用的其它形式包括但不限于: token, word, word type 等等
+ doc 表示的是一篇文档, 每一篇文档都是一个 word 列表
+ topic 表示主题, 或者是潜在语义, 每一个 topic 是一组词表的权重值, 其形式和 idf 是一致的, 在概率主题模型中, 其是词表中所有词的概率分布

按照 BoW 的方式, 我们用 TF (Term Frequency) 的形式对 doc 进行向量化操作, 可以得到 dtm (document-term matrix)。

通过 LDA 模型, 我们可以得到 doc_topic_dists 和 topic_term_dists 两个 **条件概率** 矩阵:

+ doc_topic_dists 表示的是每一个 document 的 topic 概率分布, 可以记作 $p(t|d)$
+ topic_term_dists 表示的是每一个 topic 的词表概率分布, 可以记作 $p(w|t)$

对于每一个 term, 有两种统计频数的方式:

+ TF (Term Frequency): term 在语料库中出现的总词数
+ DF (Document Frequency): term 在多少个 document 中出现

不理解 LDA 的过程是没有关系的, 但一定要知道其功能是什么。上面就属于 LDA 的功能。接下来我们来看根据上述内容我们可以进行什么样的计算, 从而帮助我们进行数据挖掘。

我们定义 DL (Document Lengths) 为文档长度, 即一个 document 中有多少个 term 。

下面所说的归一化操作指的是: 每一个数据除以这组数据的和。归一化后的数据和为 1。

**假设一**: doc 的概率分布和 DL 的概率分布是一致的。

根据假设一, 我们对所有文档的 DL 进行归一化操作, 就可以得到每一个 doc 发生的概率, 记作 $p(d)$ 。在已知 $p(d)$ , $p(t|d)$ 和 $p(w|t)$ 的情况下, 根据 **条件概率公式** 和 **边缘概率公式**, 我们可以求出 $p(t)$, $p(w)$ 和 $p(t|w)$ 。

除此之外, 我们还可以根据其去 **估算** 每一个 topic 下每一个 term 发生的频数, 我们记作 $\mathbf{ETF}(t)$; 也可以 **估算** 每一个 term 发生的频数, 记作 ETF (Estimated Term Frequency), 具体的计算方式如下:

+ 我们假设 DL 就是每一个 doc 发生的频数, 那么和 doc_topic_dists 相乘后就可以得到每一个 doc 下每一个 topic 发生的频数, 然后对 document 维度求和就可以得到每一个 topic 的总频数
+ 同样的方法, 我们将 topic 的总频数和 topic_term_dists 相乘后就可以得到每一个 topic 下每一个 term 发生的频数, 如果对 topic 维度求和, 就可以得到每一个 term 发生的频数 (ETF)

ETF 和 TF 在概念上是一致的, 但是计算方式是不一致的。我并没有找到假设一的理论依据, 是根据 tmtoolkit 和 pyLDAvis 代码总结的。如果以后找到了会另外补上。

**假设二**: 所有 topic 发生的概率是相同的。

基于假设二, 我们可以直接通过 $p(w|t)$ 去计算 $p(t|w)$, 直接对 topic_term_dists 的 term 维度进行归一化操作即可。我们将通过这种方式计算出来的每一个 term 下每一个 topic 的概率分布记作 $p^{\prime} (t|w)$ 。

在后续的计算中, 大多数情况下都是使用的假设一, 很少情况下会用假设二, 有时甚至还会混着用。(仔细想想看, 其实 $p(w)$ 可以直接从语料库中获得)

## 2. 相关性分数

这部分主要参考论文: [LDAvis: A method for visualizing and interpreting topics](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)

topic 是一个抽象的概念, 怎么去解释它呢? 答案是用其相关性分数最高的 topN 个词来解释。如何计算 topic 和 term 的相关性分数呢?

最常见的是直接用每一个 topic 下词的权重作为相关性分数, 对应 LDA 中是用 $p(w|t)$ 来表示。

但是上述方法会有问题, 那就是如果一个 term 在语料库中出现的次数非常多, 其很可能会在多个主题中的权重都很高, 于是又有人提出了一种新的计算相关性分数的方式:

$$
lift(w, t) = \frac{p(w|t)}{p(w)} \tag{3}
$$

读过前面的文章话, 这个公式应该就不难理解了。但是 lift 的问题也很大。那就是部分 topic 可能会包含一些独有的无意义词, 比方说如果 "平方米" 这个词只在和 "房屋" 相关的文章中出现, 并且出现的次数也不多, LDA 模型给予的权重也很低, 但是 lift 值可能会非常高。

怎么办呢? 答案就是引入 lambda, LDAvis 给出的计算 topic 和 term 之间的相关性公式如下:

$$
r(w, t) = \lambda \cdot \log p(w|t) + (1 - \lambda) \cdot \log lift(w, t) \tag{4}
$$

对 $p(w|t)$ 和 lift 取对数的原因是让两者可以相加。如果读过前面的文章, 这个公式应该不难理解。相关性分数自然是越大越好。

在 gensim 库以及很多其它地方, 都是直接使用 topic 下 term 的权重作为相关性分数的, 这样更加普适。对非概率主题模型, 就无法这样计算相关性分数了。

论文作者给出的最佳 lambda 值为 0.6, 也提供了可视化方式来自己选择 lambda 值, 这个之后再说。需要注意的是, 相关性分数的数值不具备参考性, 一般用于排序, 取 topN 个词。

## 3. word saliency

这部分主要参考论文: [Termite: Visualization Techniques for Assessing Textual Topic Models](https://www.researchgate.net/publication/254004974)

主题模型的作用之一就是可以从语料库中找到能够代表这个语料库中的词语。那么如何找到这个语料库中的代表性词呢?

Termite 中给出了计算 word distinctiveness 的方式: (distinctiveness 的意思是区分度)

$$
\begin{equation}
\begin{aligned}
distinctiveness(w) &= KL( P(T|w) \parallel P(T) ) \\
                   &= \sum_t p^\prime(t|w) \frac{p^\prime(t|w)}{p(t)}
\end{aligned}
\end{equation}
\tag{6}
$$

简单来说就是计算给定 term 下 topic 的概率分布和 topic 边缘概率分布之间的 KL 散度。需要注意的是, 在 tmtoolkit 中, $p(t)$ 是用假设一计算出来的, $p^\prime(t|w)$ 是用假设二计算出来的。至于为什么要使用不同的假设, 目前不知道原因。

这样的比较没有考虑到词发生的概率, 于是 Termite 中提出了 word saliency (saliency 的含义是重要性), 公式如下:

$$
saliency(w) = p(w) \cdot distinctiveness(w) \tag{7}
$$

这样, 我们就可以根据主题模型找出具有区分度的词语了。需要注意的是, 和 topic-term relevance 一样, word distinctiveness 和 saliency 值的大小不具备很大的参考性, 一般只用于排序, 取 topN 个词。

经过我自己的实验发现, 不同 n_topics 下, 词语的区分度的差别很大, 但是词语的概率差别不大, 结果是 word saliency 值差别也不是很大。

## 4. LDAvis

[pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html) 是目前主流的主题模型可视化工具。

样式可以参考 [样例](../output/vis/lda_15.html) 和 [博客](https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know)。

其右边的是一个横向条形图, 每一个横条表示一个 term, 蓝色横条长度表示的是 ETF (Estimated Term Frequency)。term 是按照 word saliency 进行排序的。考虑到 saliency 数值大小不具备参考性, 因此其没有展示在图上。

其左边的是一个数据图, 用 PCA 算法将 topic_term_dists 中每一个 topic 向量从 `vocab_size` 维度降到二维, 使用 JS 散度作为衡量距离的方式。每一个蓝色气泡表示一个 topic, 其圆心是之前降维后的结果, 圆的面积和 $p(t)$ 之间呈正比。左下角有不同 $p(t)$ 和圆面积对应的比例尺。(目前圆的面积不知道是怎么计算出来的, 需要了解 d3.js, 这个坑就先留在这里了)

蓝色气泡的面积越大, 和其它的气泡重叠部分越少, 就意味着其效果越好, topic 的可解释性越强。

除此之外, 还有选择特定 topic 和 term 的功能。如果选择某一 topic (右击蓝色气泡), 可以看到右边的条形图会发生变化, 此时 term 的排序是按照当前 topic 下的相关性分数进行排序的, lambda 的值可以通过上面的工具栏进行调节。蓝色的横条依然还是这个 term 的 ETF, 红色的横条表示当前 topic 下的 ETF。因此, 红色的横条和 $p(w|t)$ 呈正比, 红色横条和蓝色横条的比值和 lift 呈正比。

除此之外, 还有选择特定 term 的功能。如果选择某一 term (鼠标悬浮在 term 标签上, 不是悬浮在横条上), 可以看到左边的数据图会发生变化, 此时蓝色气泡的面积不是和 $p(t)$ 呈正比, 而是和 $p(t|w)$ 呈正比, 清楚的看到 term 在哪些 topic 中占比较大。

通过这个工具, 可以选择合适的 lambda 值。(其实也不容易, lambda 值的选择要综合考虑所有 topic, 而不能只看一个 topic)

另外, 通过实验发现, 这个工具也不是特别的准, 有些明显重叠的但是主题词之间的区分度还是很高的。

## 5. 总结

想实战这一部分的内容, 建议参考: [文档](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html) 。
