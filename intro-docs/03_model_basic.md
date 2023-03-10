# 基础模型

[TOC]

## 1. MMR 模型

MMR (Maximal Marginal Relevance) 是信息检索领域中的一个模型, 出自 1998 年的论文 [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) 。属于非常经典的论文了。

原本的公式太长了, 不容易理解, 这里拆开来分析。

我们用 $Q$ 表示用户查询的语句, $R$ 表示候选文本集, $D$ 表示某一文本, $sim_1(D, Q)$ 表示文本和查询语句之间的相似度, $sim_2(D_i, D_j)$ 表示 $D_i$ 文本和 $D_j$ 文本之间的相似度。

$S$ 表示模型已经选择出来的文本候选集, MMR 的计算次数和 top-N 是一致的, 比方说我们需要 top-5 的文本, 那么我们就需要使用 MMR 模型 5 次, 然后将每一次的结果放入结果集 $S$ 中。

MMR 的思想是: 检索出来的文档不仅要和用户的查询相似度高, 还要尽可能地保证结果集的差异性高。

对于文本 $D_i$, 如何保证差异性呢? MMR 给出地方式是:

$$
diversity(D_i) = \max_{D_j \in S} sim_2 (D_i, D_j) \tag{1}
$$

从公式 $(1)$ 可以看出, 对于当前的文本 $D_i$ , 需要和结果集中的 $S$ 的文本两两之间计算相关性, 然后取最高的相关性作为差异性。现在可以给出 MMR 完整地公式了:

$$
MMR = Arg \max_{Di \in R /S} \bigg [ \lambda \cdot sim_1(D_i, Q) - (1 - \lambda) \cdot diversity(D_i)  \bigg ] \tag{2}
$$

$\lambda$ 是调节因子, 取值范围是 $[0, 1]$, 当其值为 1 时, 就是一般的算法, 直接选取相关性最高的 N 个文档。将公式 $(2)$ 代入公式 $(1)$ 即可得到完整的版本。原论文的公式似乎有点问题, 参考其它的资料, 进行了小幅的修正。完整版本:

$$
MMR = Arg \max_{Di \in R /S} \bigg [ \lambda \cdot sim_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S} sim_2 (D_i, D_j)  \bigg ] \tag{3}
$$

MMR 的思想非常好, 后来被大量用于信息检索和推荐算法相关的领域。

## 2. PageRank 算法

[PageRank](http://web.mit.edu/6.033/2004/wwwdocs/papers/page98pagerank.pdf) 算法也是信息检索领域的经典算法, 其作用是对网页进行排序, 但是更主要的是其思想。具体的可以参考 [PageRank:随机游走模型（一）](https://zhuanlan.zhihu.com/p/144042830) 文章, 简单来说, 其原理如下:

+ 将所有网页都当作是结点, 构建图, 如果网页 A 中有网页 B 的链接, 那么就可以通过网页 A 到达网页 B, 那么就有结点 A 指向结点 B 的边
+ 所有的边权重值都是相等的, 无论从网页 A 中有多少网页 B 的链接, 如果某一个网页没有其它网页的链接, 那么就为其添加指向所有网页的边
+ 假设网民在浏览网页时, 有两种行为, 一是从所有网页中随机选择一个进行浏览, 二是从当前页面的链接中随便选择一个进行浏览, 在初始时刻, 只执行前者操作
+ 反复执行上述的操作, 我们可以得到每一个时刻网民访问每一个网页的概率, 根据马尔可夫链的收敛性, 最终访问每一个网页的概率会趋于一个固定值
+ 我们将上一步得到的固定值作为网页的分数, 进行排序, 得到网页的排名

需要说明的是:

+ 前三步的过程也可以用马尔科夫链来描述, 关于第四步的收敛性, 在马尔可夫链的知识中有介绍, 需要有 **随机过程** 的相关知识, 又是一个大坑
+ 从上述过程中可以看出, 对于网页 A 来说, 指向其的网页越多, 网民浏览的概率就会越大, 最终的分数也就越大, 这就是 page rank 的核心思想之一: 网页的外链 (指向其的链接) 越多, 这个网页的质量就越好。
+ 这种解决问题的方式是可以扩展的, 对于新的问题, 我们需要自己定义 **结点** 的含义, 定义 **边** 的连接方式以及权重, 就可以获取到每一个结点的分数, 拓展的算法有: ItemRank, TextRank 等等
+ PageRank 的问题是可以人为作假, 即和别人商量好, 让他人来引用自己的网页, 以增加自己网页的外链, 因此后来其他人又提出了改进的算法, 比如说 TrustRank 等等
+ 可以看出, PageRank 可以作为一种通用的解决问题的思路, 这种思路可以起一个统一的名字: Unified Score Propagation Model, 统一分值传播模型
+ PageRank 算法属于 link analysis 中的一个无监督的算法, 想了解更多知识可以搜索 link analysis

用公式表示如下:

$$
PR(n_i) = (1 - \alpha) \times \frac{1}{N} + \alpha \times \sum_{n_j \in IN(n_i)} \frac{PR(n_j)}{|OUT(n_j)|} \tag{4}
$$

其中 $n_i$ 和 $n_j$ 表示图中的结点, $PR(n_i)$ 表示 $n_i$ 结点的 PageRank 值, $IN(n_i)$ 表示所有指向 $n_i$ 的结点集合, $OUT(n_i)$ 表示 $n_i$ 指向的结点集合, $|OUT(n_j)|$ 表示 $OUT(n_i)$ 集合中元素的数量, $N$ 表示总结点数。$\alpha$ 表示从当前结点选下一个结点的概率, 是一个调节因子, 维基百科上说一般取值是 0.85 。

## 3. TF-IDF 算法

在 NLP 领域, 最初的文本向量化方式就是 TF (Term Frequency), 特征就是每一个词, 特征的值就是这个词在文本中出现的次数。但是这种只考虑词频的方式并不是很好, 一种改进方式就是 TF-IDF 。

什么样的词语对于一篇文章来说很重要呢? 我们一般认为, 如果某一个词在文档中出现的次数越多, 同时在语料库中出现的次数越少, 那么这个词语就越重要。

比方说, 如果某一语料库是关于 "犬科动物" 的资料库, 这个语料库中几乎所有的文档都包含 "小狗" 这个词, 那么这个词语就不适合当作这些文档的特征 (没有区分度), 应该给予其较低的特征值;

相反, 如果这个语料库是关于 "动物" 的资料库, "狗" 只在 5% 的文档中出现过, 那么其就适合当作这些文档的特征 (有区分度), 应该给予其更高的特征值。

我们定义: $t$ 表示 token/term/word (词语), $d$ 表示 document (文档), $N$ 表示语料库中总的文档数

$tf(t, d)$ 表示 token 在 document 中的 **词语频率(term frequency)**, 即 token 在 document 中出现的数量除以 document 的总词数。

$df(t)$ 表示 token 的 **文档频率(document frequency)**, 即其在多少篇文档中包含 token 。

需要注意的是: TF 是需要归一化的 (即需要除以 document 的总词数), DF 是不需要归一化的 (即不需要除以总的 document 数量)

$idf(t)$ 表示 token 的 **逆文档频率(inverse document frequency)**, 最基本的公式是: $idf(t) = \log \frac{N}{df(t)}$ 。

完整的公式如下: $tfidf(t, d) = td(t, d) * idf(t)$ 。其中 TF 和 IDF 的计算方式有很多的变种, 这里就先不介绍了。在 gensim 中, `tfidfmodel` 中有 `wlocal` 和 `wglobal` 参数, 可以改变计算方式。

需要注意的是: TF-IDF 是 TF 向量化的改进版本, 两者的向量稀疏性是一致的, 都属于 BOW 向量化方式。

## 4. BM25 算法

在信息检索领域 BM25 算法可以说是非常出名的, 全名是 Okapi BM25, 其中 BM 表示 Best Matching 。其是用于计算 query 和 document 相关性分数的, 很多论文中会用其作为信息检索的 baseline 算法, ElasticSearch 中也是用其作为默认的检索算法的。(这里挖一个坑, 有空用 ElasticSearch + DuReader Retrieval 数据集实现这个 baseline, 从而更加深入的理解 ElasticSearch 的用法。)

BM25 算法的计算过程可以说是 TF-IDF 算法的改进, 但是其用途和 TF-IDF 有很大区别。网上很多资料都是直接给公式, 这里我们将公式拆开, 一步一步来看。

对于信息检索任务来说, 我们会有一个大型的资料库, 然后根据用户的查询语句 (query), 从资料库中找出相关性最高的文章返回给用户。

首先, 我们定义 $Q$ 为 query, 即用户的查询语句, 其一般很短, 但是我们依然要对其进行分词, 分词后的每一个 token 记为 $q_i$ 。

我们定义 $D$ 为 document, 是资料库中的某一个文档, 然后用 $|D|$ 表示这篇文档中的总 token 数, $avgdl$ 表示资料库中所有文档的平均长度。

BM25 中是这样定义 $Q$ 和 $D$ 之间的分数: $BM25(D, Q) = \sum_i score(D, q_i)$ , 即 query 中的每一个词和 document 的相关性分数之和。

这里, token 和 document 的计算方式是从 TF-IDF 中借鉴而来, 但是 TF-IDF 中的 token 是 document 中的 token, 而这里的 token 是 query 中的 token 。具体的计算方式如下:

$$
score(q_i, D) = weight(q_i) \cdot relation(q_i, D) \tag{5}
$$

我们设 $N$ 为资料库中的总文档数, $n(q_i)$ 为资料库中包含 $q_i$ 这个 token 的总文档数, 则:

$$
weight(q_i) = idf(q_i) = \log \frac{N - n(q_i)}{n(q_i)}
$$

和 TF-IDF 中的 IDF 相比, 分子从 $N$ 变成了 $N-q(n_i)$, 简单分析可知, 当 $n(q_i) = \frac12 N$ 时, $idf(q_i)$ 就等于 0 了, 如果 idf 值算出来小于 0, 那么可以直接认为其等于 0 (分段函数)。也就是说, 如果 $q_i$ 在一半以上的文档中出现, 那么我们认为其没有检索价值, 直接使其权重值为 0。这种做法其实就是去高频词。

实际在使用时, 分子和分母会加 0.5, 使得计算更加平滑, 这里就不讨论这个问题了。当然, 如果你有更好地 IDF 计算方式, 也是可以替换地。完整版的公式如下:

$$
weight(q_i) = idf(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \tag{6}
$$

我们设 $f(q_i, D)$ 表示 $q_i$ 在 $D$ 文档中出现的次数, $tf(q_i, D)$ 表示 $q_i$ 在 $D$ 中的文档频率, 则:

$$
tf(q_i, D) = \frac{f(q_i, D)}{|D|} \cdot avgdl = \frac{f(q_i, D)}{\frac{|D|}{avgdl}}
$$

和 TF-IDF 中不同的是, 在这里, TF 还乘以了 $avgdl$, 不再是归一化了, 相当于是标准化。但是, BM25 不仅仅满足于此, 越长的文档包含的信息一般也是越多的, 因此会引入一个调节因子 $b$, 公式变为:

$$
tf(q_i, D) = \frac{f(q_i, D)}{(1-b) + b \cdot \frac{|D|}{avgdl}} \tag{7}
$$

$b$ 的取值在 `[0, 1]` 之间, 如果取 0, 则完全不考虑 长度标准化, 如果取 1, 则完全执行长度标准化。根据经验, $b$ 的取值一般是 0.75 。

TF 不进行归一化的话, 那么如果某一个词在文档中反复出现怎么办呢? 没有关系, BM25 中会给 TF 设置一个饱和度 (saturation), 方式如下:

$$
relation(q_i, D) = \frac{(k_1 + 1) \cdot tf(q_i, D)}{k_1 + tf(q_i, D)} \tag{8}
$$

可以通过画图, 也可以用洛必达法则知道:

$$
\lim \limits_{x \to +\infty} \frac{(k_1 + 1) \cdot x}{k_1 + x} = k_1+ 1
$$

也就是说, $relation(q_i, D)$ 的最大值是 $k_1 + 1$, 因此也被称为 **饱和度**, 根据经验, $k_1$ 的取值范围是 $[1.2, 2.0]$ 。

将公式 $(6)$, $(7)$ 和 $(8)$ 代入公式 $(5)$ 中, 即可得到完整的计算 score 的公式, 由于太麻烦了, 这里就不叙述了。可以看出, BM25 的设计思路比 TF-IDF 要复杂地很多。

如果 query 不是用户地输入, 而是一个很长的文档呢? 那么公式 $(5)$ 变成: $score(q_i, D) = weight(q_i) \cdot relation(q_i, D) \cdot relation(q_i, Q)$

其实 TF-IDF 和 BM25 算法还可以通过概率模型的方式来解释, 尤其是 idf, 其实就是概率的倒数求对数, 这不就是信息论中的信息量吗？这里先不介绍了, 后面再介绍。更多相关资料建议参考维基百科的 tf–idf 和 Okapi BM25 词条, [博客](https://www.cnblogs.com/geeks-reign/p/Okapi_BM25.html) 和 [文档](http://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) 。

## 5. TextRank 算法

[TextRank](https://aclanthology.org/W04-3252.pdf) 是 PageRank 的应用, 主要用于关键词抽取和关键句抽取, 我们分别来看。

对于关键词抽取, 先将 document 分词, 然后将每一个词语作为图中的 **结点**。设置一个窗口, 在窗口内的词建立 **边** 的关系。比方说, 如果窗口是 2, 那么对于任意的词 A, 就要和其左边的两个词相连, 并且和其右边的两个词相连, 这样图就构建完成了, 剩下的按照 PageRank 的方式进行, 就能获得每一个词语的分数, 分数高的就是关键词。

对于关键句抽取, 先将 document 分句, 然后将每一个句子作为图中的 **结点**。考虑到一篇文档中的句子数是远远小于词语数的, 那么就认为所有的句子都是相连的, 然后用 BM25 算法计算每两个句子之间的分数, 作为权重值, 这样图就构建完成了。需要注意的是, 在计算 BM25 时, 语料库就是文档中的每一个句子, 这也暴露出 TextRank 的问题, 如果句子较少, 那么 IDF 的计算就会很不科学, 并不能真实地反映词语的权重。

另外, TextRank 论文中用的 PageRank 公式和公式 $(4)$ 有细微的差别, 那就是 $(1-\alpha)$ 项没有除以 $N$, 这似乎是 PageRank 作者的失误导致的, 具体可以参考 PageRank 的维基百科词条。TextRank 用了错误的公式, 建议我们在实现时, 和论文保持一致 (也就是保持错误, 或者修正后自行探索合理性) 。

TextRank 只是 PageRank 最基本的改变形式, 之后还有 [Salience Rank](https://aclanthology.org/P17-2084.pdf) 和 [Topical PageRank](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf) 。他们构建图的方式和 TextRank 关键词抽取的构建方式是差不多的, 区别在于改变了随机选择结点的概率, 不再假设他们是均匀分布了, 而是假设他们的概率分布符合某一种词权重的概率分布, 比方说 word distinctiveness。
