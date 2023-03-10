
# 主题模型

[TOC]

## 简介

网上有很多关于主题模型介绍的教程, 很多时候都会将其和 LDA 画上等号, 然后开始介绍 LDA 的计算方式, 这样是不合理的。一方面, 主题模型不能和 LDA 模型画等号, 另一方面, 大部分人也不可能去研究并优化 LDA 模型, 而是需要把他当作工具, 进行数据分析和应用开发。本文将从更广阔的角度来介绍主题模型。

什么是主题模型, 有写教程喜欢举例子, 比方说 [PLSA详解](https://www.cnblogs.com/daretobe/p/4677948.html) 。喜欢看例子的可以参考这篇博客的第一部分。我打算从更加数学化的方式来介绍主题模型。

在 [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) 中, 主题模型算法被归为 **降维算法**, 在 [spark](https://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda) 中, 主题模型算法被归为 **聚类算法**。我们分别从这两方面开看主题模型。

## 降维算法

我们可以将主题模型理解为一种 TF 形式的文档向量降维方式。

对于传统的机器学习来说, 特征工程是非常重要的。在自然语言处理领域, 一般会选用 "词语" 作为特征, 词语在文档中出现的次数作为特征的值。这种方式我们一般称为 BoW 向量, 本文称为 TF 向量。这种向量化最大的问题是 **稀疏性**, 其维度是 `vocab_size`。

主题模型的主要功能是将文本的 TF 向量从 `vocab_size` 降维到 `n_topics` 。不同于后来的 word2vec 算法, 这里的文本向量每一个维度是有含义的, 即 "主题"。文本向量特征的值就是主题的权重值。

什么是 "主题" 呢? 主题在早期也被称为是 "潜在语义", 其数学形式和 idf 是一致的, 一个 `vocab_size` 维度的向量。主题向量的每一个特征表示一个词语, 特征的值就是词语的权重。

我们可以这样理解:

+ TF 文本向量化: 文本向量的特征是词, 我们可以认为: 词 → 文本
+ 主题模型文本向量化: 文本向量的特征是主题, 主题向量的特征是词, 我们可以认为是: 词 → 主题 → 文本

对于概率主题模型来说, 我们赋予特征的值更加具体的含义: 概率。即:

+ 文本向量是 "主题" 的概率分布
+ 主题向量是 "词表" 的概率分布

## 聚类算法

仔细阅读上面一部分会发现, 主题向量和 TF 文本向量的维度是一致的, 都是 `vocab_size`。我们可以认为主题向量和 TF 文本向量在同一个向量空间中。

此时, 我们可以认为主题向量是 **质心**, 降维后的文本向量表示的是 TF 文本向量和主题向量之间的相关性分数。这样就可以和聚类算法关联上了。

按照聚类算法的理论, 文本属于相关性分数最高的那一个主题。

## 主题模型家族

上面是从广义的机器学习角度来解释什么是主题模型。主题模型在发展过程中, 主要有两种方式, 一种是 **矩阵分解** 的方式, 一种是 **概率主题模型** 。概率主题模型因为 LDA 模型而火起来, 后来的模型也基本都是概率主题模型, 因此很多教程会将主题模型和概率主题模型画上等号。

主题模型的家族大致上如下:

+ 矩阵分解
  + LSI(LSA, Truncated SVD)
  + NMF
  + Random Indexing
+ 概率主题模型
  + 经典模型: pLSA, LDA, HDP
  + 短文本建模: BTM, LDA-U, TwitterLDA
  + 神经网络结合: CTM, ETM, NeuralLDA, ProdLda, Top2Vec, BERTopic
  + 其它: MG-LDA, AuthorLDA, dynamic topic models, guided topic models, LDA2VEC

## 概率主题模型

这里先挖一个坑, 主要结合 [LDA数学八卦](LDA数学八卦.pdf) 和 [LDA漫游指南](LDA漫游指南.pdf) 来介绍。

## 主题模型相关工具

+ Java 社区: Mallet, TMT, Termite
+ Python 社区: sklearn, **gensim**, **tmtoolkit**, octis, pyLDAvis
+ Spark (MLlib), MatLab, R 语言中也有相关的工具
+ 可视化工具: **pyLDAvis**
