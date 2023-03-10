
# 文本挖掘示例项目

本项目是中文文本数据挖掘的示例项目, 包括以下的内容:

## 中文文本的数据预处理

这一部分主要借鉴自 [JioNLP](https://github.com/dongrixinyu/JioNLP) 。感谢他们的开源。

1. [数据文本清洗](core/preprocess/clean.py)
2. [文本标点分句](core/preprocess/split.py)
3. 文本分词
4. [去停用词](core/preprocess/stop_words.py)

文本分词使用的是 `pkuseg` 或者 `jiojio` 库。不直接使用 `jionlp` 的原因主要有:

1. 原版使用 Python 原生的 `re` 库, 效率很低, 这里改用 `regex` 库, 匹配效率大大提高
2. 在实际的数据挖掘过程中, 文本清洗和分句对最终的效果影响非常大, 这一点后续说明

未来计划:

+ [ ] 优化 HTML 的文本清洗的方式, 按照 inline 和 block 标签的方式分句, 可以参考 [HTMLParser](https://github.com/mavlyutovrus/htmlparser)

## 大批量数据处理

本项目使用的是清华 [THUCTC](http://thuctc.thunlp.org/) 工具中的数据集, 其中包含 14 类新闻约 83 万的文本。

对这些文本的分词需要使用分词器, 不同于英文, 其分词速度非常缓慢, 需要将分词结果保存起来。但是保存分词结果是非常占内存的一件事情。本项目使用 [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) 库保存中间结果, 实现低内存的中文分词。

需要注意的是, 原版 `datasets` 使用的是第三方 [multiprocess](https://multiprocess.readthedocs.io/en/latest/index.html) 库启动多进程, 使用 [dill](https://dill.readthedocs.io/en/latest/index.html) 库序列化对象。`dill` 库在配合 `jionlp` 和 `pkuseg` 时运行速度非常缓慢, 本项目将其换成了 Python 原生的 [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) 库, 采用 [pickle](https://docs.python.org/3/library/pickle.html) 库序列化对象, 运行速度显著提升。

## 主题模型

最经典的主题模型就是 LDA 了。本项目使用 LDA 主题模型, 主要原因有以下几点:

1. LDA 模型对于 **超长文本** 的效果很好。如果语料库中大部分的文本长度都超过 2000 字, 这时候不建议选择基于神经网络的主题模型
   + LDA 模型是对 TF 文本向量的降维, 其不包含语义信息。但是对超长文本计算 "语义" 的成本很高。
   + 短文本用 TF 向量的效果往往不好, 但是对于超长文本来说, TF 向量很可能已经包含了我们所要挖掘的信息了
2. 和主题模型相关的很多理论都是基于 LDA 模型, 其它模型能否使用有待商榷

需要注意的是, 使用 LDA 主题需要非常精细的文本清洗, 所有和主题不相关的句子都应该剔除掉, 比方说: “打开微信扫一扫, 分享给好友” 这句话如果在 20% 以上的文本中出现了, 就要考虑剔除, 不然会很大程度上影响主题模型的训练, 部分主题的权重词可能会出现 `微信` 这样的词语。

这里使用地是 `gensim` 库内置的 `MmCorpus` 作为大规模语料库低内存训练的接口。

如果数据量很大的话, 数据一般是存在硬盘中的, 用 `seek` 和 `tell` 方法快速读取数据。`MmCorpus` 也是这么实现的。

未来计划:

+ [ ] 只取标题, 用 ESimCSE 训练句向量模型, 用 BERTopic 测试短文本中主题模型的效果

## 数据挖掘

主要使用 [tmtoolkit](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html) 中的内容去寻找标签, 挖掘信息。包括:

+ word distinctiveness, word saliency 的计算
+ [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/) 的可视化
  
## 抽取式摘要

复现了 `HanLP` v1 中的 TextRank 算法和 `Jionlp` 中的摘要抽取算法。主要原因是方便定制化。

在抽取式摘要中, 句子的颗粒度是非常重要的, 有时需要细粒度的标点分句 (逗号也要分), 有时需要粗粒度的分句 (逗号部分, 按句号分), 有时则需要按照换行符分。分句的颗粒度越大, 摘要的可读性越高, 相应的, 摘要包含的信息就越集中 (越少)。

未来计划:

+ [ ] 研究 LexRank 算法: [论文](https://arxiv.org/pdf/1109.2128.pdf), [博客](https://iq.opengenus.org/lexrank-text-summarization/), [代码](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/text-summarization)

## 关键词抽取

关键词抽取是从已经分过的词语中选择出重要的, 能代表文章的词语。这里复现了 `HanLP` v1 中基于 TextRank 的关键词抽取。

## 关键短语抽取

关键短语抽取不仅仅要选择重要的 **词**, 还要选择重要的 **词的组合** 。这里复现了 jionlp 中的关键短语抽取, 主要流程是:

1. 使用滑窗的方式生成候选短语
2. 根据词性对候选短语进行筛选
3. 给每一个候选短语打分, 并排序
4. 根据短语的相似度 (相同 token 的占比) 对候选短语进行筛选, 保留下分数高的短语
5. 按照用户需求返回 topn 短语

打分的方式有很多, 我们可以使用 TextRank 算法, 也可以使用其它算法, jionlp 中的打分综合考虑了: token 在 document 中的 TF-IDF 分数, token 的数量, phrase 首尾 token 的词性组合, token 的 word distinctiveness 等等。

当然, 也可以使用 [Salience Rank](https://aclanthology.org/P17-2084.pdf) 和 [Topical PageRank](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf) 。他们主要是改变随机选择结点的概率。

很可惜的是, `jionlp` 的作者并没有说明 phrase 首尾 token 的词性组合是怎么计算出来的, 也没有说明 token 数量的权重是怎么得到的, 仅仅说是可调节参数, 这一点很可惜, 有时间的话需要进一步地研究。

## 运行方式

本项目并不是一个工具包, 而是一个项目。除了 `core.utils` 中的代码, 其它代码希望你都能一步一步的看懂, 尤其是 `core.preprocess` 和 `core.algorithms` 中的代码。

运行方式如下:

**数据预处理:**

```shell
python examples/data_preprocessor.py
```

**LDA 模型训练:**

```shell
python examples/train_lda_model.py
```

**TF-IDF 模型训练:**

```shell
python examples/train_idf_weight.py
```

**数据挖掘:**

[代码](examples/data_mining.py) 是在 vscode 工具中的 interactive windows 中运行的, 至少需要安装 `ipython`, 建议直接安装 `jupyter`。

**初始化权重**:

```shell
python weights/init_script.py
```

**摘要, 短语抽取**:

```shell
python examples/summary.py

python examples/keywords_phrase.py
```
