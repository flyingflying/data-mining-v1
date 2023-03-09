# -*- coding:utf-8 -*-
# Author: lqxu

from _prepare import *

import os 
import shutil

from datasets import Dataset as HFDataset

from gensim.corpora import Dictionary, MmCorpus  #, TextCorpus
from gensim.models.callbacks import DiffMetric, CoherenceMetric, PerplexityMetric, ConvergenceMetric

from core import LDA


def tokens_generator(hf_dataset: HFDataset):
    """ 返回文本的 tokens 形式, 此方法和 TextCorpus.get_texts 方法是一致的 """    
    for sample in hf_dataset:
        yield sample["title_tokens"] + sample["text_tokens"]


def bow_generator(hf_dataset: HFDataset, dictionary: Dictionary):
    """ 返回文本的 bow 形式, 此方法和 TextCorpus.__iter__ 方法是一致的 """
    for tokens in tokens_generator(hf_dataset):
        yield dictionary.doc2bow(tokens)

    
if __name__ == "__main__":

    """
    对于大规模数据处理来说, 迭代器是最基本的操作。
    
    在 gensim 中, 提供了两种类型的 corpus 对象:
    1. TextCorpus
        a. 针对文本的数据集, 包括数据预处理, 生成 dictionary 对象, 迭代 toekns 形式的文本, 迭代 bow 形式的文本等等功能。
        b. 其子类有: TextDirectoryCorpus 和 WikiCorpus, 可以处理特定格式的数据集
        c. 对中文 NLP 很不友好。中文 NLP 往往需要分词, 而分词运行的速度十分缓慢, 因此需要将分词结果保存起来, 但是其没有提供相关的功能。
        d. 本项目采用的是使用 HuggingFace Datasets 进行数据的存储和并行化操作, 具体见 `data_preprocessor.py`, 这里单独使用两个生成器解决问题。
    2. IndexedCorpus
        a. 将 bow 形式的文本存储在硬盘中, 使用 tell 构建索引, seek 进行查询, readline 读取文本, 可以进行迭代和索引 (random indexing)。
        b. 存储的格式多种多样, gensim 中提供了 6 种格式, 分别对应 6 个子类: MmCorpus, BleiCorpus, SvmLightCorpus, LowCorpus, UciCorpus, MalletCorpus
        c. 和 TextCorpus 不同的是, IndexedCorpus 需要反复迭代, 因此不能是 Iterator 对象, 必须是 Iterable 对象
        d. 这里和 gensim 的教程保持一致, 使用 MmCorpus (可能是存储最小的方案了)
        e. 需要注意的是, 如果你的数据集非常大, 需要构建多个 IndexedCorpus, 可以使用 core.SimpleNestedSequence 将多个数据集封装成一个数据集
    
    主题模型往往也会用于数据挖掘, 如果文本数量太多, doc_topic_dists 很可能无法存储在内存中, 此时建议使用 numpy.memmap 方法进行存储。
    """

    # ## global settings 
    use_cache, debug_mode, use_visdom = False, False, False  # use_visdom 只有在 debug_mode 是 True 时才有用

    # ## step0: 基础设置
    prefix = "debug_" if debug_mode else ""
    lda_model_dir = os.path.join(model_dir, f"{prefix}lda_model")
    logger = prepare_logging(os.path.join(log_dir, f"{prefix}train_lda.log"))

    # dict 和 corpus 都是以 text 形式存储的
    dict_path = os.path.join(lda_model_dir, "dict.txt")
    corpus_path = os.path.join(lda_model_dir, "corpus.txt")
    
    model_path = os.path.join(lda_model_dir, "model.{}.bin")
    mining_path = os.path.join(lda_model_dir, "mining.{}.bin")

    hf_dataset = HFDataset.load_from_disk(data_dir)
    if debug_mode:
        hf_dataset = [hf_dataset[idx] for idx in range(150)]
    
    if not use_cache and os.path.exists(lda_model_dir):
        shutil.rmtree(lda_model_dir)
    os.makedirs(lda_model_dir, exist_ok=True)

    # ## step1: 初始化字典
    if not use_cache or not os.path.exists(dict_path):
        logger.warning("初始化字典 ... ")
        
        dictionary = Dictionary()
        # 经过测试, 这里可以使用 generator/iterator 对象, 不需要使用 iterable 对象
        dictionary.add_documents(tokens_generator(hf_dataset), prune_at=int(2e6))
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=int(1e5))  # 结果只会保存十万个词, 注意 1e5 是 float 类型
        dictionary.save_as_text(dict_path)
        
        logger.warning("字典初始化完成 !!! ")

    else:
        dictionary = Dictionary.load_from_text(dict_path)
    
    # ## step2: 初始化数据集
    if not use_cache or not os.path.exists(corpus_path):
        logger.warning("初始化 bow 数据集 ... ")
        MmCorpus.serialize(corpus_path, bow_generator(hf_dataset, dictionary))
        logger.warning("bow 数据集初始化完成 !!! ")
    
    # 如果有多个 corpus, 或者说对 corpus 进行了分片, 可以使用 core.SimpleNestedSequence 类进行封装
    corpus = MmCorpus(corpus_path)
    
    # ## step3: 训练主题模型
    # candidate_num_topics = [15, 50, 100, 200]
    candidate_num_topics = [15, 50]  # 太大, 没有必要 (num_topics 对 word distinctiveness 影响比较大, 对 word saliency 影响比较小)

    for num_topics in candidate_num_topics:
        
        """
        callback: 控制每一个 epoch 开始和结束时运行的代码, gensim 中提供了四个输出 metrics 的方法
        
        可以在控制台输出, 也可以用 visdom 进行可视化。
        
        如果用控制台输出, 可以用指令: `cat output/log/debug_train_lda.log | grep Epoch` 查看结果
        
        如果用 visdom 进行可视化, 需要事先进行以下操作:
        1. 安装 visdom 工具, 即 `pip install visdom`
        2. 运行 visdom, 必须通过本机的 8097 端口访问, 即 `visdom --hostname 0.0.0.0`
        
        用 visdom 进行输出需要注意:        
        1. 每一个 epoch/pass 结尾会往面板上输出数据, 如果 passes=1, 由于点不清楚, 无法知道数值的大小, 因此建议用 shell 输出
        2. perplexity, coherence, convergence 都是折线图, diff 是热力图
        
        四种方式指标的含义:
        1. perplexity: sklearn 中的 score 方法返回值, 正常应该是负数, 值越大越好, gensim 将其取反, 变成了正数, 越小越好
        2. coherence: 一致性分数, 负数, 越大越好
        3. diff: 当前 epoch 和上一个 epoch 主题之间的差异, 可以判断本次迭代过程中哪些主题的参数进行了更新 (都不更新表示模型训练好了)
        4. convergence: 将每一个主题下面所有的 diff 值加起来, 越小表示模型更新的越少
        """
        
        if debug_mode and use_visdom:
        
            callbacks = [
                PerplexityMetric(corpus=corpus, logger="visdom", title=f"perplexity with {num_topics} topics"),
                CoherenceMetric(corpus=corpus, logger="visdom", coherence="u_mass", title=f"coherence with {num_topics} topics"),
                DiffMetric(logger="visdom", title=f"diff with {num_topics} topics"),
                ConvergenceMetric(logger="visdom", title=f"convergence with {num_topics} topics")
            ]
            
            epoches = 5  # epoch 为 1 看不出效果
        
        else:
        
            callbacks = [
                PerplexityMetric(corpus=corpus, logger="shell"),  # 这一步的计算是很缓慢的
                CoherenceMetric(corpus=corpus, logger="shell", coherence="u_mass", topn=20),
                ConvergenceMetric(logger="shell",)
            ]
            
            epoches = 1
        
        model = LDA(
            num_topics=num_topics, id2word=dictionary, chunksize=4000,
            eval_every=10, iterations=50, passes=epoches, callbacks=callbacks
        )

        logger.warning(f"开始训练 {model} ... ")

        model.fit(corpus)
        # metric = model.score(corpus)
        model.save(model_path.format(num_topics))
        model.save_mining_data(mining_path.format(num_topics), corpus)

        # logger.warning(f"{model} 训练完成, topic coherence 值是 {metric:.2f}")
        logger.warning(f"{model} 训练完成!!!")

        # vim 中: ctrl+b 表示 page up, ctrl + f 表示 page down, 可以用其快速查看日志 
