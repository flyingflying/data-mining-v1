# -*- coding:utf-8 -*-
# Author: lqxu

from _prepare import *

import os 
import json 

from gensim.models import TfidfModel
from gensim.corpora import Dictionary  # , MmCorpus

logger = prepare_logging(os.path.join(log_dir, "train_idf.log"))


if __name__ == "__main__":

    lda_model_dir = os.path.join(model_dir, f"lda_model")

    dict_path = os.path.join(lda_model_dir, "dict.txt")
    # corpus_path = os.path.join(lda_model_dir, "corpus.txt")

    logger.info("开始加载数据 ... ")
    dictionary = Dictionary.load_from_text(dict_path)
    dictionary[0]  # 不这样做, 没有 id2token 字典
    # corpus = MmCorpus(corpus_path)
    logger.info("数据加载完成")

    logger.info("开始训练模型 ... ")
    model = TfidfModel(dictionary=dictionary)
    logger.info("训练模型完成")

    logger.info("开始保存数据 ... ")
    with open("./weights/specified_idf.json", "w", encoding="utf-8") as writer:
        json.dump(
            {dictionary.id2token[word_id]: word_idf for word_id, word_idf in model.idfs.items()}, 
            writer, 
            ensure_ascii=False, 
            indent=0
        )
    logger.info("数据保存完成")
    