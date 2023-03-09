# -*- coding:utf-8 -*-
# Author: lqxu

from _prepare import *

import os 
import shutil
from typing import * 

import datasets as hf_datasets

import core 

core.init_hf_datasets_acceleration()

text_analyzer = core.TextAnalyzer()

logger = prepare_logging(os.path.join(log_dir, "data_preprocessor.log"))

cache_dir = os.path.join(output_dir, "cache")


if __name__ == "__main__":

    # step0: 删除已存数据
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # step1: 加载数据集
    logger.info("加载原始数据中 ... ")
    # 总共 4 个字段, "id", "title", "text" 和 "label"
    raw_dataset = hf_datasets.Dataset.from_json(
        os.path.join(raw_data_dir, "corpus.jsonl"), keep_in_memory=False, cache_dir=cache_dir,
    )
    logger.info("原始数据加载完成")

    # step2: 打乱数据集中数据的顺序 (使后续的任务时间更加平滑)
    logger.info("开始打乱数据集 ... ")
    raw_dataset = raw_dataset.shuffle(seed=666)
    logger.info("数据集打乱完成")

    # step3: 处理数据集
    def processor(sample: Dict[str, str]) -> Dict[str, str]:
        sample["title"] = sample["title"].strip()
        sample["text"] = sample["text"].strip()

        sample["title_tokens"] = text_analyzer(sample["title"])
        sample["text_tokens"] = text_analyzer(sample["text"])
        return sample 
    
    logger.info("对数据进行分词中 ... ")
    new_dataset = raw_dataset.acc_map(
        processor, batched=False, num_proc=40, new_fingerprint=core.generate_random_fingerprint()
    )
    print()
    new_dataset.save_to_disk(data_dir)
    logger.info("分词完成")

    # step4: 删除缓存文件
    del raw_dataset
    shutil.rmtree(cache_dir)
