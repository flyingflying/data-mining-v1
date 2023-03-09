# -*- coding:utf-8 -*-
# Author: lqxu

import sys; sys.path.insert(0, "./")
from time import time

sys.stdout, stdout = None, sys.stdout; import jionlp; sys.stdout = stdout

import pkuseg 
import pandas as pd 
from tqdm import tqdm

import core 


def test_performance():

    df = pd.read_json("./datasets/mini_corpus.jsonl", lines=True)
    tokenizer = pkuseg.pkuseg()
    jionlp_time, core_time = 0., 0.

    remove_stopwords = core.StopWordsRemover()

    for _, row in tqdm(df.iterrows(), desc="性能测试", total=df.shape[0]):
        text = core.clean_text(row["text"])
        tokens = tokenizer.cut(text)

        kwargs = {
            "remove_time": True, "remove_location": True, "remove_number": True, 
            "remove_non_chinese": True, "save_negative_words": False
        }

        jionlp.remove_stopwords([""], **kwargs)  # 调用 _prepare 函数

        s1 = time()
        jionlp.remove_stopwords(tokens, **kwargs)
        s2 = time()
        jionlp_time += s2 - s1

        s1 = time()
        remove_stopwords(tokens)
        s2 = time()

        core_time += s2 - s1

    print(f"jionlp 运行时间: {round(jionlp_time, 2)} 秒")
    print(f"core 运行时间: {round(core_time, 2)} 秒")


def debug():

    # jionlp 中的匹配有一些 bug, 比方说, 上周末 属于时间 token, 但是 jionlp 不会识别出来
    tokens = ["上周末"]

    remove_stopwords = core.StopWordsRemover()

    print(remove_stopwords(tokens))
    print(jionlp.remove_stopwords(tokens, remove_time=True, remove_location=True, remove_number=True, remove_non_chinese=True, save_negative_words=False))


if __name__ == "__main__":
    # test_performance()

    debug()
