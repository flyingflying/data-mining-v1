# -*- coding:utf-8 -*-
# Author: lqxu

import sys; sys.path.insert(0, "./")
from time import time

import html 
sys.stdout, stdout = None, sys.stdout; import jionlp; sys.stdout = stdout
import pandas as pd 
from tqdm import tqdm

import core 


def test_basic():

    """
    本代码的测试仅仅针对早期的开发版本, 后续在 clean_text 中添加了内容, 因此一致性测试肯定是过不了的
    """

    df = pd.read_json("./datasets/corpus.jsonl", lines=True)

    diff, jionlp_time, core_time = 0, 0, 0

    clean_text = core.TextCleaner()

    for _, row in tqdm(df.iterrows(), desc="一致性测试", total=df.shape[0]):
        text = row["text"]

        s1 = time()
        result1 = jionlp.clean_text(text)
        s2 = time()
        jionlp_time += s2 - s1

        s1 = time()
        result2 = clean_text(text)
        s2 = time()

        core_time += s2 - s1

        assert result1 == result2

        if result1 != text:
            diff += 1

    print(f"清洗的有效文本数量是: {diff} ")

    print(f"jionlp 运行时间: {round(jionlp_time, 2)} 秒")
    print(f"core 运行时间: {round(core_time, 2)} 秒")


def test_special_case():
    text_cases = [
        "888【（括号去除的状态）】888", 
        "https://pypi.douban.com/simple",
        "html 转义的情况: &nbsp;               |              okok"
    ]

    clean_text = core.TextCleaner()

    for text_case in text_cases:
        print(jionlp.clean_text(text_case))
        print(clean_text(text_case))


if __name__ == "__main__":
    # test_basic()
    test_special_case()
