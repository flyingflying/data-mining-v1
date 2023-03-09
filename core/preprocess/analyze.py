# -*- coding:utf-8 -*-
# Author: lqxu

"""
文本分析, 将文本清洗, 分词和去停用词组合在一起  
"""

from typing import *

from pkuseg import pkuseg

from .clean import TextCleaner
from .stop_words import StopWordsRemover


class TextAnalyzer:

    """ 默认的文本分析器, 包括清洗文本, 分词 和 去停用词 """

    def __init__(self):
        self.clean_text = TextCleaner()
        self.segmenter = pkuseg()
        self.remove_stopwords = StopWordsRemover()

    def __call__(self, text: str) -> List[str]:
        text = self.clean_text(text)
        tokens = self.segmenter.cut(text)

        # 根据短语的长度预先筛选一波
        tokens = [token for token in tokens if 2 <= len(token) <= 6]

        tokens = self.remove_stopwords(tokens)
        return tokens
