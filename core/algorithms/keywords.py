# -*- coding:utf-8 -*-
# Author: lqxu

from typing import * 

from pkuseg import pkuseg

from core.preprocess import TextCleaner, StopWordsRemover
from core.algorithms import text_rank_for_keyword


class TextRankKeywords:
    """ 这里的方式是从文章中选取关键的词语, 并不是抽取关键词 """
    def __init__(self) -> None:
        self.clean_text = TextCleaner()
        self.segment = pkuseg().cut 
        self.remove_stopwords = StopWordsRemover()
    
    def __call__(self, text: str, topn: int = 10) -> List[str]:
        text = self.clean_text(text)
        tokens = self.segment(text)
        tokens = self.remove_stopwords(tokens)
        
        return text_rank_for_keyword(tokens=tokens)[:topn]
