# -*- coding:utf-8 -*-
# Author: lqxu

""" 算法类, 主要提供三类算法: 抽取式文本摘要, 关键词抽取 和 新词发现 """

# TODO: 用 cython 加速, 太慢了 !!!

from .basic_page_rank import page_rank

from .basic_text_rank import text_rank_for_summary, text_rank_for_keyword

from .summary import TextRankSummary, JioSummary

from .keywords import TextRankKeywords

from .keyphrases import JioKeyphrases


__all__ = [
    "page_rank", 
    "text_rank_for_summary", "text_rank_for_keyword", 
    "TextRankSummary", "JioSummary", "TextRankKeywords", "JioKeyphrases"
]
