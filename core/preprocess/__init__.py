# -*- coding:utf-8 -*-
# Author: lqxu

"""
数据预处理类

包括: 文本清洗, 分句, 去停用词等功能

本模块主要是将 jionlp 的相关功能单独拿出来, 目的有三个:

1. 加速, 原版的代码使用 Python 原生的 re 模块, 效率太低, 这里我改用 regex, 效率显著提升
2. 代码重构, 原版代码有一些冗余, 重构后方便理解和修改

jionlp 版本: 1.4.30 
"""

import sys 

# suppress jionlp's output 
sys.stdout, stdout = None, sys.stdout 
import jionlp
sys.stdout = stdout

from .clean import TextCleaner
from .analyze import TextAnalyzer
from .split import SentenceSpliter
from .stop_words import StopWordsRemover


__all__ = ["TextCleaner", "TextAnalyzer", "SentenceSpliter", "StopWordsRemover"]
