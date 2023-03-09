# -*- coding:utf-8 -*-
# Author: lqxu

"""
将 jionlp 中的 clean_text 方法抽出来, 方便修改和运行加速

jionlp 版本: 1.4.30
"""

import html 
import regex as re 
# from bs4 import BeautifulSoup
from jionlp.rule.rule_pattern import *


class TextCleaner:

    def __init__(self) -> None:
        
        self._html_tag_pattern = re.compile(HTML_TAG_PATTERN)

        self._exception_pattern = re.compile(EXCEPTION_PATTERN)

        self._redundant_pattern = re.compile(  # (?<=) 后向肯定 (以此开头)
            "|".join(
                '(?<={char}){char}+'.format(char=re.escape(char)) for char in REDUNDANT_PATTERN
            )
        )

        self._parentheses_pattern = re.compile(
            "|".join(
                f"{re.escape(left)}[^{re.escape(left)}{re.escape(right)}]*{re.escape(right)}" 
                for left, right in zip(PARENTHESES_PATTERN[0::2], PARENTHESES_PATTERN[1::2])
            )
        )

        self._url_pattern = re.compile(URL_PATTERN)

        self._email_pattern = re.compile(EMAIL_PATTERN)

        self._full_angle_pattern = str.maketrans(FULL_ANGLE_ALPHABET, HALF_ANGLE_ALPHABET)

        self._cell_phone_pattern = re.compile(CELL_PHONE_PATTERN)

        self._landline_phone_pattern = re.compile(LANDLINE_PHONE_PATTERN)


    def __call__(self, text: str) -> str:

        """ 清洗文本 """

        # step1: 去除 HTML 标签, 转义 HTML 字符 (不知道为什么 jionlp 中没有对 html 字符进行转义)
        # text = BeautifulSoup(text, "lxml").get_text()
        text = self._html_tag_pattern.sub("", text)
        text = html.unescape(text)  # reference: https://m.imooc.com/wenda/detail/600769

        # step2: 去除异常字符, 包括不可打印字符, 数学公式符号, 其它语言符号等等
        text = self._exception_pattern.sub(" ", text)

        # step3: 全角转半角
        text = text.translate(self._full_angle_pattern)

        # step4: 去除冗余字符, 比方说文本有时会出现多个换行符, 我们将其转换成单个换行符
        # TODO: "\n \n" ==> "\n"
        text = self._redundant_pattern.sub("", text)

        # step5: 去除括号, 以及括号内的文字
        length = len(text)
        while True:
            text = self._parentheses_pattern.sub("", text)
            if len(text) == length:
                break
            length = len(text)
        
        text = "".join(['￥', text, '￥'])

        # step6: 去除网址
        text = self._url_pattern.sub("", text)

        # step7: 去除 email
        text = self._email_pattern.sub("", text)

        # step8: 去除电话号码
        text = self._cell_phone_pattern.sub("", text)
        text = self._landline_phone_pattern.sub("", text)

        text = text[1:-1]

        return text 
