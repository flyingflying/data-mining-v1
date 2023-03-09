# -*- coding:utf-8 -*-
# Author: lqxu

""" 
文档分句, 从 JioNLP 的方法中抽取出来, 方便定制和加速

jionlp 版本: 1.4.30
"""

import regex as re 


class SentenceSpliter:
    
    """ 标点符号分句 """
    
    def __init__(self):
        # 细粒度分词时的标点
        self.puncs_fine = {'……', '\r\n', '，', '。', ';', '；', '…', '！', '!', '?', '？', '\r', '\n', '“', '”', '‘', '’', '：'}
        # 细粒度分词时的正则
        self.puncs_fine_ptn = re.compile('([，：。;“”；…！!?？\r\n])')
        # 粗粒度分词时的标点
        self.puncs_coarse = {'。', '！', '？', '\n', '“', '”', '‘', '’'}
        # 粗粒度分词时的正则
        self.puncs_coarse_ptn = re.compile('([。“”！？\n])')
        # 前引号和后引号列表
        self.front_quote_set = {'“', '‘'}
        self.back_quote_set = {'”', '’'}

    def __call__(self, text, criterion='coarse'):

        if criterion == "coarse":
            puncs_ptn = self.puncs_coarse_ptn
            puncs_set = self.puncs_coarse
        elif criterion == "fine":
            puncs_ptn = self.puncs_fine_ptn
            puncs_set = self.puncs_fine
        else:
            raise ValueError('The parameter `criterion` must be `coarse` or `fine`.')

        # 这样分词会将标签分出来, 因此需要处理标点的情况
        # 需要对 segments 有基础的认识: 即 segment 要不然是文本, 要不然是标点。
        # 文本 segment 的左右一定是标点符号, 不会存在两个文本 segment 相连的情况
        # 但是会出现多个标点 segment 相连的情况
        # 另外, 所有的标点 segment 一定是单字符的
        # 所有的文本都会保留, 分句前后的文本长度应该是一致的
        segments = puncs_ptn.split(text)        
        sentences, buffer = [], []
        
        for segment in segments:
            # 如果 segment 是空串 ==> 过滤掉
            if len(segment) == 0:
                continue

            # 如果 buffer 是空 ==> segment 直接添加到 buffer 中
            if len(buffer) == 0:
                buffer.append(segment)
                continue

            # 如果 segment 是标签符号, 原则上是直接加入 buffer 中, 除非出现 标点+前引号 的情况, 需要切断
            if segment in puncs_set:
                if segment in self.front_quote_set and buffer[-1] in puncs_set:
                    sentences.append("".join(buffer))
                    buffer.clear()

                buffer.append(segment)
                continue
            
            # 此时的 segment 不是标签符号了, 而是文本, 也就意味着上一个 segment 一定是标签符号
            # 原则上来说, 是在每一个文本 segment 前切断, 但是需要处理引号的情况
            if buffer[-1] in self.front_quote_set:  # 如果上一个 segment 是前引号 ==> 不切断
                buffer.append(segment)
                continue

            # 如果上一个 segment 是后引号, 原则上来说也是不切断的, 但是有特殊情况
            if buffer[-1] in self.back_quote_set:
                # 如果前两个 segment 是标点符号 ==> 切断
                if len(buffer) != 1 and buffer[-2] in puncs_set:
                    sentences.append("".join(buffer))
                    buffer.clear()

                buffer.append(segment)
                continue
            
            # 上一个 segment 既不是引号 ==> 分句
            sentences.append("".join(buffer))
            buffer.clear()
            buffer.append(segment)
        
        if len(buffer) != 0:
            sentences.append("".join(buffer))
        
        # TODO: 流程优化
        # 标点+前引号 ==> 切断, 如果把 `前引号` 当作普通字符, 就不存在问题了
        # 标点+后引号+文本 ==> 在 后引号 和 文本 之间切割, 能否将 标点+后引号 作为标点, 而不是将后引号单独作为标点

        return sentences
