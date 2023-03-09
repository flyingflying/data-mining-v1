# -*- coding:utf-8 -*-
# Author: lqxu

import os 
import json
from typing import * 
from itertools import chain

import jionlp
from pkuseg import pkuseg

from core.preprocess import TextCleaner, SentenceSpliter, StopWordsRemover
from core.algorithms import text_rank_for_summary


def _load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    return data


class TextRankSummary:
    """ HanLP v1 中的 TextRank 算法复现 """
    def __init__(self) -> None:
        self.clean_text = TextCleaner()
        self.split_sentence = SentenceSpliter()
        self.tokenize = pkuseg().cut 
        self.remove_stopwords = StopWordsRemover()

    def __call__(self, document: str, n_sentences: int = 5) -> str:
        # step1: 清洗文本
        document = self.clean_text(document)
        # step2: 分句
        sentences = self.split_sentence(document)
        # 分句过程中是不会删除任何字符的, 清洗文本中只会将多个换行符变成单个换行符
        # 这里我们需要将所有的换行符都去掉
        sentences = [sentence.strip() for sentence in sentences]
        # step3: 分词 + 去停用词
        tokens_list = [self.remove_stopwords(self.tokenize(sentence)) for sentence in sentences]
        # step4: 对句子进行排序
        rank = text_rank_for_summary(tokens_list=tokens_list)
        # step5: 后处理
        return "".join([sentences[idx] for idx in rank[:n_sentences]]) 


class JioSummary:
    """
    jionlp 中的摘要抽取算法复现
    
    流程:
        1. 对 document 进行清洗, 分句 等操作得到 sentences
        2. 对 sentences 中的每一个句子进行过滤, 分词, 词性标注等操作, 得到 word_pos_lists
        3. 以 document 为基础, 计算每一个词的 tf-idf 权重 (注意: 是以 document 为基础, 不是以 sentence 为基础)
        4. 对每一个 sentence, 计算 sentence_weight = topic_weight * topic_theta + tfidf_weight
            其中 topic_weight 是 sentence 中每一个 word distinctiveness 相加得到, tfidf_weight 是上一步计算出来的每一个词的权重相加
        5. 最后按分数排序, 按原文顺序输出 
    """
    
    def __init__(
        self, idf_file_name: str = "default_idf.json", 
        word_weights_file_name: str = "default_word_weights.json", 
        segment_tool: str = "jiojio"
    ):
        
        self.clean_text = TextCleaner()
        self.split_sentence = SentenceSpliter()

        if segment_tool == "pkuseg":
            self.segment = pkuseg(postag=True).cut
            self.pos_name = {
                "n", "t", "s", "f", "m", "q", "b", "r", "v", "a", "z", "d", "p", "c", "u", "y", "e", "o", "i", "l",
                "j", "h", "k", "g", "x", "w", "nr", "ns", "nt", "nx", "nz", "vd", "vn", "vx", "ad", "an"
            }

        elif segment_tool == "jiojio":
            import jiojio 
            jiojio.init(pos_rule=True, pos=True)
            self.segment = jiojio.cut
            self.pos_name = set(sorted(list(jiojio.pos_types()['model_type'].keys())))
        
        else:
            raise ValueError("目前只支持 jiojio 和 pkuseg 分词器")
        
        # self.strict_pos_name = {'a', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'ad', 'an', 'vn', 'vd', 'vx'}

        self.stop_words = set(jionlp.stopwords_loader())
        self._idf_loader(idf_file_name)
        self._word_weights_loader(word_weights_file_name)

    def _idf_loader(self, file_name: str):
        """ 加载 idf 字典, 并计算 unk 的 idf 值 (中位数) """
        self.idf_dict: Dict[str, float] = _load_json(
            os.path.join("./weights", file_name)
        )
        self.unk_idf: float = sorted(self.idf_dict.values())[len(self.idf_dict) // 2]

    def _word_weights_loader(self, file_name: str):
        """ 用主题模型计算出来的权重值, 默认是 word distinctiveness, 再进行 min-max 归一化 """
        self.word_weights_dict: Dict[str, float] = _load_json(
            os.path.join("./weights", file_name)
        )
        # 计算 未知词/未登录词/oov/unk 的权重: 平均数除以2 (不知道为什么要除以 2)
        self.unk_word_weights: float = sum(self.word_weights_dict.values()) / (2 * len(self.word_weights_dict))

    def __call__(self, document: str, max_length: int = 200, topic_theta: float = 0.2, lead_3_weight: float = 1.2) -> str:

        # step0: 参数检查
        assert 0.0 <= topic_theta <= 1.0, "topic_theta 应该在 0 到 1 之间"
        assert lead_3_weight >= 1.0, "lead_3_weight 应该大于 0"

        # step1: 文本清洗
        document = self.clean_text(document)
        if len(document) < max_length:  # 文本过短, 则不需要提取, 直接返回即可
            return document

        # step2: 文本分句, 句子分词, 词性标注
        # sentences: List[str] = self.split_sentence(document)
        sentences = [sentence for sentence in jionlp.split_sentence(document) if jionlp.check_any_chinese_char(sentence)]
        if len(sentences) == 1:
            return document[:max_length]
        word_pos_lists: List[List[Tuple[str, str]]] = [self.segment(sentence) for sentence in sentences]

        # step3: 计算 TF 值
        word_lists = [[word for word, pos in word_pos_list] for word_pos_list in word_pos_lists]
        all_words = list(chain(*word_lists))
        tf_dict = Counter(all_words)

        # step4: 计算每一个句子的权重值
        sentence_weights = []
        for sentence, word_pos_list in zip(sentences, word_pos_lists):
            tfidf_weight = topic_weight = 0.0

            for word, pos in word_pos_list:
                # 增加 word_weight
                topic_weight += self.word_weights_dict.get(word, self.unk_word_weights)

                # 增加 tfidf 的权重值
                if word in self.stop_words and pos not in self.pos_name:  # 停用词部分的 tfidf 值为 0.0
                    continue
                tfidf_weight += tf_dict[word] * self.idf_dict.get(word, self.unk_idf) / len(all_words)

            # 句子的 tfidf 值等于句子中每一个词的 tfidf 值取平均, topic_weight 同理
            tfidf_weight /= len(word_pos_list)
            topic_weight /= len(word_pos_list)

            sentence_weight = topic_weight * topic_theta + tfidf_weight

            if not 15 <= len(sentence) <= 70:  # 过长和过短的句子需要减少其权重值
                sentence_weight *= 0.7

            if len(sentence_weights) < 3:  # 前三句需要增大其权重值
                sentence_weight *= lead_3_weight

            sentence_weights.append(sentence_weight)

        # step5: 构建摘要文本
        indices = [index for index, value in sorted(enumerate(sentence_weights), key=lambda item: item[1])][::-1]
        
        cur_length = 0
        results = []  # index
        last_index = len(sentences) - 1

        for index in indices:
            sentence = sentences[index]
            
            if cur_length + len(sentence) > max_length:
                if len(results) == 0:  # rank 最高的那一句话长度已经超过 max_length
                    return sentence[:max_length]
                return "".join(sentences[idx] for idx in sorted(results))

            else:
                results.append(index)
                if index == last_index:  # 如果有句子的分数比最后一句还小, 那么就不需要了
                    return "".join(sentences[idx] for idx in sorted(results))
                cur_length += len(sentences[index])

        return document[:max_length]
