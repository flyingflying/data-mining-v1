# -*- coding:utf-8 -*-
# Author: lqxu

import os
import json
from typing import * 
from itertools import chain

import jionlp 

from core.preprocess import TextCleaner, SentenceSpliter


def _load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    return data


class JioKeyphrases:
    
    """
    jionlp 中的关键词抽取方案
    
    大体上流程如下:
        1. 清洗文本, 细粒度分句, 分词, 词性标注
        2. 通过滑窗的方式构建候选短语, 根据规则 (停用词和词性) 筛选掉不符合要求的候选短语
        3. 给每一个短语进行打分, 打分公式为: tf_idf_weight * length_weight * pos_combo_weight + topic_weight * topic_theta
        4. 将短语按照 window_size / n_tokens 排序, 然后用 mmr 算法对分数进行修正 (两个短语的相似度等于两个短语中相同 token 的占比)
        5. 按照分数进行排序, 取分数高的短语即可
    
    TODO: 
        1. 如果需要实现其它算法, 一般情况下, 除了 `短语打分` 部分, 其它部分都是可以复用的, 那么可以抽象出一个基类来, 方便接下来的实现
        2. 如果想更换分词器, 需要大范围测试分词器的效果, 调整短语筛选的方案
    """
    
    def __init__(
        self, segment_tool: str = "jiojio", 
        # 不希望出现在 短语 中的 词语
        remove_word_list: List[str] = None,
        # 希望出现在 短语 中的词语, value 表示需要增加的 tf-idf 权重值, 建议是 1 / tf
        specified_words: Dict[str, float] = None, 
        # 不希望出现在结果中的 短语
        remove_phrase_list: List[str] = None,
        
        idf_file_name: str = "default_idf.json", 
        word_weights_file_name: str = "default_word_weights.json", 
    ):

        self.remove_word_list = set(remove_word_list) if remove_word_list is not None else None
        self.remove_phrase_list = set(remove_phrase_list) if remove_phrase_list is not None else None
        self.specified_words = specified_words

        if segment_tool == "jiojio":
            
            import jiojio 

            # 词性参考: https://hanlp.hankcs.com/docs/annotations/pos/pku.html
            # 词性参考: https://github.com/dongrixinyu/jiojio/blob/master/jiojio/pos/pos_types.yml

            # 所有的词性列表
            self.all_pos = set(sorted(list(jiojio.pos_types()['model_type'].keys())))

            # 虚词列表, 分别是: u助词, p介词, c连词, (y语气词), (e叹词), o拟声词, w标点符号
            self.func_pos = {'u', 'p', 'c', 'y', 'e', 'o', 'w'}

            # 短语中允许出现的词性
            
            # 宽松版, 只要不是虚词即可
            self.loose_pos = self.all_pos - self.func_pos
            
            # 严格版, 只允许名词和形容词出现, 大致可以分成三大类:
            # 1. 名词类: n名词, nr人名, ns地名, nt机构名, (nx外文字符), (nz其它专名)
            # 2. 形容词类: a形容词, ad副形词, an名行词
            # 3. 其它词类: vn名动词, vd副动词, (vx)
            self.strict_pos = {'a', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'ad', 'an', 'vn', 'vd', 'vx'}
            
            # 词性转换字典
            self.trans_map = {'nr1': 'nr', 'nrf': 'nr', 'vi': 'v', 'wx': 'w', 'x': 'nz'}

            # remove_word_list: 短语中不希望出现的词语
            if self.remove_word_list is not None:
                jiojio.init(cws_user_dict=True, pos_rule=True, pos=True)
                for word in self.remove_word_list:
                    jiojio.add_word(word=word)
            
            else:
                jiojio.init(pos_rule=True, pos=True)

            self.segment = jiojio.cut 
        
        else:
            
            raise ValueError("目前只支持 jiojio 分词器")

        # 加载词语的 idf 权重值, UNK 的 idf 权重值是 `中位数``
        self.idf_weight_dict = _load_json(os.path.join("./weights", idf_file_name))
        self.unk_idf_weight = sorted(self.idf_weight_dict.values())[len(self.idf_weight_dict) // 2]

        # 加载通过主题模型计算出来的词权重, UNK 的权重值是 `平均数` 的一半
        self.word_weights_dict = _load_json(os.path.join("./weights", word_weights_file_name))
        self.unk_word_weight = sum(self.word_weights_dict.values()) / len(self.word_weights_dict) / 2

        # 短语的 token 数权重
        # 人工抽取的 关键短语 长度符合一个分布, 算法提取的 关键短语 长度符合一个分布, 两个分布通过这个权重进行修正
        # TODO: 这个权重是怎么计算出来的? 
        self.length_weight_dict = {
            1: 1.0, 2: 5.6, 3: 1.1, 4: 2.0, 5: 0.7, 6: 0.9, 
            7: 0.48, 8: 0.43, 9: 0.24, 10: 0.15, 11: 0.07, 12: 0.05
        }
        self.unk_length_weight = 0.01  # 在大于 7 时选取

        # 短语词性组合权重字典 (短语的 `首词语` 和 `尾词语` 词性组合的权重)
        # TODO: 这个权重是怎么计算出来的? 按照作者的说法, 其相当于可调节参数 ...
        # reference: https://github.com/dongrixinyu/chinese_keyphrase_extractor/issues/6 
        self.pos_combo_weight_dict = _load_json("./weights/pos_combine_weights.json")
        self.unk_pos_combo_weight = 1.0

        # 停用词列表 (代码中仅仅让停用词的 tf-idf 值为 0, 和主题模型相关的权重值没有设为 0)
        self.stop_words = jionlp.stopwords_loader()
        self.stop_words.append('\n')
        self.stop_words = set(self.stop_words)
        
        # 数据预处理方法
        self.clean_text = TextCleaner()
        self.split_sentence = SentenceSpliter()

    def __call__(
        self, text: str, top_k: int = 5,
        use_strict_pos: bool = True, allow_location_pos: bool = True, allow_person_pos: bool = True, 
        topic_theta: float = 0.5, max_window_size: int = 12, return_scores: bool = False
    ):

        # 短语中允许出现的词性
        if use_strict_pos:
            allow_pos = set(self.strict_pos)
        else:
            allow_pos = set(self.loose_pos)
        
        # 经典的词性: ns 地名 nr 人名 nt 机构名
        if allow_location_pos:
            allow_pos.add("ns")
        if allow_person_pos:
            allow_pos.add("nr")
        
        # 数据清洗
        text = self.clean_text(text)
        sentences = self.split_sentence(text, "fine")

        # 分词 & 词性标注
        segmented_results = [self.segment(sentence) for sentence in sentences]
        token_lists = [[token for token, pos in result] for result in segmented_results]
        pos_lists = [[self.trans_map.get(pos, pos) for token, pos in result] for result in segmented_results]

        # 计算词频
        all_tokens = list(chain(*token_lists))
        n_tokens = len(all_tokens)
        tf_dict = Counter(all_tokens)
        tf_dict = {token: tf / n_tokens for token, tf in tf_dict.items()}

        # 计算 tf-idf 权重值
        tf_idf_lists = []
        for token_list, pos_list in zip(token_lists, pos_lists):
            
            tf_idf_list = []
            
            for token, pos in zip(token_list, pos_list):
                # 停用词的权重值为 0.0
                if token in self.stop_words or pos not in self.all_pos:
                    tf_idf_list.append(0.0)
                    continue
                
                tf_idf = tf_dict[token] * self.idf_weight_dict.get(token, self.unk_idf_weight)

                # 看到这里, 我有一点懵
                if self.specified_words is not None and token in self.specified_words:
                    tf_idf += self.specified_words[token]
            
                tf_idf_list.append(tf_idf)

            tf_idf_lists.append(tf_idf_list)

        # 构建候选短语集合
        phrase_tokens_list = []
        phrase_score_list = []
        
        for token_list, pos_list, tf_idf_list in zip(token_lists, pos_lists, tf_idf_lists):
            n_tokens = len(token_list)
            
            for window_size in range(min(max_window_size, n_tokens)):
                window_size += 1  # window_size 表示的是滑窗大小
                
                for start_idx in range(0, n_tokens - window_size + 1):
                    end_idx = start_idx + window_size

                    phrase = token_list[start_idx:end_idx]
                    phrase_pos = pos_list[start_idx:end_idx]
                    
                    if not self._is_candidate_phrase(token_list=phrase, pos_list=phrase_pos, allow_pos=allow_pos, use_strict_rule=use_strict_pos):
                        continue
                    
                    if self.remove_word_list is not None and any(token in self.remove_word_list for token in phrase):
                        continue
                    
                    if self.specified_words is not None and not any(token in self.specified_words for token in phrase):
                        continue
                    
                    if self.remove_phrase_list is not None and "".join(phrase) in self.remove_phrase_list:
                        continue

                    pos_combo = phrase_pos[0] if window_size == 1 else "|".join([phrase_pos[0], phrase_pos[-1]])
                    pos_combo_weight = self.pos_combo_weight_dict.get(pos_combo, self.unk_pos_combo_weight)

                    length_weight = self.length_weight_dict.get(window_size, self.unk_length_weight)

                    topic_weight = sum([self.word_weights_dict.get(token, self.unk_word_weight) for token in phrase]) / window_size
                    
                    tf_idf_weight = sum(tf_idf_list[start_idx:end_idx])

                    phrase_tokens_list.append(phrase)
                    phrase_score_list.append(
                        tf_idf_weight * length_weight * pos_combo_weight + topic_weight * topic_theta
                    )

        # 使用 MMR 策略进行短语筛选, 保证返回短语之间的差异性
        indices = [result[0] for result in sorted(enumerate(phrase_tokens_list), key=lambda item: len(item[1]), reverse=True)]
        
        prev_phrases = []
        for index in indices:
            cur_phrase = set(phrase_tokens_list[index])

            if len(prev_phrases) == 0:
                prev_phrases.append(cur_phrase)
                continue

            sim_ratio = max([len(cur_phrase & prev_phrase) / len(cur_phrase) for prev_phrase in prev_phrases])

            phrase_score_list[index] *= (1 - sim_ratio)

            if sim_ratio != 1:  # 加速计算
                prev_phrases.append(cur_phrase)

        indices = [result[0] for result in sorted(enumerate(phrase_score_list), key=lambda item: item[1], reverse=True) if result[1] != 0]
        if return_scores:
            return [("".join(phrase_tokens_list[index]), phrase_score_list[index]) for index in indices][:top_k]
        
        return ["".join(phrase_tokens_list[index]) for index in indices][:top_k]

    def _is_candidate_phrase(
        self, token_list, pos_list, 
        allow_pos, use_strict_rule: bool = True,
        min_phrase_len: int = 1, max_phrase_len: int = 25, 
        max_func_tokens: int = 1, max_stop_tokens: int = 0
    ):

        # rule 1: 限制短语的 token 数
        if len(token_list) > 12:
            return False

        # rule 2: 短语本身限制
        phrase = "".join(token_list)
        # 2.1 短语长度限制
        if not min_phrase_len <= len(phrase) <= max_phrase_len:
            return False
        # 2.2 短语不能是停用词
        if phrase in self.stop_words:
            return False

        # rule 3: 词性限制
        if use_strict_rule:
            # 严格版的词性只能是 allow_pos 中的词
            if not all(pos in allow_pos for pos in pos_list):
                return False

        else:
            # 非严格版的允许有虚词出现, 但是限制停用词的情况
            n_func_tokens = sum([pos in self.func_pos for pos in pos_list])
            if n_func_tokens > max_func_tokens:
                return False

            n_stop_tokens = sum([token in self.stop_words for token in token_list])
            if n_stop_tokens > max_stop_tokens:
                return False

        # rule4: 首尾词限制
        if use_strict_rule:
            # 严格版限制首词不能是动词, 尾词必须是动词/形容词
            if pos_list[0] in {"v", "vd", "vx"} or pos_list[-1] in {"a", "ad", "vd", "vx", "v"}:
                return False

        else:
            # 4.1 首尾词不能是虚词
            if pos_list[0] in self.func_pos or pos_list[-1] in self.func_pos:
                return False
            # 4.2 首尾词不能是停用词
            if token_list[0] in self.stop_words or token_list[-1] in self.stop_words:
                return False
            # 4.2 尾词不能时 动词 / 副词
            if pos_list[-1] in {"v", "d"}:
                return False

        # rule 5: 其它后处理
        if token_list[-1] == "时":
            return False

        return True 
