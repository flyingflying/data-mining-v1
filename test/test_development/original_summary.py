# -*- coding=utf-8 -*-
# library: jionlp
# author: dongrixinyu
# license: Apache License 2.0
# Email: dongrixinyu.89@163.com
# github: https://github.com/dongrixinyu/JioNLP
# description: Preprocessing tool for Chinese NLP


import os
import json
import numpy as np

from collections import Counter

import jiojio
import jionlp
from jionlp import logging
from jionlp.rule import clean_text
from jionlp.rule import check_any_chinese_char
from jionlp.gadget import split_sentence
from jionlp.dictionary import stopwords_loader
from jionlp.dictionary import idf_loader

DIR_PATH = os.path.dirname(os.path.join(jionlp.__file__, "algorithm", "summary"))
# DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class ChineseSummaryExtractor(object):
    """ 从中文文本中抽取关键句子，作为文本摘要。主要针对新闻效果较好。
    但改进空间很大，此功能仅作为 baseline。

    原理简述：为每个文本中的句子分配权重，权重计算包括 tfidf 方法的权重，以及
    LDA 主题权重，以及 lead-3 得到位置权重，并在最后结合 MMR 模型对句子做筛选，
    得到抽取式摘要。（默认使用 jiojio 的分词工具效果好）

    Args:
        text(str): utf-8 编码中文文本，尤其适用于新闻文本
        summary_length(int): 指定文摘的长度（软指定，有可能超出）
        lead_3_weight(float): 文本的前三句的权重强调，取值必须大于1
        topic_theta(float): 主题权重的权重调节因子，默认0.2，范围（0~无穷）
        allow_topic_weight(bool): 考虑主题突出度，它有助于过滤与主题无关的句子

    Returns:
        str: 文本摘要

    Examples:
        >>> import jionlp as jio
        >>> text = '不交五险一金，老了会怎样？众所周知，五险一金非常重要...'
        >>> summary = jio.summary.extract_summary(text)
        >>> print(summary)

        # '不交五险一金，老了会怎样？'

    """
    def __init__(self):
        self.unk_topic_prominence_value = 0.

    def _prepare(self):
        self.pos_name = set(sorted(list(jiojio.pos_types()['model_type'].keys())))
        # self.pos_name = set(['a', 'ad', 'an', 'c', 'd', 'f', 'm', 'n', 'nr', 'nr1', 'nrf', 'ns', 'nt',
        #                      'nz', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'vd', 'vi', 'w', 'wx', 'x'])

        self.strict_pos_name = ['a', 'n', 'nr', 'ns', 'nt', 'nx', 'nz',
                                'ad', 'an', 'vn', 'vd', 'vx']
        jiojio.init(pos_rule=True, pos=True)

        # 加载 idf，计算其 oov 均值
        self.idf_dict = idf_loader()
        self.median_idf = sorted(self.idf_dict.values())[len(self.idf_dict) // 2]

        # 读取停用词文件
        self.stop_words = stopwords_loader()

        # 加载 lda 模型参数
        self._lda_prob_matrix()

    def _lda_prob_matrix(self):
        """ 读取 lda 模型有关概率分布文件，并计算 unk 词的概率分布 """
        # 读取 p(topic|word) 概率分布文件，由于 lda 模型过大，不方便加载并计算
        # 概率 p(topic|word)，所以未考虑 p(topic|doc) 概率，可能会导致不准
        # 但是，由于默认的 lda 模型 topic_num == 100，事实上，lda 模型是否在
        # 预测的文档上收敛对结果影响不大（topic_num 越多，越不影响）。

        dict_dir_path = os.path.join(os.path.dirname(os.path.dirname(DIR_PATH)), 'dictionary')

        with open(os.path.join(dict_dir_path, 'topic_word_weight.json'),
                  'r', encoding='utf8') as f:
            self.topic_word_weight = json.load(f)
        self.word_num = len(self.topic_word_weight)

        # 读取 p(word|topic) 概率分布文件
        with open(os.path.join(dict_dir_path, 'word_topic_weight.json'),
                  'r', encoding='utf8') as f:
            self.word_topic_weight = json.load(f)
        self.topic_num = len(self.word_topic_weight)

        self._topic_prominence()  # 预计算主题突出度

    def __call__(self, text, summary_length=200, lead_3_weight=1.2,
                 topic_theta=0.2, allow_topic_weight=True):

        # 输入检查
        if type(text) is not str:
            raise ValueError('type of `text` should only be str')
        try:
            # 初始化加载
            if self.unk_topic_prominence_value == 0.:
                self._prepare()

            if lead_3_weight < 1:
                raise ValueError('the params `lead_3_weight` should not be less than 1.0')
            if len(text) <= summary_length:
                return text

            # step 0: 清洗文本
            text = clean_text(text)

            # step 1: 分句，并逐句清理杂质
            sentences_list = split_sentence(text)

            # step 2: 分词与词性标注
            sentences_segs_dict = dict()
            counter_segs_list = list()
            for idx, sen in enumerate(sentences_list):
                if not check_any_chinese_char(sen):  # 若无中文字符，则略过
                    continue

                sen_segs = jiojio.cut(sen)
                sentences_segs_dict.update({sen: [idx, sen_segs, list(), 0]})
                counter_segs_list.extend(sen_segs)

            # step 3: 计算词频
            total_length = len(counter_segs_list)
            freq_counter = Counter([item[0] for item in counter_segs_list])
            freq_dict = dict(freq_counter.most_common())

            # step 4: 计算每一个词的权重
            for sen, sen_segs in sentences_segs_dict.items():
                sen_segs_weights = list()
                for word_pos in sen_segs[1]:
                    word, pos = word_pos
                    if pos not in self.pos_name and word in self.stop_words:  # 虚词权重为 0
                        weight = 0.0
                    else:
                        weight = freq_dict[word] * self.idf_dict.get(
                            word, self.median_idf) / total_length
                    sen_segs_weights.append(weight)

                sen_segs[2] = sen_segs_weights
                sen_segs[3] = len([w for w in sen_segs_weights if w != 0]) / len(sen_segs_weights) \
                    if len(sen_segs_weights) == 0 else 0

            # step 5: 得到每个句子的权重
            for sen, sen_segs in sentences_segs_dict.items():
                # tfidf 权重
                tfidf_weight = sum(sen_segs[2]) / len(sen_segs[2])

                # 主题模型权重
                if allow_topic_weight:
                    topic_weight = 0.0
                    for item in sen_segs[1]:
                        topic_weight += self.topic_prominence_dict.get(
                            item[0], self.unk_topic_prominence_value)
                    topic_weight = topic_weight / len(sen_segs[1])
                else:
                    topic_weight = 0.0

                sen_weight = topic_weight * topic_theta + tfidf_weight

                # 句子长度超过限制，权重削减
                if len(sen) < 15 or len(sen) > 70:
                    sen_weight = 0.7 * sen_weight

                # LEAD-3 权重
                if sen_segs[0] < 3:
                    sen_weight *= lead_3_weight

                sen_segs[3] = sen_weight

            # step 6: 按照 MMR 算法重新计算权重，并把不想要的过滤掉
            sentences_info_list = sorted(sentences_segs_dict.items(),
                                         key=lambda item: item[1][3], reverse=True)
            print([sentence_info[1][3] for sentence_info in sentences_info_list])
            
            for sentence_info in sentences_info_list:
                print(sentence_info[0].replace("\n", ""), sentence_info[1][3])
            
            mmr_list = list()
            for sentence_info in sentences_info_list:
                # 计算与已有句子的相似度
                sim_ratio = self._mmr_similarity(sentence_info, mmr_list)
                sentence_info[1][3] = (1 - sim_ratio) * sentence_info[1][3]
                mmr_list.append(sentence_info)

            # step 7: 按重要程度进行排序，选取若干个句子作为摘要
            if len(sentences_info_list) == 1:
                return sentences_info_list[0][0]
            total_length = 0
            summary_list = list()
            for idx, item in enumerate(sentences_info_list):
                if len(item[0]) + total_length > summary_length:
                    if idx == 0:
                        return item[0]
                    else:
                        # 按序号排序
                        summary_list = sorted(
                            summary_list, key=lambda item: item[1][0])
                        summary = ''.join([item[0] for item in summary_list])
                        return summary
                else:
                    summary_list.append(item)
                    total_length += len(item[0])
                    if idx == len(sentences_info_list) - 1:
                        summary_list = sorted(
                            summary_list, key=lambda item: item[1][0])
                        summary = ''.join([item[0] for item in summary_list])
                        return summary

            return text[:summary_length]
        except Exception as e:
            logging.error('the text is illegal. \n{}'.format(e))
            return ''

    def _mmr_similarity(self, sentence_info, mmr_list):
        """ 计算出每个句子和之前的句子相似性 """
        sim_ratio = 0.0
        notional_info = set([item[0] for item in sentence_info[1][1]
                             if item[1] in self.strict_pos_name])
        if len(notional_info) == 0:
            return 1.0
        for sen_info in mmr_list:
            no_info = set([item[0] for item in sen_info[1][1]
                           if item[1] in self.strict_pos_name])
            common_part = notional_info & no_info
            if sim_ratio < len(common_part) / len(notional_info):
                sim_ratio = len(common_part) / len(notional_info)
        return sim_ratio

    def _topic_prominence(self):
        """ 计算每个词语的主题突出度 """
        init_prob_distribution = np.array([self.topic_num for i in range(self.topic_num)])

        topic_prominence_dict = dict()
        for word in self.topic_word_weight:
            conditional_prob_list = list()
            for i in range(self.topic_num):
                if str(i) in self.topic_word_weight[word]:
                    conditional_prob_list.append(self.topic_word_weight[word][str(i)])
                else:
                    conditional_prob_list.append(1e-5)
            conditional_prob = np.array(conditional_prob_list)

            tmp_dot_log_res = np.log2(np.multiply(conditional_prob, init_prob_distribution))
            kl_div_sum = np.dot(conditional_prob, tmp_dot_log_res)  # kl divergence
            topic_prominence_dict.update({word: float(kl_div_sum)})

        tmp_list = [i[1] for i in tuple(topic_prominence_dict.items())]
        max_prominence = max(tmp_list)
        min_prominence = min(tmp_list)
        for k, v in topic_prominence_dict.items():
            topic_prominence_dict[k] = (v - min_prominence) / (max_prominence - min_prominence)

        self.topic_prominence_dict = topic_prominence_dict

        # 计算未知词汇的主题突出度，由于停用词已经预先过滤，所以这里不需要再考停用词无突出度
        tmp_prominence_list = [item[1] for item in self.topic_prominence_dict.items()]
        self.unk_topic_prominence_value = sum(tmp_prominence_list) / (2 * len(tmp_prominence_list))


if __name__ == '__main__':
    title = '全面解析拜登外交政策，至关重要的“对华三条”'
    text = """
新浪科技讯 北京时间10月22日消息，据英国《每日电讯报》报道，美国哈佛大学正在启动一个名为“个人基因研究项目”的实验，将对10位目前知名的科学家的DNA进行研究，并会将这些科学家独特的DNA奥秘公布在互联网上。

志愿公开DNA的科学家包括哈佛大学著名心理学家史蒂芬-平克、实习航天员阿瑟-戴森以及杜克大学助理教授玛沙-安格里斯特等人。他们将向哈佛大学“个人基因研究项目”捐献一小片皮肤用于研究并同意将研究结果公布于互联网之上。个人基因研究项目由哈佛大学医学院具体负责，旨在揭示人类基因的秘密，挑战人类的传统智慧。科学家们志愿捐献DNA的目的在于希望能够加快医学研究的进程，而不再像以往的实验那样为了保护研究对象的个人隐私而刻意防范。公布和可公开利用的基因信息越多，医学研究的进程也就越快。志愿者们同意，他们的基因信息、照片、病史、药物过敏史、种族背景以及其他的特点都可以公布于互联网之上。

研究人员承认，该项目既是一个科学实验，同时也是一个社会实验。至于公众对于这种公开个人基因信息的做法意见是否存在分歧，目前还不清楚。在美国，法律严禁保险公司或雇主因为健康问题歧视被保险人或求职者。但个人基因研究项目的首批10位志愿者都可能会遭遇被保险公司拒绝投保的问题。个人基因研究项目负责人、哈佛大学人类遗传学家乔治-切奇表示，“我们目前还不知道这项研究所带来的后果，但值得我们去探索。”当然，首批10位志愿者也要拥有丰富的基因知识，以保证他们能够理解根据他们的在线基因数据所得出的研究结论。此外，还有一个潜在的问题，那就是志愿者可能会遭到他们父母、兄弟姐妹以及儿女等亲人的反对，因为这些人身上也拥有志愿者一半的基因而且不愿意公开。

有的志愿者也在担心孩子们的未来，此外他们还要考虑配偶的看法。研究人员将会建议志愿者首先征求家人的意见。幸运的是，首批10位志愿者好象都比较健康，至少到目前为止还没有发现他们身上存在任何严重的健康问题。在实验中，志愿者将可能比以往更多地了解一些常见疾病的发病机率。首批志愿者之一、哈佛大学医学院首席信息官约翰-汉拉姆卡早期偶然看到了自己的基因信息，竟然发现他患有肥胖病的机率是常人的两倍。美国454生命科学公司创始人乔纳森-罗思伯格也承认，与“人类基因组图谱计划”相比，沃森个人的基因组图谱所用时间和金钱大幅下降，这主要得益于技术进步。尽管反复6次核查为60亿个碱基对排序，但绘制沃森基因组图谱的“工程”前后只用了不到2年时间，花费只有100万美元。

如果罗思伯格的预言得以实现，普通人出钱绘制个人基因组图谱将成为趋势。科学家介绍说，个人基因组图谱隐藏的遗传信息好似“生命密码”，一旦拥有，人们就可以在孩子出生之日起采取相应对策，减少患上特定疾病的风险，防患于未然。有例子表明，如果一个孩子的基因组图谱显示，这个孩子患上糖尿病的风险较高，那么就应该严格控制这个孩子的体重。这样一来，在这个孩子学会走路之前，他患上糖尿病的风险已经大大降低。还有不少科学家认为，绘制出个人基因组图谱，意义不仅在于降低患病风险，还可以铲除疾病根源。目前人类已知，单个基因缺陷能够引起难以治愈的疾病，比如遗传性胰腺病囊肿性纤维化和遗传性慢性舞蹈病等等。但如果拥有个人基因组图谱，我们就能够对病症做及早预防。

沃森曾对媒体表示，他希望通过自己的行动带动更多的人进行基因测序。 他认为，了解这些信息有助于提早预防癌症、心脏病、阿尔茨海默氏症等多种顽疾，甚至还能让人更富有同情心。“我们会了解有些人的天生局限，我们会放弃按自己的意愿培养孩子的一些不切实际的想法。会去帮助他们，而不是对他们发火。”个人DNA图谱时代的来临也同样面临着许多伦理问题。人们对待那些DNA存在异常的人群时能否不戴有色眼镜？同性恋已被证明与基因有关，但对同性恋者的排斥仍普遍存在。(刘妍)
"""

    cse_obj = ChineseSummaryExtractor()
    summary = cse_obj(text, topic_theta=0.2)
    print('summary_0.2topic: ', summary)
    # summary = cse_obj(text, topic_theta=0)
    # print('summary_no_topic: ', summary)
    # summary = cse_obj(text, topic_theta=0.5)
    # print('summary_0.5topic: ', summary)

