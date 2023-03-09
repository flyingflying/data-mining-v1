# -*- coding:utf-8 -*-
# Author: lqxu

import jionlp

from core import SentenceSpliter

if __name__ == '__main__':
    
    split_sentence = SentenceSpliter()
    
    test_cases = [
        "。” 呵呵",
        "。。。。。。哈哈",
        "今天的天气真好, “清空万里。” 明天不知道怎么样？出来玩吗",
        "红红火火，嘻嘻哈哈"
    ]
    
    for test_case in test_cases:
        print(split_sentence(test_case))
        print(jionlp.split_sentence(test_case))
        
        print(split_sentence(test_case, criterion="fine"))
        print(jionlp.split_sentence(test_case, criterion="fine"))
