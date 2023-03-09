# -*- coding:utf-8 -*-
# Author: lqxu

""" 基础的工具类 """

from typing import *
from itertools import accumulate

import numpy as np


def _generate_random_string(char_set: Set[str], str_len: int):
    """ 生成随机的字符串, 字符在 char_set 中, 长度由 str_len 决定 """
    char_list = list(char_set)
    indices = np.random.randint(low=0, high=len(char_set), size=str_len)
    return "".join([char_list[index] for index in indices])


def generate_random_fingerprint():

    """
    生成随机的 fingerprint, 替代 huggingface 生成的 fingerprint \n
    原版的使用 dill 进行序列化, 速度很慢, 不使用 fingerprint, 运行速度会快很多, 但是对代码质量的要求偏高 \n
    关于 fingerprint, 可以参考: https://huggingface.co/docs/datasets/main/en/about_cache
    """

    from string import ascii_lowercase, digits
    char_set = set(ascii_lowercase + digits)
    str_len = 16
    return _generate_random_string(char_set, str_len)


class SimpleNestedSequence:
    """
    将多个 Sequence 对象合并称为一个 Sequence 对象。
    注意: 这里并没有实现所有的 Sequence 接口, 比方说 `__reversed__`, `__contains__` 方法等等,
    只实现了最简单的 `__iter__` 和 `__getitem__` 方法。
    完整版的接口请参考: https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes 
    """
    
    def __init__(self, sequence_objs) -> None:        
        self.sequence_objs = list(sequence_objs)  # 只拷贝第一层, 不拷贝第二层
        
        self.lengths = [len(sequence_obj) for sequence_obj in sequence_objs]
        self.total_length = sum(self.lengths)
        
        self.endpoints = list(accumulate(self.lengths))
        
    def __iter__(self) -> Iterator:
        for sequence_obj in self.sequence_objs:
            for obj in sequence_obj:
                yield obj 
    
    def __len__(self) -> int:
        return self.total_length
    
    def _find_outer_idx(self, idx: int) -> int:
        for outer_idx, endpoint in enumerate(self.endpoints):
            if idx < endpoint:
                return outer_idx
        
        raise IndexError("index out of range")

    def __getitem__(self, idx: int) -> Any:
        
        outer_idx = endpoint = -1
        
        for outer_idx, endpoint in enumerate(self.endpoints):
            if idx < endpoint:
                break 
        
        if outer_idx == -1:
            raise IndexError("index out of range")
        
        inner_idx = idx - endpoint
        
        return self.sequence_objs[outer_idx][inner_idx]
