# -*- coding:utf-8 -*-
# Author: lqxu

"""
将 jionlp 中的 remove_stopwords 方法单独抽出来, 方便修改和运行加速。

jionlp 版本: 1.4.30
"""

import regex as re 
from typing import * 

from jionlp.dictionary.dictionary_loader import stopwords_loader, world_location_loader, china_location_loader, negative_words_loader
from jionlp.rule.rule_pattern import TIME_PATTERN, NUMBER_PATTERN, CHINESE_CHAR_PATTERN, LOCATION_PATTERN


class StopWordsRemover:

    def __init__(self) -> None:
        # 停用词表, 包含否定词
        self._stop_words1 = set(stopwords_loader())
        # 停用词表, 去除否定词
        self._stop_words2 = self._stop_words1 - set(negative_words_loader())
        # 地点词表, 包括中国和世界地名
        self._location_words = set(self._location_loader())

        self._time_pattern = re.compile("^" + TIME_PATTERN + "$")
        self._location_pattern = re.compile("^" + LOCATION_PATTERN + "$")
        self._number_pattern = re.compile("^" + NUMBER_PATTERN + "$")
        self._chinese_char_pattern = re.compile(CHINESE_CHAR_PATTERN)

    @staticmethod
    def _location_loader() -> List[str]:
        """ 加载世界和中国的位置词典 """
        china_location = china_location_loader()
        china_list = list()
        china_list.extend(list(china_location.keys()))
        for prov, cities in china_location.items():
            china_list.append(prov)
            china_list.append(cities['_full_name'])
            china_list.append(cities['_alias'])
            for city, counties in cities.items():
                if city.startswith('_'):
                    continue
                china_list.append(city)
                china_list.append(counties['_full_name'])
                china_list.append(counties['_alias'])
                for county, info in counties.items():
                    if county.startswith('_'):
                        continue
                    china_list.append(county)
                    china_list.append(info['_full_name'])
                    china_list.append(info['_alias'])
                    
        world_location = world_location_loader()
        world_list = list()
        world_list.extend(list(world_location.keys()))
        for _, countries in world_location.items():
            world_list.extend(list(countries.keys()))
            for _, info in countries.items():
                if 'main_city' in info:
                    world_list.extend(info['main_city'])
                world_list.append(info['full_name'])
                world_list.append(info['capital'])

        return china_list + world_list

    def __call__(
            self, tokens: List[str], remove_time=True, remove_location=True, 
            remove_number=True, remove_non_chinese=True, save_negative_words=False
        ) -> List[str]:
        
        """ 去除停用词 """

        # 根据 否定词 来确定使用的通用词表
        stop_words = self._stop_words2 if save_negative_words else self._stop_words1

        results = []

        for token in tokens:
            if token in stop_words:
                # 如果 token 在停用词表中
                continue

            if remove_time and self._time_pattern.match(token):
                # 如果 token 和 time_pattern 相互匹配
                continue

            if remove_location and (token in self._location_words or self._location_pattern.match(token)):
                # 如果 token 在地名表中, 或者 token 和 _location_pattern 相互匹配
                continue

            if remove_number and self._number_pattern.match(token):
                # 如果 token 和 _number_pattern 相互匹配
                continue

            if remove_non_chinese and not self._chinese_char_pattern.search(token):
                # 如果 token 中不包含中文字符
                continue

            results.append(token)
        
        return results
