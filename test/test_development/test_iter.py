# -*- coding:utf-8 -*-
# Author: lqxu

from tqdm import tqdm


a = [1, 2, 3, 4, 5]

# 有总数
for _ in tqdm(a):
    pass 

print()

# 没有总数
for _ in tqdm(enumerate(a)):
    pass 

print()

# 有总数
for _ in enumerate(tqdm(a, desc="哈哈")):
    print(_)

print()

# 没有总数
for _ in tqdm(a.__iter__()):
    pass 

print()
