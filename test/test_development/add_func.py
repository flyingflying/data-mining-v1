# -*- coding:utf-8 -*-
# Author: lqxu

"""
在 Python 中, 如果要替换函数, 一定要替换 **类** 的函数, 不要替换 **实例** 的函数。

替换 **类** 的函数后, 无论是替换之前申明的实例, 还是替换之后申明的实例, 都会运行替换后的函数。
"""

import pandas as pd 
from pandarallel import pandarallel

pandarallel.initialize()

df = pd.DataFrame({
    "f1": [100, 200], 
    "f2": [300, 400]
})

pandarallel.initialize()

print(hasattr(df, "parallel_apply"))

print(df.parallel_apply)


class AddFuncsTest:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        print(a)


t = AddFuncsTest()

t.forward(50)


def custom_forward(self, a):
    print(a)
    print(a)


print(AddFuncsTest.forward)
AddFuncsTest.forward = custom_forward
print(AddFuncsTest.forward)

t.forward(50)
