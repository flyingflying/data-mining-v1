# -*- coding:utf-8 -*-
# Author: lqxu

"""
HuggingFace Datasets 是一个优秀的数据并行化处理库, 但是其使用的多进程库是 multiprocess \n
multiprocess 库使用 dill 0.3.5 库进行数据序列化和存储, 或许在之前的 python 版本中, 其效率比 pickle 高。
实测在 python 3.8 的版本, pickle 库数据序列化的效率远远高于 dill 0.3.5, 可以通过以下代码进行测试:

>>> import time
>>> import dill
>>> import pkuseg
>>> import pickle

>>> tokenizer = pkuseg.pkuseg()

>>> start_time = time.time()
>>> with open("test.pkl", "wb") as writer:
>>>     pickle.dump(tokenizer, writer)
>>> end_time = time.time()
>>> print(f"pickle 库数据存储的时间: {round(end_time - start_time, 2)} 秒")
>>> start_time = time.time()

>>> with open("test.pkl", "wb") as writer:
>>>     dill.dump(tokenizer, writer)
>>> end_time = time.time()
>>> print(f"dill 库数据存储的时间: {round(end_time - start_time, 2)} 秒")

或许 dill 数据存储更加安全, 但是这个近 10 倍的运行效率差别是不可能接受的, 因此这里将 map 函数的多进程库从 multiprocessing 改成了
multiprocess, 使得运行效率得以提升。是否使用这个功能由用户自行决定。

修改过的函数仅仅使得 map 函数的运行时间变得可以接受, 但是并不是特别快, 如果想快, 可能还是需要从底层进行优化。
比方说, 在计算 figureprint 时, 也用到了 dill。
"""

import os
from typing import *
from copy import deepcopy
from multiprocessing import Pool, RLock

from tqdm import tqdm

import datasets as hf_datasets
from datasets import Dataset, Features
from datasets.arrow_dataset import _concatenate_map_style_datasets  # noqa
from datasets.arrow_dataset import logger, NonExistentDatasetError, is_caching_enabled, logging


# noinspection PyProtectedMember
def _acc_map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
) -> "Dataset":

    if keep_in_memory and cache_file_name is not None:
        raise ValueError("Please use either `keep_in_memory` or `cache_file_name` but not both.")

    if num_proc is not None and num_proc <= 0:
        raise ValueError("num_proc must be an integer > 0.")

    if len(self) == 0:
        if self._indices is not None:  # empty indices mapping
            self = Dataset(
                self.data.slice(0, 0),
                info=self.info.copy(),
                split=self.split,
                fingerprint=new_fingerprint,
            )
        if remove_columns:
            return self.remove_columns(remove_columns)
        else:
            return self

    if function is None:
        function = lambda x: x  # noqa: E731

    if isinstance(input_columns, str):
        input_columns = [input_columns]

    if input_columns is not None:
        for input_column in input_columns:
            if input_column not in self._data.column_names:
                raise ValueError(
                    f"Input column {input_column} not in the dataset. "
                    f"Current columns in the dataset: {self._data.column_names}"
                )

    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]

    if remove_columns is not None and any(col not in self._data.column_names for col in remove_columns):
        raise ValueError(
            f"Column to remove {list(filter(lambda col: col not in self._data.column_names, remove_columns))} "
            f"not in the dataset. Current columns in the dataset: {self._data.column_names}"
        )

    load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

    if fn_kwargs is None:
        fn_kwargs = {}

    if num_proc is not None and num_proc > len(self):
        num_proc = len(self)
        logger.warning(
            f"num_proc must be <= {len(self)}. Reducing num_proc to {num_proc} for dataset of size {len(self)}."
        )

    disable_tqdm = not logging.is_progress_bar_enabled()

    if num_proc is None or num_proc == 1:
        return self._map_single(
            function=function,
            with_indices=with_indices,
            with_rank=with_rank,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            new_fingerprint=new_fingerprint,
            disable_tqdm=disable_tqdm,
            desc=desc,
        )
    else:

        # noinspection PyShadowingNames
        def format_cache_file_name(cache_file_name, rank):
            sep = cache_file_name.rindex(".")
            base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
            cache_file_name = base_name + suffix_template.format(rank=rank, num_proc=num_proc) + extension
            logger.info(f"Process #{rank} will write at {cache_file_name}")
            return cache_file_name

        # noinspection PyShadowingNames
        def format_new_fingerprint(new_fingerprint, rank):
            return new_fingerprint + suffix_template.format(rank=rank, num_proc=num_proc)

        prev_env = deepcopy(os.environ)
        if prev_env.get("TOKENIZERS_PARALLELISM", "false").lower() not in (
                "",
                "off",
                "false",
                "f",
                "no",
                "n",
                "0",
        ):
            logger.warning("Setting TOKENIZERS_PARALLELISM=false for forked processes.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        initargs, initializer = None, None
        if not disable_tqdm:
            initargs, initializer = (RLock(),), tqdm.set_lock

        shards = [
            self.shard(num_shards=num_proc, index=rank, contiguous=True, keep_in_memory=keep_in_memory)
            for rank in range(num_proc)
        ]
        kwds_per_shard = [
            dict(
                self=shards[rank],
                function=function,
                with_indices=with_indices,
                with_rank=with_rank,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=format_cache_file_name(cache_file_name, rank)
                if cache_file_name is not None
                else None,
                writer_batch_size=writer_batch_size,
                features=features.copy() if features is not None else None,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                rank=rank,
                offset=sum(len(s) for s in shards[:rank]),
                disable_tqdm=disable_tqdm,
                new_fingerprint=format_new_fingerprint(new_fingerprint, rank)
                if new_fingerprint is not None
                else None,
                desc=desc,
            )
            for rank in range(num_proc)
        ]

        # We search for already cached shards
        def catch_non_existent_error(func, kwargs):
            try:
                return func(**kwargs)
            except NonExistentDatasetError:
                return None

        transformed_shards = [
            catch_non_existent_error(self.__class__._map_single, dict(cache_only=True, **kwds))
            for kwds in kwds_per_shard
        ]

        # We try to create a pool with as many workers as dataset not yet cached.
        nb_of_missing_shards = transformed_shards.count(None)
        if nb_of_missing_shards > 0:
            with Pool(nb_of_missing_shards, initargs=initargs, initializer=initializer) as pool:
                os.environ = prev_env
                logger.info(f"Spawning {num_proc} processes")
                results = {
                    i: pool.apply_async(self.__class__._map_single, kwds=kwds)
                    for i, (kwds, cached_shard) in enumerate(zip(kwds_per_shard, transformed_shards))
                    if cached_shard is None
                }
                assert len(results) == nb_of_missing_shards, \
                    "The number of missing cached shards needs to correspond " \
                    "to the number of `_map_single` we're running"

                for index, async_result in results.items():
                    transformed_shards[index] = async_result.get()

        assert (
                transformed_shards.count(None) == 0
        ), "All shards have to be defined Datasets, none should still be missing."

        logger.info(f"Concatenating {num_proc} shards")
        result = _concatenate_map_style_datasets(transformed_shards)
        if new_fingerprint is not None:
            result._fingerprint = new_fingerprint
        return result


def init_hf_datasets_acceleration() -> bool:

    if hf_datasets.__version__ != "2.9.0":
        raise ImportError(
            "本代码仅在 2.9.0 版本的 HuggingFace Datasets 中进行了测试, 其它版本的请自行修改代码适配 \n"
            "方式很简单, 将 Dataset.map 的相关代码取出, 然后用 multiprocessing 代替 multiprocess 即可"
        )

    Dataset.acc_map = _acc_map

    return True 
