# -*- coding:utf-8 -*-
# Author: lqxu

import os 
import sys 
import logging

sys.path.insert(0, "./")

__all__ = ["raw_data_dir", "output_dir", "log_dir", "data_dir", "model_dir", "prepare_logging"]

raw_data_dir = "./datasets/"

output_dir = "./output"
log_dir = os.path.join(output_dir, "log")
data_dir = os.path.join(output_dir, "data")
model_dir = os.path.join(output_dir, "models")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def prepare_logging(file_path: str) -> logging.Logger:
    if os.path.exists(file_path):
        os.remove(file_path)
    
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", 
        level=logging.DEBUG, 
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(filename=file_path, mode="w")], 
    )

    logger = logging.getLogger("main")

    return logger
