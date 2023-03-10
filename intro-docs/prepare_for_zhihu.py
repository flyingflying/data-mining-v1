# -*- coding:utf-8 -*-
# Author: lqxu

import os 

file_dir = "intro-docs"


if __name__ == "__main__":
    for file_name in os.listdir(file_dir):
        if not file_name.endswith(".md") or file_name.endswith(".zhihu.md"):
            continue

        file_base_name = file_name.split(".")[0]
        file_path = os.path.join(file_dir, file_name)
        new_file_path = os.path.join(file_dir, f"{file_base_name}.zhihu.md")

        with open(file_path, "r", encoding="utf-8") as reader:
            with open(new_file_path, "w", encoding="utf-8") as writer:
                for line in reader:
                    if not line.startswith("$"):
                        line = line.replace("$", "$$")
                    writer.write(line)
