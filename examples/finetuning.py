# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import sys
import os

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")
sys.path.append(current_path)

from train_llm import main

if __name__ == "__main__":
    main()
