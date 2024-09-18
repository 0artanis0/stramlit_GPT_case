# -*- ecoding: utf-8 -*-
# @ModuleName: emb_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2024/1/29 17:25
from typing import List
import numpy as np
from .openai_emb import OpenAIEmbedding


class EmbModel:
    def __init__(self, config):
        self.config = config
        if self.config['model_type'] == "openai":
            self.model = OpenAIEmbedding(self.config)
        else:
            raise Exception(f"model type:{self.config['model_type']} has not been implemented")

    def get_emb(self, texts: List[str]) -> np.array:
        return self.model.get_emb(texts)

    def get_all_text_emb(self, texts: List[str], chuck_size: int = 256):
        return self.model.get_all_text_emb(texts, chuck_size=chuck_size)

    def save_emb(self, text_list: List[str], text_emb: np.array, emb_file: str):
        self.model.save_emb(text_list, text_emb, emb_file)
