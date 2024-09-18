# -*- ecoding: utf-8 -*-
# @ModuleName: chat_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2024/1/29 17:04
from typing import Dict
from .openai_model import OpenAI
from knowledge_db import VectorDB


class ChatModel:
    def __init__(self, config: Dict, vector_db: VectorDB):
        self.config = config
        if self.config['model_type'] == 'openai':
            self.model = OpenAI(config, vector_db)
        else:
            raise Exception(f"model type:{self.config['model_type']} has not been implemented")

    def ask(self, question):
        return self.model.ask(question)
