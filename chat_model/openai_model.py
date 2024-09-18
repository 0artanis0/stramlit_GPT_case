# -*- ecoding: utf-8 -*-
# @ModuleName: openai_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/5/13 14:41
import openai
from typing import Dict
from tenacity import retry, stop_after_attempt, wait_fixed
from knowledge_db import VectorDB


class OpenAI:
    def __init__(self, config: Dict, vector_db: VectorDB):
        self.config = config
        self.vector_db = vector_db

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    def ask(self, question):
        res = self.vector_db.query(question, topk=2)
        messages = [
            {"role": "user",
             "content": f"下面请你根据我提供的参考资料来回答问题，注意，这里只允许你根据参考资料来回答，如果参考资料提供的内容无法回答问题，则回复不知道。已知内容:{res} \n 我的问题:{question} "}
        ]

        response = openai.ChatCompletion.create(model=self.config.get('model', "gpt-3.5-turbo"),
                                                messages=messages)
        msg = response.choices[0].message.content
        return {
            'response': msg,
            'knowledge': res
        }
