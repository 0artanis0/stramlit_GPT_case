# -*- ecoding: utf-8 -*-
# @ModuleName: openai_embedding
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/5/13 14:41
from typing import List, Dict
import numpy as np
import requests
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from utils import write_json_file
from tqdm import tqdm


class OpenAIEmbedding:
    def __init__(self, config: Dict):
        self.config = config
        self.key = self.config['key']
        self.url = self.config.get('url', 'https://api.openai-proxy.org')

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    def get_emb(self, texts: List[str]) -> np.array:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
        }
        payload = {
            "model": 'text-embedding-ada-002',
            "input": texts
        }
        response = requests.post(self.url,
                                 headers=headers, json=payload, stream=False, timeout=180)
        response = json.loads(response.text)
        # print(texts,response)
        emb_array = []
        for i in range(len(response['data'])):
            emb_array.append(response['data'][i]['embedding'])
        return np.array(emb_array)

    def get_all_text_emb(self, texts: List[str], chuck_size: int = 256):
        text_emb = []
        for i in tqdm(range(0, len(texts), chuck_size)):
            temp_emb = self.get_emb(texts[i:i + chuck_size])
            text_emb.append(temp_emb)
        # 此时text_emb是一个元素都是ndarray的list，这里需要将其转换为新的ndarray
        # 这里的实现方法有两种，一种是concatenate，一种是stack
        text_emb = np.concatenate(text_emb)
        return text_emb

    def save_emb(self, text_list: List[str], text_emb: np.array, emb_file: str):
        text2emb = {}
        for i in range(len(text_list)):
            text2emb[text_list[i]] = text_emb[i].tolist()
        write_json_file(text2emb, emb_file)
