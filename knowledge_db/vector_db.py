# -*- ecoding: utf-8 -*-
# @ModuleName: vector_db
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/5/13 14:42
import faiss
import numpy as np
import json

class VectorDB:
    def __init__(self, emb_model):
        self.text2emb = {}  # 初始化一个空字典，用于存储文本到向量的映射
        self.id2text = []   # 构建faiss检索后的id到text的映射
        self.index = None   # 初始化一个空的Faiss索引
        self.emb_model = emb_model

    def load_emb(self, emb_file):
        """从文件中加载文本到向量的映射"""
        with open(emb_file, 'r') as f:
            self.text2emb = json.load(f)
        # 从存储的向量创建Faiss索引
        emb_dim = len(next(iter(self.text2emb.values())))  # 获取向量的维度
        self.index = faiss.IndexFlatL2(emb_dim)
        for text, emb in self.text2emb.items():
            self.id2text.append(text)
            self.index.add(np.array(emb).reshape(1,-1).astype('float32'))

    def save_emb(self, emb_file):
        """将文本到向量的映射保存到文件中"""
        with open(emb_file, 'w') as f:
            json.dump(self.text2emb, f)

    def add_texts(self, texts, embs):
        """向数据库中添加新的文本和对应的向量"""
        for text, emb in zip(texts, embs):
            self.text2emb[text] = emb.tolist()
            self.id2text.append(text)
            self.index.add(np.array(emb).reshape(1,-1).astype('float32'))

    def query(self, text, topk):
        """查询与给定文本最相似的topk个文本"""
        emb = self.emb_model.get_emb([text]).astype('float32')
        D, I = self.index.search(emb.reshape(1, -1), topk)  # 检索最相似的向量
        return [self.id2text[i] for i in I[0]]
