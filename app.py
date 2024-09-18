# -*- ecoding: utf-8 -*-
# @ModuleName: app
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2024/1/28 15:09
import streamlit as st
from chat_model import ChatModel
from knowledge_db import EmbModel, VectorDB
from pdf_processor import extract_text_by_paragraph
from utils import save_uploaded_file
import os

st.title("王门智能问答小助手-内测版")

# 对话session初始化
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "你好啊👋我是你的智能问答小助手，我会根据您上传的文档来回答您的问题，希望可以帮助您"}]
# chat组件设置
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 侧边栏
with st.sidebar:
    # Embedding model选择
    embedding_model = st.selectbox(
        "Select Embedding Model",
        ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'],
    )
    # chat model选择
    chat_model = st.selectbox(
        "Select Chat Model",
        ['gpt-3.5-turbo-16k', "gpt-3.5-turbo", 'gpt-3.5-turbo-1106', "gpt-4", 'gpt-4-0125-preview',
         'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'gpt-4o-2024-08-06 '],
    )
    # openai_key设置
    apikey = st.text_input("输入你的OpenAI API密钥", type="password")
    apikey = 'sk-NqzRPn167jVENtffBIWrZwXZ7ALXu347511FkEDVtrDJuXDR'

    # 文件上传器
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # 处理上传的PDF知识库
    if uploaded_file is not None:
        # 保存文件
        saved_file = save_uploaded_file(uploaded_file)
        if saved_file:
            # 提取PDF文件内容，目前只支持PDF文件
            paragraphs = extract_text_by_paragraph(saved_file)
            if paragraphs:
                # 抽取Embedding
                emb_model = EmbModel({'model_type': 'openai', 'key': apikey})
                text_emb = emb_model.get_all_text_emb(paragraphs)
                emb_model.save_emb(paragraphs, text_emb, './db/temp.json')
                st.success("Embeddings extracted successfully！")
            else:
                st.write("No text extracted from the PDF.")
            # 删除临时保存的文件
            if os.path.exists(saved_file):
                os.remove(saved_file)
    else:
        st.stop()

emb_model = EmbModel({'model_type': 'openai', 'key': apikey})
vector_db = VectorDB(emb_model)
vector_db.load_emb('./db/temp.json')
model = ChatModel({'openai-key': apikey, 'model': chat_model, 'model_type': 'openai'}, vector_db)

if prompt := st.chat_input():
    # 对于user侧，先把用户输入内容写到session里面，然后写到对话框里面
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # 调用model获取大模型答复
    response = model.ask(prompt)
    # 对大模型侧，先把模型输出内容写到session里面，然后写到对话框里面
    st.session_state.messages.append({'role': 'assistant', 'content': response['response']})
    st.chat_message("assistant").write(response['response'])
