# -*- ecoding: utf-8 -*-
# @ModuleName: utils
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2024/1/28 15:20
import json
import streamlit as st
def save_uploaded_file(uploaded_file):
    """
    Saves the uploaded file to the local filesystem.

    :param uploaded_file: The uploaded file to save.
    :return: The path to the saved file.
    """
    try:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)