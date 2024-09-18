# -*- ecoding: utf-8 -*-
# @ModuleName: pdf_processor
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2024/1/28 15:11
import PyPDF2

def extract_text_by_paragraph(pdf_path):
    """
    Extracts text from a PDF file and splits it into paragraphs.

    :param pdf_path: The path to the PDF file.
    :return: A list of text paragraphs.
    """
    paragraphs = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                # 分割段落，这里使用两个换行符作为段落分隔符
                # 可根据PDF的具体格式进行调整
                page_paragraphs = text.split('\n\n')
                paragraphs.extend(page_paragraphs)

    return paragraphs

