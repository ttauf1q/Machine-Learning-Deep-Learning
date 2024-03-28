from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss


import os
os.environ["OPENAI_API_KEY"] = "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"

PdfReader = PdfReader('Enter Your PDF name or PATH here')

from typing_extensions import Concatenate

raw_text=''
for i, page in enumerate(PdfReader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
        
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
from langchain_community.vectorstores.faiss import FAISS
document_search = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type='stuff')

query = "Enter Your Question Here "
docs = document_search.similarity_search(query)
chain.run(input_documents = docs, question = query)