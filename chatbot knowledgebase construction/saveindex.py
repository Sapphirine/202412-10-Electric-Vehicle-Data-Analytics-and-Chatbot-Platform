from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.chat_models import BedrockChat
from langchain_community.llms import Bedrock
from langchain.evaluation import load_evaluator
from langchain.document_loaders import BSHTMLLoader
import os
from langchain.llms import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
base = []
csv_list=["sampled_charging_stations.csv","ev_charging_patterns.csv","EV_cars.csv","Grouped_Electric_Vehicle_Data.csv"]
for file in csv_list:
    loader=CSVLoader(file,encoding="utf-8")
    base+=loader.load()
os.environ["OPENAI_API_KEY"] = ("Your-Api-Key")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=300,

)


# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",

)
filelist1 = ['Policy Explorer 2024.pdf']

doc1 = []

for f in filelist1:
    loader = PyPDFLoader(f)
    doc1.append(loader.load())

text_splitter1 = RecursiveCharacterTextSplitter(  # Create a textsplitter
    separators=["\n\n", "\n", "    ","     "],
    chunk_size=600,  #
    chunk_overlap=30
)


for each in doc1:
    base += text_splitter1.split_documents(each)
print(len(base))





index_creator = VectorstoreIndexCreator(

    embedding=embeddings,
    text_splitter=CharacterTextSplitter(
        separator="dsjdjsjs",
        chunk_size=9000,
        chunk_overlap=100,
        length_function=len

    ),
    vectorstore_cls=FAISS
)


index = index_creator.from_documents(base)

# 假设 index.vectorstore 是 FAISS 实例
index.vectorstore.save_local("vectotstore")

retriever1=index.vectorstore.as_retriever(search_kwargs={"k":3})





template = """You are acting as a helpful assistant of electrical vehicle platform . Use the provided pieces of context to answer the question at the end. 
If contexts are contradictory to your prior knowledge,please follow the provided context.
If you find multiple phrases containing the relevant information, try to combine them when answering.
If the user's question is not related to electrical vehicle topic, you can decline to answer it.
Remember to keep the answer as concise as possible and don't reply too much words. Say "Thank you for asking!" at the end of the response. 
{context}
Question: {question}
Helpful Answer:"""


"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory1 = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever1,
    memory=memory1,
    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
    return_source_documents=True
)
"""