from flask import Flask, render_template, jsonify, request

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
from charging_station_predictor import ChargingStationPredictor
import pandas as pd
from joblib import load

# 设置 API 密钥
os.environ["OPENAI_API_KEY"] = ("Your-Api-Key")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=300,

)

chargingdata=pd.read_csv("alt_fuel_stations.csv", low_memory=False)
pricepredictor = load('rf_price_model.joblib')
"""chargingdata['Total_Ports'] = chargingdata['EV Level1 EVSE Num'].fillna(0) + \
                              chargingdata['EV Level2 EVSE Num'].fillna(0) + \
                              chargingdata['EV DC Fast Count'].fillna(0)"""



charging_predictor=ChargingStationPredictor(chargingdata)
charging_predictor.preprocess_data()
charging_predictor.load_models("charging_station_models")


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",

)
vectorstore = FAISS.load_local("vectorstore", embeddings,allow_dangerous_deserialization=True)
retriever1=vectorstore.as_retriever(search_kwargs={"k":4})

template = """You are acting as a helpful assistant of electrical vehicle platform . Use the provided pieces of context to answer the question at the end. 
If contexts are contradictory to your prior knowledge,please follow the provided context.
If you find multiple phrases containing the relevant information, try to combine them when answering.
If the user's question is not related to electrical vehicle topic, you should decline to answer it.
Remember to keep the answer as concise as possible and don't reply too much words. Say "Thank you for asking!" at the end of the response. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory1 = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever1,
    memory=memory1,
    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
    return_source_documents=True
)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

points = [

]

@app.route("/price")
def price():
    return render_template("pricepredict.html")

@app.route('/price/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':



        input_data = pd.DataFrame({
            'Battery': [request.form.get('Battery')],
            'Efficiency': [request.form.get('Efficiency')],
            'Fast_charge': [request.form.get('Fast_charge')],
            'Range': [request.form.get('Range')],
            'Top_speed': [request.form.get('Top_speed')],
            'acceleration_time': [request.form.get('acceleration_time')]
        })



        prediction = pricepredictor.predict(input_data)[0]
        print(prediction)

        return render_template('pricepredict.html', prediction=prediction)



@app.route("/charge")
def charge():
    return render_template('charge.html')



@app.route('/charge/get_coordinates', methods=['POST'])
def get_coordinates():
    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    prediction=charging_predictor.predict(latitude=latitude,
        longitude=longitude,
        access_type='Public',
        )
    print(0)
    return jsonify({'latitude': latitude, 'longitude': longitude,"prediction":prediction})

@app.route('/charge/get_points')
def get_points():
    return jsonify(points)




@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.json.get("message")  # 获取用户的消息
        print(user_input)
        response = qa_chain({"question":user_input})["answer"]
        return jsonify({"response": response})  # 返回 JSON 响应
    return render_template("chat.html")  # 渲染前端页面


if __name__ == '__main__':
    app.run(debug=True)
