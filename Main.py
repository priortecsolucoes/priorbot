from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import time

app = FastAPI()

caminhos = [
    "Manual Sennebogen - Operação - 870M - 870.0.381.pdf"
]

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

class PriorBotQuestion(BaseModel):
    question: str
    user: str
    key: str
    callbackUrl: str

def processa_e_callback(question, user, callbackUrl, requestId):
    try:
        # Carrega e processa os documentos (igual ao original)
        paginas = []
        for caminho in caminhos:
            loader = PyPDFLoader(caminho)
            paginas.extend(loader.load())
        recur_split = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        documents = recur_split.split_documents(paginas)

        diretorio = 'chroma_vectorstore'
            
        #Criação da base de dados de vetores
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(),
            persist_directory=diretorio
        )
        #Leitura da base de dados de vetores
        #vectorstore = Chroma(
        #    embedding_function=OpenAIEmbeddings(),
        #    persist_directory=diretorio
        #)
        prompt = ChatPromptTemplate.from_template(
            '''Responda as perguntas se baseando no contexto fornecido.
            contexto: {contexto}
            pergunta: {pergunta}'''
        )
        retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})

        setup = RunnableParallel({
            'pergunta': RunnablePassthrough(),
            'contexto': retriever
        }) | join_documents
        chain = setup | prompt | ChatOpenAI() | StrOutputParser()
        print(question)
        response = chain.invoke(question)
        print(response)
        # Faz o callback
        payload = {
            "requestId": requestId,
            "user": user,
            "response": response
        }
        cb_resp = requests.post(callbackUrl, json=payload, timeout=15)
        cb_resp.raise_for_status()
        print("Callback enviado com sucesso.")
    except Exception as cb_err:
        print(f"Erro ao processar ou chamar callback: {cb_err}")

@app.post("/ask_question")
async def ask_question(request: PriorBotQuestion, background_tasks: BackgroundTasks):
    try:
        print("ask_question")
        if (request.key != os.getenv('PRIORBOT_KEY')):
            raise HTTPException(status_code=401, detail="Chave incorreta")

        # Gere um requestId único, pode ficar melhor ainda se for passado pelo client
        requestId = f"{request.user}_{int(round(time.time() * 1000))}"

        print("RequestId: " + requestId)

        # PROCESSO EM BACKGROUND!
        background_tasks.add_task(
            processa_e_callback, 
            request.question, 
            request.user, 
            request.callbackUrl,
            requestId
        )

        return JSONResponse(content={
            "message": "Pergunta recebida! Você receberá a resposta em breve.",
            "requestId": requestId
        })

    except HTTPException as http_err:
        print("Erro na API: " + http_err.detail)
        return JSONResponse(status_code=http_err.status_code, content={"message": http_err.detail})
    except Exception as e:
        print("Erro generico na API: " + str(e))
        return JSONResponse(status_code=500, content={"message": f"Erro ao tentar responder pergunta: {str(e)}"})
