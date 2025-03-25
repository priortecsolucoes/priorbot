from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

caminhos = [
    "Manual_de_instruções_Saveiro_24A.5B1.SAV.66_LOW (1).pdf"
]

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

class PriorBotQuestion(BaseModel):
    question: str
    user: str
    key: str

@app.post("/ask_question")
def ask_question(request: PriorBotQuestion):
    try:
        if (request.key != os.getenv('PRIORBOT_KEY')):
            raise HTTPException(status_code=401, detail="Chave incorreta")         
        
        #Carregamento e dividindo documentos
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
        #vectorstore = FAISS.from_documents(
        #    documents=documents,
        #    embedding=OpenAIEmbeddings()
        #)
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=diretorio
        )
        
        #Criando Estrutura de Conversa
        prompt = ChatPromptTemplate.from_template(
            '''Responda as perguntas se baseando no contexto fornecido.
            contexto: {contexto}
            pergunta: {pergunta}'''
        )
        
        #Configurando o Retriever
        retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})
               
        #Juntando os Documentos
        setup = RunnableParallel({
            'pergunta': RunnablePassthrough(),
            'contexto': retriever
        }) | join_documents
        
        
        chain = setup | prompt | ChatOpenAI() | StrOutputParser()

        #print(request.question)
        response = chain.invoke(request.question)
        #print(response)
        
        return JSONResponse(content={
            "message": "Resposta gerada com sucesso",
            "response": response
        })
        return response
    except HTTPException as http_err:
        return JSONResponse(status_code=http_err.status_code, content={"message": http_err.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Erro ao tentar responder pergunta: {str(e)}"})
