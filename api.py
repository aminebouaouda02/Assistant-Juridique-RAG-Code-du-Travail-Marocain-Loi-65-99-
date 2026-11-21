import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb.utils.embedding_functions as ef

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api")

app = FastAPI(
    title="API Assistant Juridique RAG - Code du Travail Marocain",
    description="API RAG avec LangChain, ChromaDB et Groq pour interroger la Loi 65-99 avec citation de sources.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChromaONNXEmbeddings(Embeddings):
    def __init__(self):
        self.fn = ef.DefaultEmbeddingFunction()
    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self.fn(texts)]
    def embed_query(self, text):
        return [float(x) for x in self.fn([text])[0]]

embeddings = ChromaONNXEmbeddings()
db = Chroma(persist_directory="./chroma_db_code_travail", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    groq_api_key=groq_api_key,
    temperature=0.0
)

system_prompt = """Tu es un assistant juridique expert du Code du travail marocain (Loi 65-99).
Reponds a la question de l'utilisateur en te basant EXCLUSIVEMENT sur les articles juridiques fournis dans le contexte ci-dessous.

Regles obligatoires :
1. CITE FORMELLEMENT le ou les numeros d'articles sources pour chaque affirmation (ex: 'Selon l'Article 14...', '[Source : Article 14]').
2. Si les extraits fournis ne contiennent pas les elements necessaires pour repondre avec certitude a la question, reponds STRICTEMENT : 'Les extraits du Code du travail fournis ne contiennent pas d'informations suffisantes pour repondre a cette question.' N'invente AUCUNE information et n'extrapole pas.
3. Sois precis, clair et neutre.

Contexte juridique :
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def format_docs(docs):
    formatted = []
    for d in docs:
        art = d.metadata.get("article", "Article inconnu")
        formatted.append(f"--- {art} ---\n{d.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API de l'Assistant RAG Juridique (Code du travail marocain).",
        "docs_url": "/docs"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas etre vide.")
    
    docs = retriever.invoke(req.question)
    sources = list(set(d.metadata.get("article", "Article inconnu") for d in docs))
    answer = rag_chain.invoke(req.question)
    
    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
