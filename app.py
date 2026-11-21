import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb.utils.embedding_functions as ef

st.set_page_config(
    page_title="Assistant RAG - Code du Travail Marocain",
    page_icon="⚖️",
    layout="wide"
)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api")

class ChromaONNXEmbeddings(Embeddings):
    def __init__(self):
        self.fn = ef.DefaultEmbeddingFunction()
    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self.fn(texts)]
    def embed_query(self, text):
        return [float(x) for x in self.fn([text])[0]]

@st.cache_resource
def get_rag_chain():
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

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

chain, retriever = get_rag_chain()

st.title("⚖️ Assistant Juridique RAG — Code du Travail Marocain (Loi 65-99)")
st.markdown("""
Cet assistant documentaire utilise **LangChain**, **ChromaDB** et **Groq** pour répondre à vos questions avec citation obligatoire des articles sources.
""")

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input(
        "Posez votre question juridique :",
        placeholder="Ex: Quelle est la durée de la période d'essai pour un CDI des cadres ?"
    )
    if st.button("Poser la question", type="primary") and user_query:
        with st.spinner("Recherche des articles et génération de la réponse..."):
            retrieved_docs = retriever.invoke(user_query)
            answer = chain.invoke(user_query)

        st.subheader("💡 Réponse de l'Assistant :")
        st.write(answer)

        with st.expander("📚 Voir les extraits d    'articles sources récupérés :"):
            for d in retrieved_docs:
                st.markdown(f"**{d.metadata.get('article')}** (Page {d.metadata.get('page')})")
                st.text(d.page_content)
                st.divider()

with col2:
    st.subheader("📌 Exemples de test")
    examples = [
        "Quelle est la durée de la période d'essai pour un CDI des cadres ?",
        "Quelles sont les fautes graves commises par l'employeur envers le salarié ?",
        "Quelle est la durée légale normale de travail par an et par semaine ?",
        "Quel est le taux de l'impôt sur les sociétés ? (Test Hors-domaine)"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            with st.spinner("Traitement..."):
                retrieved_docs = retriever.invoke(ex)
                ans = chain.invoke(ex)
            st.markdown(f"**Question :** {ex}")
            st.markdown(f"**Réponse :**\n{ans}")
