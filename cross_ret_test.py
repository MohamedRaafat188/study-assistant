import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from typing import Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereRerank


# callback_handler = CallbackHandler()

load_dotenv()

# Set Hugging Face token for authenticated requests
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

def get_llm(
        llm : Literal["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                      "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct",
                      "moonshotai/kimi-k2-instruct-0905"
                      ] = "llama-3.3-70b-versatile",
        temperature : float = 0.7,
        api_key=None
):
    return ChatGroq(model=llm, temperature=temperature, api_key=api_key )

def get_llm_ollama(
        llm: str = "qwen3.5:9b",
        temperature: float = 0.7,
):
    return ChatOllama(model=llm, temperature=temperature).bind(parallel_tool_calls=False)


def load_vectorstore(path, embeddings) -> Chroma:
    return Chroma(collection_name="GeneralDomainKnowledge", embedding_function=embeddings, persist_directory=path)



cohere_reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=3,
    cohere_api_key=os.getenv("COHERE_API_KEY")  # add this to your .env
)

CHROMA_PATH = r"temp_chroma\temp_chroma"
Embedding_Model = "BAAI/bge-large-en-v1.5" 
reranker_model = "BAAI/bge-reranker-v2-m3"
embeddings = HuggingFaceEmbeddings(model_name=Embedding_Model)
reranker = HuggingFaceCrossEncoder(model_name=reranker_model)


retrieval_llm = get_llm_ollama(temperature=0.7)
vector_store = load_vectorstore(CHROMA_PATH, embeddings)
retriever_store = vector_store.as_retriever(search_kwargs={"k": 5})

retreiver_answers = retriever_store.invoke("SAR Definition?")
print("Retriever Answers: ")
for idx, ans in enumerate(retreiver_answers):
    print(f"{idx+1}. {ans.page_content}\n")


cross_encoder = CrossEncoderReranker(model=reranker, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever_store,
    base_compressor=cohere_reranker
)


reranked_docs = compression_retriever.invoke("What is backscatter anomalies?")
print("Compression Retriever Answer: ")
for idx, doc in enumerate(reranked_docs):
    print(f"{idx+1}. {doc.page_content}\n")


# --- Modular Pipeline Functions ---

def build_vectorstore(chroma_path: str, collection_name: str, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=chroma_path
    )


def build_base_retriever(vectorstore: Chroma, k: int = 10):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_cohere_reranker(top_n: int = 3, model: str = "rerank-english-v3.0") -> CohereRerank:
    return CohereRerank(
        model=model,
        top_n=top_n,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )


def build_cross_encoder_reranker(cross_encoder_model: HuggingFaceCrossEncoder, top_n: int = 3) -> CrossEncoderReranker:
    return CrossEncoderReranker(model=cross_encoder_model, top_n=top_n)


def build_compression_retriever(base_retriever, reranker) -> ContextualCompressionRetriever:
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )


def retrieve(
    query: str,
    chroma_path: str,
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    reranker_type: Literal["cohere", "cross_encoder"] = "cohere",
    k: int = 10,
    top_n: int = 3,
    cross_encoder_model: HuggingFaceCrossEncoder = None,
) -> list:
    vs = build_vectorstore(chroma_path, collection_name, embedding_model)
    base_retriever = build_base_retriever(vs, k=k)

    if reranker_type == "cohere":
        reranker = build_cohere_reranker(top_n=top_n)
    else:
        reranker = build_cross_encoder_reranker(cross_encoder_model, top_n=top_n)

    pipeline = build_compression_retriever(base_retriever, reranker)
    return pipeline.invoke(query)