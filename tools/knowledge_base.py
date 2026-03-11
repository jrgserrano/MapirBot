from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

#embeddings = HuggingFaceEmbeddings(
#    model_name="./models/sentence-transformers/all-MiniLM-L6-v2",
#    model_kwargs={'device': 'mps'}, # Or 'cuda' or 'cpu'
#    encode_kwargs={'normalize_embeddings': True}
#)

from datetime import datetime, timezone
from agent.graphiti_client import get_graphiti
from graphiti_core.nodes import EpisodeType

_embeddings = None
_vector_store = None

def get_vector_store():
    global _embeddings, _vector_store
    if _vector_store is None:
        print("[INFO] Initializing ChromaDB vector store lazily...")
        from langchain_ollama import OllamaEmbeddings
        from langchain_chroma import Chroma
        
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        _vector_store = Chroma(
            persist_directory="./database/chroma_db",
            embedding_function=_embeddings
        )
    return _vector_store

@tool
async def knowledge_base(query: str) -> str:
    """Search for relevant information in the knowledge base."""

    # 1. Chroma Search
    vs = get_vector_store()
    docs = vs.similarity_search(query, k=3)
    chroma_info = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant information found in vector storage."

    # 2. Graphiti Search
    graphiti = get_graphiti()
    graph_results = await graphiti.search(query)
    graph_info = "\n".join([f"- {edge.fact}" for edge in graph_results]) if graph_results else "No relevant relationships found in graph."

    print(f"[INFO]: Knowledge Base Search Results (Chroma): {len(docs)} docs")
    print(f"[INFO]: Knowledge Base Search Results (Graphiti): {len(graph_results)} facts")

    combined_info = f"--- VECTOR STORAGE ---\n{chroma_info}\n\n--- KNOWLEDGE GRAPH ---\n{graph_info}"
    return combined_info

@tool
async def knowledge_base_update(text: str, source_name: str) -> str:
    """Update the knowledge base with new information from a document/file."""

    # 1. Update Chroma (Vector Store for files)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = text_splitter.split_text(text)

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": source_name}
        )
        for chunk in chunks
    ]

    vs = get_vector_store()
    vs.add_documents(docs)

    return f"RAG updated successfully. {len(docs)} chunks added from {source_name}."
    

    
    