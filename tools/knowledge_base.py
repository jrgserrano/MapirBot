from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="./models/sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'mps'}, # Or 'cuda' or 'cpu'
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = Chroma(
    persist_directory="./database/chroma_db",
    embedding_function=embeddings
)

@tool
def knowledge_base(query: str) -> str:
    """Search for relevant information in the knowledge base."""

    docs = vector_store.similarity_search(query, k=3)

    if not docs:
        return "No relevant information found."

    return "\n\n".join([d.page_content for d in docs])

@tool
def knowledge_base_update(text: str, source_name: str) -> str:
    """Update the knowledge base with new information."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
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

    vector_store.add_documents(docs)

    return f"Knowledge base updated successfully with {len(docs)} chunks from {source_name}."
    

    
    