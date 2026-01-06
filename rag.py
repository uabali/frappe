import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_pdfs(data_dir: str, max_files: int = 5) -> list:
    pdf_files = glob(os.path.join(data_dir, "*.pdf"))[:max_files]
    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())
    return docs


def create_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )


def create_semantic_chunks(docs, embeddings):
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(chunks, embeddings, collection_name: str = "rag_collection"):
    client = QdrantClient(path="./qdrant_db")
    
    try:
        client.get_collection(collection_name)
        client.delete_collection(collection_name)
    except Exception:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    template = """You are a helpful assistant that answers questions based only on the provided context.

Instructions:
- Answer the question using ONLY the information from the context below.
- If the answer cannot be found in the context, respond with: "I couldn't find relevant information in the provided documents."
- Do not make up information or use external knowledge.
- Always respond in Turkish.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    embeddings = create_embeddings()
    
    docs = load_pdfs("data/", max_files=5)
    print(f"Loaded {len(docs)} pages")
    
    chunks = create_semantic_chunks(docs, embeddings)
    print(f"Created {len(chunks)} chunks")
    
    vectorstore = create_vectorstore(chunks, embeddings)
    print("Vectorstore created")
    
    llm = ChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M",
        temperature=0.1,
        base_url="http://localhost:11434"
    )
    
    rag_chain = create_rag_chain(vectorstore, llm)
    
    question = "GAN nedir?"
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()