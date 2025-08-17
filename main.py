import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "example_data", "RAG.pdf")

db_dir = os.path.join(current_dir, "db")

loader = PyPDFLoader(file_path)
documents = loader.load()

chunk_size = 500
chunk_overlap = 50
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={"normalize_embeddings": True}
)

emb = embedding.embed_query("test sentence")
print("Embedding vector length:", len(emb))

def create_vector_store(chunks, embedding, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print("Creating new Chroma vector store...")
        Chroma.from_documents(chunks, embedding, persist_directory=persistent_dir)
    else:
        print("Vector store already exists")
    return persistent_dir

store_name = "chromaDB"
persistent_dir = create_vector_store(chunks, embedding, store_name)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Based on the below information:\n\n{information}\n\nAnswer the question:\n\n{question}")
])

output_parser = StrOutputParser()
model = ChatOpenAI(
    model="gpt-3.5-turbo"
)
chain = prompt | model | output_parser

def get_relevant_data(query, embedding_func, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        db = Chroma(persist_directory=persistent_dir, embedding_function=embedding_func)
        dense_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.3, 0.7]
        )
        return ensemble_retriever.invoke(query)
    else:
        print("No vector store found. Run create_vector_store first.")
        return []

try:
    while True:
        query = input("\nYour question: ")
        relevant_data = get_relevant_data(query, embedding, store_name)
        if not relevant_data:
            print("No relevant data found.")
            continue
        result = chain.invoke({
            "information": "\n".join([doc.page_content for doc in relevant_data]),
            "question": query
        })
        print(f"\nAnswer:\n{result}\n")

except KeyboardInterrupt:
    print("Program finished")
