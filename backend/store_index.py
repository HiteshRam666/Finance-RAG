import os
from dotenv import load_dotenv
# from backend.src.helper import load_pdf_files, filter_docs, text_split, create_advanced_retriever
from backend.src.helper import load_pdf_files, filter_docs, text_split, create_advanced_retriever
from pinecone import Pinecone 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec

load_dotenv()

openai = os.environ.get("OPENAI_API_KEY")
pinecone = os.environ.get("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = openai 
os.environ["PINECONE_API_KEY"] = pinecone

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data')

# Initialize variables for documents
text_chunks = []

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

pc = Pinecone(api_key=pinecone)
index_name = "finance-bot"

# Create index
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name, 
        dimension=3072,  # text-embedding-3-large dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


docsearch = PineconeVectorStore.from_existing_index(
    index_name= index_name, 
    embedding=embedding_model
)

# Check if index is empty before storing documents
try:
    # Get index stats to check if it's empty
    index_stats = pc.describe_index(index_name)
    index_info = pc.Index(index_name)
    stats = index_info.describe_index_stats()
    
    # Only store documents if index is empty
    if stats.total_vector_count == 0:
        print(f"Index '{index_name}' is empty. Loading and processing documents...")
        # Load and process documents only when needed
        extracted_data = load_pdf_files(data=data_path)
        filtered_data = filter_docs(extracted_data)
        text_chunks = text_split(filtered_data, chunk_size=800, chunk_overlap=100)
        
        print(f"Storing {len(text_chunks)} document chunks...")
        docsearch.add_documents(text_chunks)
        print(f"Successfully stored {len(text_chunks)} document chunks in Pinecone index '{index_name}'")
    else:
        print(f"Index '{index_name}' already contains {stats.total_vector_count} vectors. Skipping document storage.")
        # Still need to load documents for BM25 index
        extracted_data = load_pdf_files(data=data_path)
        filtered_data = filter_docs(extracted_data)
        text_chunks = text_split(filtered_data, chunk_size=800, chunk_overlap=100)
        
except Exception as e:
    print(f"Error checking/storing documents: {e}")
    # Fallback: try to store documents anyway
    try:
        print("Fallback: Loading and processing documents...")
        extracted_data = load_pdf_files(data=data_path)
        filtered_data = filter_docs(extracted_data)
        text_chunks = text_split(filtered_data, chunk_size=800, chunk_overlap=100)
        
        docsearch.add_documents(text_chunks)
        print(f"Successfully stored {len(text_chunks)} document chunks in Pinecone index '{index_name}'")
    except Exception as store_error:
        print(f"Error storing documents: {store_error}")
        # Load documents for BM25 even if storage fails
        extracted_data = load_pdf_files(data=data_path)
        filtered_data = filter_docs(extracted_data)
        text_chunks = text_split(filtered_data, chunk_size=800, chunk_overlap=100)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {'k': 3})

# Create Retriever
advanced_retriever = create_advanced_retriever(docsearch, embedding_model, text_chunks)

retriever = advanced_retriever