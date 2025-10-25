from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings 
from typing import List
from langchain.schema import Document 
import numpy as np 
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# Extract text from pdf 
def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )

    documents = loader.load() 
    return documents

def filter_docs(docs: List[Document]) -> List[Document]:
    """
    Enhanced document filtering with better metadata preservation.
    Given a list of document objects, Return a new list of document objects
    containing only 'source' in metadata and the original page content
    """
    filtered_docs: List[Document] = [] 
    for doc in docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page", 0)

        # Clean and enhance content 
        cleaned_content = clean_text(doc.page_content)

        filtered_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata = {
                    "source": src, 
                    "page": page, 
                    "content_length": len(cleaned_content), 
                    "word_count": len(cleaned_content.split())
                    }
            )
        )
    return filtered_docs

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()

# Creating chunks 
def text_split(filtered_docs, chunk_size=800, chunk_overlap=100):
    """
    Enhanced text splitting with semantic chunking
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    text_chunks = text_splitter.split_documents(filtered_docs)
    # Add chunk metadata
    for i, chunk in enumerate(text_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "chunk_length": len(chunk.page_content),
            "chunk_words": len(chunk.page_content.split())
        })
    
    return text_chunks

class HybridRetrievar:
    """
    Advanced hybrid retrieval combining semantic search, BM25, and reranking
    """
    def __init__(self, vector_store, embedding_model, rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_store = vector_store 
        self.embedding_model = embedding_model 
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize BM25
        self.bm25 = None 
        self.documents = [] 

        # Initialize Reranked 
        try:
            from sentence_transformers import CrossEncoder 
            self.reranker = CrossEncoder(rerank_model_name)
        except:
            self.reranker = None 
            print("Warning: Crossencoder not available")

    def build_bm25_index(self, documents: List[Document]):
        """Build BM25 index from documents"""
        self.documents = documents 
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def semantic_search(self, query: str, k: int = 10) -> List[Document]:
        """Semantic similarity search"""
        retriever = self.vector_store.as_retriever(
            search_type = "similarity", 
            search_kwargs = {"k": k}
            )
        return retriever.invoke(query)
    
    def bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """BM25 keyword-based search"""
        if not self.bm25:
            return [] 
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top k documents 
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Document]:
        """
        Hybrid search combining semantic and BM25 with weighted scoring
        """
        # Get semantic results 
        semantic_docs = self.semantic_search(query, k*2)
        semantic_scores = {doc.page_content: 1.0 - (i / len(semantic_docs)) for i, doc in enumerate(semantic_docs)}

        # Get BM25 Results 
        bm25_docs = self.bm25_search(query, k * 2)
        bm25_scores = {doc.page_content: 1.0 - (i / len(bm25_docs)) for i, doc in enumerate(bm25_docs)}

        # Combine Scores - deduplicate by content since Document objects are not hashable
        all_docs = semantic_docs + bm25_docs
        # Remove duplicates by content
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        combined_scores = {} 
        for doc in unique_docs:
            semantic_score = semantic_scores.get(doc.page_content, 0)
            bm25_score = bm25_scores.get(doc.page_content, 0)
            combined_scores[doc.page_content] = alpha * semantic_score + (1 - alpha) * bm25_score
        
        # Sort by combined scores 
        sorted_docs = sorted(unique_docs, key = lambda x: combined_scores[x.page_content], reverse = True)
        return sorted_docs[:k]
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents using cross-encoder for better relevance
        """
        if not self.reranker or not documents:
            return documents[:top_k]

        # Prepare query-document pairs for reranking
        pairs = [(query, doc.page_content) for doc in documents]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Sort documents by rerank scores 
        doc_scores = list(zip(documents, rerank_scores))
        doc_scores.sort(key = lambda x: x[1], reverse = True)

        return [doc for doc, score in doc_scores[:top_k]]
    
    def retrieve(self, query: str, k: int = 5, use_reranking: bool = True) -> List[Document]:
        """
        Main retrieval method with hybrid search and optional reranking
        """
        # Get more documents initially for better reranking
        initial_k = k * 3 if use_reranking else k 

        # Perform hybrid search
        documents = self.hybrid_search(query, initial_k)

        # Apply Reranking if enabled
        if use_reranking and len(documents) > k:
            documents = self.rerank_documents(query, documents, k)
        
        return documents[:k]

def create_advanced_retriever(vector_store, embedding_model, documents):
    """Create an advanced hybrid retriever"""
    retriever = HybridRetrievar(vector_store, embedding_model)
    retriever.build_bm25_index(documents)
    return retriever