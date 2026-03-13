from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

# Split text into chunks for embeddings
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    return text_splitter.split_documents(docs)

# Generate embeddings
def get_embeddings(doc_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [model.encode(chunk.page_content) for chunk in doc_chunks]
    return embeddings