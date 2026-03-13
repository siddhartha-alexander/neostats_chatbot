# utils/rag_pipeline.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGPipeline:

    def __init__(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vectordb = Chroma(
                collection_name="neostats",
                embedding_function=self.embedding_model
            )
        except Exception as e:
            raise RuntimeError(f"RAGPipeline init failed: {str(e)}")

    def load_pdf(self, path):
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()

            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(docs)
            self.vectordb.add_documents(chunks)
            return len(chunks)  # useful to confirm how many chunks were added

        except Exception as e:
            raise RuntimeError(f"PDF loading failed: {str(e)}")

    def retrieve(self, query):
        try:
            results = self.vectordb.similarity_search_with_score(query, k=2)

            if not results:
                return None, None

            best_doc, score = results[0]
            return best_doc.page_content, score

        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")