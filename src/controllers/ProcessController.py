from .BaseController import BaseController
from .ProjectController import ProjectController
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import BaseNode
import os
from typing import List, Optional
import psycopg2
import json
from llama_index.vector_stores.postgres import PGVectorStore
import textwrap
from llama_index.embeddings.ollama import OllamaEmbedding
from contextlib import contextmanager
from sqlalchemy import make_url


class ProcessController(BaseController):

    def __init__(self, project_id: str, default_llm_model: str = "qwen3:0.6b-q4_K_M"):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.reader = DoclingReader(export_type=DoclingReader.ExportType.JSON, keep_image=False)
        self.default_llm_model = default_llm_model
        self._llm = None  # Lazy initialization
        self._vector_store = None
        self._index = None

    @property
    def vector_store(self) -> PGVectorStore:
        """Lazy initialization of PGVectorStore"""
        if self._vector_store is None:
            url = make_url(self.app_settings.DB_CONNECTION_STRING)
            self._vector_store = PGVectorStore.from_params(
                database="rag",
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=f"document_chunks_project_{self.project_id}",  # Project-specific table
                embed_dim=1024,
                hybrid_search=True,
                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                    "hnsw_dist_method": "vector_cosine_ops",
                },
            )
        return self._vector_store

    @property
    def index(self) -> VectorStoreIndex:
        """Get or create VectorStoreIndex"""
        if self._index is None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Try to load existing index first
            try:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context
                )
                self.logger.info(f"Loaded existing index for project {self.project_id}")
            except Exception as e:
                self.logger.warning(f"Could not load existing index: {e}")
                # Create new empty index
                self._index = VectorStoreIndex([], storage_context=storage_context)
                self.logger.info(f"Created new index for project {self.project_id}")
                
        return self._index
        """Lazy initialization of LLM"""
        if self._llm is None:
            self._llm = self.initialize_ollama_llm(self.default_llm_model)
        return self._llm

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        db_connection_string = self.app_settings.DB_CONNECTION_STRING
        conn = None
        try:
            conn = psycopg2.connect(db_connection_string)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def load_documents(self, file_id: str) -> List[Document]:
        """
        Load document from file using DoclingReader and return list of Documents.
        """
        file_path = os.path.join(self.project_path, file_id)
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return []

        try:
            documents = self.reader.load_data(file_path)
            # Assign source filename as metadata
            for doc in documents:
                doc.metadata["source_file"] = file_id
            self.logger.info(f"Loaded {len(documents)} documents from '{file_id}'.")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading document {file_id}: {e}")
            return []

    def chunk_documents(self, documents: List[Document]) -> Optional[List[BaseNode]]:
        """
        Convert list of loaded Documents into semantic chunks/nodes with DoclingNodeParser.
        Returns a list of nodes or None on failure.
        """
        if not documents:
            self.logger.warning("No documents provided for chunking.")
            return None

        try:
            node_parser = DoclingNodeParser()
            nodes = node_parser.get_nodes_from_documents(documents)
            for node in nodes:
                # Add project_id metadata for provenance
                node.metadata['project_id'] = self.project_id
            self.logger.info(f"Chunked documents into {len(nodes)} nodes.")
            return nodes
        except Exception as e:
            self.logger.error(f"Error chunking documents: {e}")
            return None

    def initialize_ollama_llm(self, model_name: str) -> Ollama:
        """
        Initialize and return an Ollama LLM instance with the specified model.
        """
        try:
            ollama_llm = Ollama(model=model_name)
            self.logger.info(f"Ollama LLM initialized with model '{model_name}'.")
            return ollama_llm
        except Exception as e:
            self.logger.error(f"Error initializing Ollama LLM with model '{model_name}': {e}")
            raise

    def generate_context_for_chunks(self, nodes: List[BaseNode], llm_model_name: Optional[str] = None) -> List[BaseNode]:
        """Generate context summaries for chunks using LLM"""
        if not nodes:
            self.logger.warning("No nodes provided for context generation.")
            return []
        
        
        
        
        for i, node in enumerate(nodes):
            if i % 10 == 0:
                self.logger.info(f"Generating context for chunk {i+1}/{len(nodes)}")
            
            prompt = f"""Summarize the key context of the following text in two sentences:

{node.get_content()}

Context summary:"""
            
            try:
                response = llm_instance.complete(prompt)
                context_summary = response.text.strip()
                node.metadata['ai_context'] = context_summary
            except Exception as e:
                self.logger.warning(f"LLM context generation failed for chunk {i}: {e}")
                node.metadata['ai_context'] = "No context available"
        
        self.logger.info("Completed generating context for all chunks.")
        return nodes

    def embed_and_store_chunks(self, nodes: List[BaseNode], embedding_model: str = "mxbai-embed-large:latest") -> bool:
        """
        Generate embeddings for chunks using Ollama and store them in pgvector database.
        Returns True if successful, False otherwise.
        """
        if not nodes:
            self.logger.warning("No nodes provided for embedding.")
            return False

        try:
            # Initialize Ollama embedding model
            embed_model = OllamaEmbedding(
                model_name=embedding_model,
                base_url=self.app_settings.OLLAMA_BASE_URL,
            )
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        project_id VARCHAR(255),
                        source_file VARCHAR(255),
                        chunk_text TEXT,
                        ai_context TEXT,
                        metadata JSONB,
                        embedding VECTOR(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Process each node
                for i, node in enumerate(nodes):
                    if i % 10 == 0:
                        self.logger.info(f"Embedding chunk {i+1}/{len(nodes)}")
                    
                    # Generate embedding
                    chunk_text = node.get_content()
                    embedding = embed_model.get_text_embedding(chunk_text)
                    
                    # Extract metadata
                    metadata = node.metadata
                    project_id = metadata.get('project_id')
                    source_file = metadata.get('source_file')
                    ai_context = metadata.get('ai_context', '')
                    
                    # Insert into database
                    cursor.execute("""
                        INSERT INTO document_chunks 
                        (project_id, source_file, chunk_text, ai_context, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        project_id,
                        source_file, 
                        chunk_text,
                        ai_context,
                        json.dumps(metadata),
                        embedding
                    ))
                
                conn.commit()
                self.logger.info(f"Successfully stored {len(nodes)} chunks with embeddings")
                return True
            
        except Exception as e:
            self.logger.error(f"Error embedding and storing chunks: {e}")
            return False

    def get_file_content(self, file_id: str) -> bytes:
        """
        Read file content from the project directory.
        Returns file content as bytes.
        """
        file_path = os.path.join(self.project_path, file_id)
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File {file_id} not found in project {self.project_id}")

        try:
            with open(file_path, 'rb') as file:
                content = file.read()
            self.logger.info(f"Read {len(content)} bytes from file '{file_id}'")
            return content
        except Exception as e:
            self.logger.error(f"Error reading file {file_id}: {e}")
            raise

    def process_file_content(self, file_content: bytes, file_id: str, chunk_size: int, overlap_size: int) -> List[dict]:
        """
        Process file content and return chunks (for direct API response).
        This is different from the embedding pipeline - just returns chunk data.
        """
        try:
            
            documents = self.load_documents(file_id)
            if not documents:
                return []

            # Chunk documents  
            nodes = self.chunk_documents(documents)
            if not nodes:
                return []

            # Convert nodes to dict format for API response
            chunks = []
            for i, node in enumerate(nodes):
                chunks.append({
                    "chunk_id": i,
                    "content": node.get_content(),
                    "metadata": node.metadata,
                    "project_id": self.project_id,
                    "source_file": file_id
                })
            
            self.logger.info(f"Processed file {file_id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing file content: {e}")
            return []

    def process_document_pipeline(self, file_id: str, generate_context: bool = True) -> bool:
        """
        Complete pipeline: load -> chunk -> generate context -> embed & store
        """
        try:
            # Load documents
            documents = self.load_documents(file_id)
            if not documents:
                return False
            
            # Chunk documents
            nodes = self.chunk_documents(documents)
            if not nodes:
                return False
            
            # Generate context if requested
            if generate_context:
                nodes = self.generate_context_for_chunks(nodes)
            
            # Embed and store
            return self.embed_and_store_chunks(nodes)
            
        except Exception as e:
            self.logger.error(f"Error in document processing pipeline for {file_id}: {e}")
            return False