from .BaseController import BaseController
from .ProjectController import ProjectController
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import BaseNode
import os
from typing import List, Optional
import psycopg2
import json
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
import textwrap
from llama_index.embeddings.ollama import OllamaEmbedding
from contextlib import contextmanager


class ProcessController(BaseController):

    def __init__(self, project_id: str, default_llm_model: str = "qwen3:0.6b-q4_K_M"):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.reader = DoclingReader(export_type=DoclingReader.ExportType.JSON, keep_image=False)
        self.default_llm_model = default_llm_model
        self._llm = None  # Lazy initialization

    @property
    def llm(self) -> Ollama:
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
        
        # Use specified model or default
        llm_instance = self.llm if llm_model_name is None else self.initialize_ollama_llm(llm_model_name)
        
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