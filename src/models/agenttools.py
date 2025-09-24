import os
import requests
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import tool
from typing import Dict, Union, Any
from urllib.parse import urlparse
from sqlalchemy import make_url

_vector_store_cache = {}
_index_cache = {}

def get_vector_store(project_id: str = "default") -> PGVectorStore:
    """Get or create vector store for the project"""
    global _vector_store_cache
    
    if project_id in _vector_store_cache:
        return _vector_store_cache[project_id]
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")

    url = make_url(DATABASE_URL)
    
    # Use the same table naming convention as ProcessController
    table_name = f"document_chunks_project_{project_id}"
    
    vector_store = PGVectorStore.from_params(
        host=url.host,
        port=url.port,
        database=url.database,
        user=url.username,
        password=url.password,
        table_name=table_name,
        embed_dim=1024,  # Match your embedding dimension
        hybrid_search=True,
        text_search_config="english",
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    
    _vector_store_cache[project_id] = vector_store
    return vector_store

def get_index(project_id: str = "default") -> VectorStoreIndex:
    """Get or create index for the project"""
    global _index_cache
    
    if project_id in _index_cache:
        return _index_cache[project_id]
    
    vector_store = get_vector_store(project_id)
    
    # Initialize embedding model
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = OllamaEmbedding(
        model_name="mxbai-embed-large:latest",  
        base_url=ollama_base_url,
        request_timeout=120.0
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    try:
        # Try to load existing index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        print(f"DEBUG: Loaded existing index for project {project_id}")
    except Exception as e:
        print(f"DEBUG: Could not load existing index, creating new one: {e}")
        # Create new empty index
        index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
    
    _index_cache[project_id] = index
    return index

def warm_up_ollama(base_url: str, model_name: str) -> bool:
    """Pre-warm Ollama model to avoid cold start delays"""
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Could not warm up Ollama model: {e}")
        return False

@tool("Document Retrieval Tool")  
def document_retrieval_tool(query: Union[str, Dict[str, Any]], project_id: str = "default") -> str:
    """
    Retrieves relevant context from a collection of policy and standards documents. 
    Use this tool to search for information in policy documents, manuals, and standards.
    
    Args:
        query: The search query (can be string or dict from CrewAI)
        project_id: The project identifier to search within
    """
    try:
        # Debug: Log what actually receive
        print(f"DEBUG: Tool received query parameter: {repr(query)}")
        print(f"DEBUG: Tool received project_id: {project_id}")
        
        # Handle CrewAI's parameter passing - extract actual query from different formats
        search_query = None
        
        if isinstance(query, str):
            search_query = query
        elif isinstance(query, dict):
            # CrewAI passes: {"description": "actual query", "type": "str"}
            if "description" in query:
                search_query = query["description"]
            elif "query" in query:
                search_query = query["query"]
            else:
                # Fallback: convert dict to string
                search_query = str(query)
        else:
            search_query = str(query)
        
        # Validate we have a proper query string
        if not search_query or not isinstance(search_query, str):
            return "Error: No valid search query provided."
        
        # Check if we got a placeholder description instead of real query
        if search_query in ["The search query to find relevant documents", ""]:
            return "Error: Tool received schema placeholder instead of actual query."
        
        search_query = search_query.strip()
        print(f"DEBUG: Extracted search query: {repr(search_query)}")
        
        # Get the index for this project
        index = get_index(project_id)
        
        # Pre-warm the Ollama model
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        warm_up_ollama(ollama_base_url, "mxbai-embed-large:latest")
        
        # Create a query engine with hybrid search mode
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=5,  # Retrieve top 5 most similar chunks
            sparse_top_k=5       # Use sparse search as well
        )
        
        # Query using hybrid search (combines vector + text search)
        response = query_engine.query(search_query)
        retrieved_nodes = response.source_nodes
        
        if not retrieved_nodes:
            return "No relevant documents found for this query."
        
        # Format the retrieved context with source metadata and contextual information
        formatted_chunks = []
        for i, node in enumerate(retrieved_nodes, 1):
            content = node.node.get_content()
            
            # Extract source file information from metadata
            source_info = "Unknown source"
            context_info = ""
            project_info = ""
            
            if hasattr(node.node, 'metadata') and node.node.metadata:
                # File information
                source_file = node.node.metadata.get('source_file', 'Unknown file')
                source_info = f"Source: {source_file}"
                
                # AI-generated contextual information (if available)
                ai_context = node.node.metadata.get('ai_context', '')
                if ai_context:
                    context_info = f"\nContext: {ai_context}"
                
                # Project information
                proj_id = node.node.metadata.get('project_id', '')
                if proj_id:
                    project_info = f" (Project: {proj_id})"
            
            # Include similarity score if available
            score_info = ""
            if hasattr(node, 'score') and node.score is not None:
                score_info = f" [Similarity: {node.score:.3f}]"
            
            formatted_chunk = f"**Document Chunk {i}**\n{source_info}{project_info}{score_info}{context_info}\n\nContent:\n{content}"
            formatted_chunks.append(formatted_chunk)
        
        context = "\n\n" + "="*50 + "\n\n".join(formatted_chunks)
        
        print(f"DEBUG: Retrieved {len(retrieved_nodes)} chunks for query: {search_query}")
        return context
        
    except Exception as e:
        error_msg = f"Error retrieving documents: {str(e)}"
        print(f"DEBUG: {error_msg}")
        return error_msg

@tool("Project Specific Document Search")
def project_document_search(query: str, project_id: str) -> str:
    """
    Search for documents within a specific project.
    
    Args:
        query: The search query
        project_id: The specific project to search within
    """
    return document_retrieval_tool(query, project_id)

# Additional utility functions for CrewAI agents

@tool("Check Database Connection")
def check_database_connection() -> str:
    """Check if the database connection is working and return status"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            return "Error: DATABASE_URL environment variable not set."
        
        url = make_url(DATABASE_URL)
        
        # Try to create a vector store connection
        vector_store = PGVectorStore.from_params(
            host=url.host,
            port=url.port,
            database=url.database,
            user=url.username,
            password=url.password,
            table_name="test_connection",
            embed_dim=1024,
        )
        
        return f"Database connection successful. Connected to {url.host}:{url.port}/{url.database}"
        
    except Exception as e:
        return f"Database connection failed: {str(e)}"

@tool("List Available Projects")  
def list_available_projects() -> str:
    """List all available projects in the database"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            return "Error: DATABASE_URL environment variable not set."
        
        import psycopg2
        url = make_url(DATABASE_URL)
        
        conn = psycopg2.connect(
            host=url.host,
            port=url.port,
            database=url.database,
            user=url.username,
            password=url.password
        )
        
        cursor = conn.cursor()
        
        # Find all tables that match the project pattern
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'document_chunks_project_%'
        """)
        
        tables = cursor.fetchall()
        
        if not tables:
            return "No projects found in database."
        
        projects = []
        for (table_name,) in tables:
            project_id = table_name.replace("document_chunks_project_", "")
            projects.append(project_id)
        
        conn.close()
        
        return f"Available projects: {', '.join(projects)}"
        
    except Exception as e:
        return f"Error listing projects: {str(e)}"