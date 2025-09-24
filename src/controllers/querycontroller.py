from .BaseController import BaseController
from .ProcessController import ProcessController
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from typing import List, Dict, Any
import json

class QueryController(BaseController):
    
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.process_controller = ProcessController(project_id=project_id)
        self._query_engine = None
        
    @property
    def query_engine(self):
        """Lazy initialization of query engine"""
        if self._query_engine is None:
            # Initialize LLM first
            llm = Ollama(
                model=self.process_controller.default_llm_model,
                base_url=self.app_settings.OLLAMA_BASE_URL
            )
            
            # Get the index from ProcessController
            index = self.process_controller.index
            
            # Configure retriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5,  # Retrieve top 5 most similar chunks
            )
            
            # Configure response synthesizer with Ollama
            response_synthesizer = get_response_synthesizer(
                llm=llm,
                response_mode="compact",  # Combine retrieved chunks efficiently
            )
            
            # Add post-processor to filter by similarity threshold
            node_postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=0.7)  # Only include relevant chunks
            ]
            
            # Create query engine
            self._query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=node_postprocessors,
            )
            
            self.logger.info(f"Query engine initialized for project {self.project_id}")
        
        return self._query_engine
    
    def query(self, query_text: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the document collection and return response with citations.
        
        Args:
            query_text: The user's question
            include_sources: Whether to include source information in response
            
        Returns:
            Dict containing response, sources, and metadata
        """
        try:
            # Execute the query
            response = self.query_engine.query(query_text)
            
            # Extract response text
            response_text = str(response)
            
            # Extract source information if requested
            sources = []
            if include_sources and response.source_nodes:
                for node in response.source_nodes:
                    source_info = {
                        "content": node.node.get_content()[:200] + "...",  # First 200 chars
                        "score": node.score if hasattr(node, 'score') else None,
                        "metadata": node.node.metadata,
                        "source_file": node.node.metadata.get('source_file', 'Unknown'),
                        "project_id": node.node.metadata.get('project_id', self.project_id)
                    }
                    sources.append(source_info)
            
            result = {
                "response": response_text,
                "sources": sources,
                "query": query_text,
                "project_id": self.project_id,
                "num_sources": len(sources)
            }
            
            self.logger.info(f"Query processed: '{query_text}' -> {len(sources)} sources found")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query '{query_text}': {e}")
            return {
                "response": "I apologize, but I encountered an error processing your query.",
                "sources": [],
                "query": query_text,
                "project_id": self.project_id,
                "error": str(e)
            }
    
    def semantic_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search without response generation.
        Useful for debugging or getting raw similar chunks.
        
        Args:
            query_text: Search query
            top_k: Number of similar chunks to retrieve
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Get retriever
            retriever = VectorIndexRetriever(
                index=self.process_controller.index,
                similarity_top_k=top_k,
            )
            
            # Retrieve similar nodes
            nodes = retriever.retrieve(query_text)
            
            # Format results
            results = []
            for node in nodes:
                result = {
                    "content": node.node.get_content(),
                    "score": node.score if hasattr(node, 'score') else None,
                    "metadata": node.node.metadata,
                    "source_file": node.node.metadata.get('source_file', 'Unknown'),
                }
                results.append(result)
            
            self.logger.info(f"Semantic search: '{query_text}' -> {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search '{query_text}': {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            vector_store = self.process_controller.vector_store
            
            # This will depend on your vector store implementation
            # For now, return basic info
            stats = {
                "project_id": self.project_id,
                "table_name": vector_store.table_name if hasattr(vector_store, 'table_name') else 'Unknown',
                "embedding_dimension": 1024,
                "status": "active"
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}