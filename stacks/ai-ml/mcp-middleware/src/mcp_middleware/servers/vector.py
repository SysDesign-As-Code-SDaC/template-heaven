"""
Vector Search Server for MCP Middleware.

This server provides AI assistants with vector database capabilities for
semantic search, similarity matching, and embedding operations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class VectorSearchServer(BaseMCPServer):
    """
    MCP server for vector search operations.

    Provides tools for vector database operations, embedding generation,
    similarity search, and vector indexing.
    """

    SUPPORTED_BACKENDS = {
        "chromadb": {
            "description": "ChromaDB - Open-source embedding database",
            "requires": ["chromadb"]
        },
        "pinecone": {
            "description": "Pinecone - Managed vector database",
            "requires": ["pinecone-client"]
        },
        "weaviate": {
            "description": "Weaviate - Vector search engine with ML capabilities",
            "requires": ["weaviate-client"]
        },
        "qdrant": {
            "description": "Qdrant - Vector similarity search engine",
            "requires": ["qdrant-client"]
        },
        "milvus": {
            "description": "Milvus - Cloud-native vector database",
            "requires": ["pymilvus"]
        },
        "in_memory": {
            "description": "In-memory vector storage (for testing)",
            "requires": []
        }
    }

    def __init__(self, name: str, config: Dict[str, Any], auth: Optional[Dict[str, Any]] = None):
        super().__init__(name, config, auth)
        self.backend = None
        self.client = None
        self.collections: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize the vector search server."""
        try:
            # Validate configuration
            self._validate_config()

            # Initialize vector backend
            backend_type = self.config.get("backend", "in_memory")
            await self._initialize_backend(backend_type)

            logger.info(f"Vector search server '{self.name}' initialized with backend '{backend_type}'")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vector search server '{self.name}': {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the vector search server."""
        try:
            if self.client:
                # Close connections based on backend type
                if hasattr(self.client, 'close'):
                    await self.client.close()
                elif hasattr(self.client, 'delete_collection'):
                    # ChromaDB cleanup
                    pass
                self.client = None

            self.collections.clear()
            logger.info(f"Vector search server '{self.name}' shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down vector search server '{self.name}': {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            backend_type = self.config.get("backend", "in_memory")

            health_status = {
                "status": "healthy" if self.client else "unhealthy",
                "backend": backend_type,
                "collections_count": len(self.collections),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Test basic connectivity
            if self.client:
                try:
                    await self._test_backend_connectivity()
                    health_status["connectivity"] = "ok"
                except Exception as e:
                    health_status["connectivity"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed for vector search server '{self.name}': {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available vector search tools."""
        return [
            {
                "name": "create_collection",
                "description": "Create a new vector collection/index",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to create"
                        },
                        "dimension": {
                            "type": "integer",
                            "description": "Vector dimension",
                            "minimum": 1,
                            "maximum": 4096
                        },
                        "distance_metric": {
                            "type": "string",
                            "enum": ["cosine", "euclidean", "dot_product"],
                            "description": "Distance metric for similarity",
                            "default": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional collection metadata"
                        }
                    },
                    "required": ["collection_name", "dimension"]
                }
            },
            {
                "name": "add_vectors",
                "description": "Add vectors with metadata to a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Target collection name"
                        },
                        "vectors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "vector": {"type": "array", "items": {"type": "number"}},
                                    "metadata": {"type": "object"}
                                },
                                "required": ["id", "vector"]
                            },
                            "description": "List of vectors to add"
                        }
                    },
                    "required": ["collection_name", "vectors"]
                }
            },
            {
                "name": "search_vectors",
                "description": "Search for similar vectors in a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Collection to search in"
                        },
                        "query_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query vector"
                        },
                        "query_text": {
                            "type": "string",
                            "description": "Query text (will be embedded if embedder configured)"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10
                        },
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Include metadata in results",
                            "default": true
                        },
                        "filter": {
                            "type": "object",
                            "description": "Metadata filter for search"
                        }
                    },
                    "required": ["collection_name"]
                }
            },
            {
                "name": "generate_embedding",
                "description": "Generate embeddings for text using configured embedder",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to embed"
                        },
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to embed"
                        }
                    }
                }
            },
            {
                "name": "delete_vectors",
                "description": "Delete vectors from a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "vector_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of vectors to delete"
                        }
                    },
                    "required": ["collection_name", "vector_ids"]
                }
            },
            {
                "name": "list_collections",
                "description": "List all collections in the vector database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "detailed": {
                            "type": "boolean",
                            "description": "Include detailed information",
                            "default": false
                        }
                    }
                }
            },
            {
                "name": "get_collection_info",
                "description": "Get information about a specific collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Collection name"
                        }
                    },
                    "required": ["collection_name"]
                }
            },
            {
                "name": "batch_embed_and_store",
                "description": "Embed texts and store them in a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Target collection name"
                        },
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "text": {"type": "string"},
                                    "metadata": {"type": "object"}
                                },
                                "required": ["id", "text"]
                            },
                            "description": "Documents to embed and store"
                        }
                    },
                    "required": ["collection_name", "documents"]
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a vector search tool."""
        try:
            if tool_name == "create_collection":
                return await self._create_collection(arguments)
            elif tool_name == "add_vectors":
                return await self._add_vectors(arguments)
            elif tool_name == "search_vectors":
                return await self._search_vectors(arguments)
            elif tool_name == "generate_embedding":
                return await self._generate_embedding(arguments)
            elif tool_name == "delete_vectors":
                return await self._delete_vectors(arguments)
            elif tool_name == "list_collections":
                return await self._list_collections(arguments)
            elif tool_name == "get_collection_info":
                return await self._get_collection_info(arguments)
            elif tool_name == "batch_embed_and_store":
                return await self._batch_embed_and_store(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing vector search tool '{tool_name}': {e}")
            return [{
                "type": "text",
                "text": f"Error executing {tool_name}: {str(e)}"
            }]

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available vector search resources."""
        resources = []

        # Add collections as resources
        try:
            collections = await self._get_collections_list()
            for collection_name in collections:
                resources.append({
                    "uri": f"vector://{self.name}/collection/{collection_name}",
                    "mimeType": "application/json",
                    "description": f"Vector collection: {collection_name}"
                })
        except Exception as e:
            logger.warning(f"Failed to list collections for resources: {e}")

        # Add backend information
        resources.append({
            "uri": f"vector://{self.name}/backend",
            "mimeType": "application/json",
            "description": "Vector database backend information"
        })

        return resources

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a vector search resource."""
        try:
            if not uri.startswith("vector://"):
                raise ValueError(f"Invalid vector resource URI: {uri}")

            # Parse URI: vector://server_name/resource_type/resource_id
            parts = uri[9:].split("/", 2)  # Remove "vector://" prefix
            if len(parts) < 2:
                raise ValueError(f"Invalid vector resource URI format: {uri}")

            server_name = parts[0]
            resource_type = parts[1]

            if server_name != self.name:
                raise ValueError(f"Resource belongs to different server: {server_name}")

            if resource_type == "collection" and len(parts) > 2:
                collection_name = parts[2]
                info = await self._get_collection_details(collection_name)
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(info, indent=2)
                    }]
                }

            elif resource_type == "backend":
                backend_info = self._get_backend_info()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(backend_info, indent=2)
                    }]
                }

            else:
                raise ValueError(f"Unknown resource type: {resource_type}")

        except Exception as e:
            logger.error(f"Error reading vector resource '{uri}': {e}")
            raise

    def _validate_config(self):
        """Validate vector search server configuration."""
        backend = self.config.get("backend", "in_memory")
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")

        # Validate backend-specific configuration
        if backend == "pinecone":
            if not self.config.get("api_key"):
                raise ValueError("Pinecone backend requires api_key")
            if not self.config.get("environment"):
                raise ValueError("Pinecone backend requires environment")

        elif backend == "weaviate":
            if not self.config.get("url"):
                raise ValueError("Weaviate backend requires url")

        # Validate embedder configuration if present
        embedder = self.config.get("embedder", {})
        if embedder:
            embedder_type = embedder.get("type")
            if embedder_type and embedder_type not in ["openai", "sentence-transformers", "cohere"]:
                logger.warning(f"Unknown embedder type: {embedder_type}")

    async def _initialize_backend(self, backend_type: str):
        """Initialize the vector database backend."""
        try:
            if backend_type == "in_memory":
                # Simple in-memory storage for testing
                self.backend = "in_memory"
                self.client = {}
                logger.info("Initialized in-memory vector backend")

            elif backend_type == "chromadb":
                try:
                    import chromadb
                    self.client = chromadb.PersistentClient(path=self.config.get("persist_directory", "./chroma_db"))
                    self.backend = "chromadb"
                except ImportError:
                    raise ImportError("chromadb package is required for ChromaDB backend")

            elif backend_type == "pinecone":
                try:
                    import pinecone
                    pinecone.init(
                        api_key=self.config["api_key"],
                        environment=self.config["environment"]
                    )
                    self.client = pinecone
                    self.backend = "pinecone"
                except ImportError:
                    raise ImportError("pinecone-client package is required for Pinecone backend")

            elif backend_type == "weaviate":
                try:
                    import weaviate
                    self.client = weaviate.Client(self.config["url"])
                    self.backend = "weaviate"
                except ImportError:
                    raise ImportError("weaviate-client package is required for Weaviate backend")

            elif backend_type == "qdrant":
                try:
                    from qdrant_client import QdrantClient
                    url = self.config.get("url", "localhost:6333")
                    self.client = QdrantClient(url=url)
                    self.backend = "qdrant"
                except ImportError:
                    raise ImportError("qdrant-client package is required for Qdrant backend")

            elif backend_type == "milvus":
                try:
                    from pymilvus import connections
                    connections.connect(
                        alias="default",
                        host=self.config.get("host", "localhost"),
                        port=self.config.get("port", 19530)
                    )
                    self.client = connections
                    self.backend = "milvus"
                except ImportError:
                    raise ImportError("pymilvus package is required for Milvus backend")

            else:
                raise ValueError(f"Unsupported backend: {backend_type}")

        except Exception as e:
            logger.error(f"Failed to initialize {backend_type} backend: {e}")
            raise

    async def _test_backend_connectivity(self):
        """Test backend connectivity."""
        if self.backend == "in_memory":
            pass  # Always connected
        elif self.backend == "chromadb":
            self.client.heartbeat()
        elif self.backend == "pinecone":
            self.client.list_indexes()
        elif self.backend == "weaviate":
            self.client.is_ready()
        elif self.backend == "qdrant":
            self.client.get_collections()
        elif self.backend == "milvus":
            # Test connection by trying to list collections
            from pymilvus import utility
            utility.list_collections()

    async def _create_collection(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a new vector collection."""
        collection_name = args["collection_name"]
        dimension = args["dimension"]
        distance_metric = args.get("distance_metric", "cosine")
        metadata = args.get("metadata", {})

        try:
            if self.backend == "in_memory":
                self.client[collection_name] = {
                    "vectors": {},
                    "dimension": dimension,
                    "distance_metric": distance_metric,
                    "metadata": metadata,
                    "created_at": datetime.utcnow().isoformat()
                }

            elif self.backend == "chromadb":
                self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "dimension": dimension,
                        "distance_metric": distance_metric,
                        **metadata
                    }
                )

            elif self.backend == "pinecone":
                self.client.create_index(
                    name=collection_name,
                    dimension=dimension,
                    metric=distance_metric,
                    pods=1,
                    replicas=1
                )

            # Add to local collections cache
            self.collections[collection_name] = {
                "dimension": dimension,
                "distance_metric": distance_metric,
                "created_at": datetime.utcnow().isoformat()
            }

            return [{
                "type": "text",
                "text": f"Successfully created collection '{collection_name}' with dimension {dimension}"
            }]

        except Exception as e:
            raise Exception(f"Failed to create collection '{collection_name}': {str(e)}")

    async def _add_vectors(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add vectors to a collection."""
        collection_name = args["collection_name"]
        vectors_data = args["vectors"]

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        try:
            if self.backend == "in_memory":
                collection = self.client[collection_name]
                for vector_data in vectors_data:
                    vector_id = vector_data["id"]
                    vector = vector_data["vector"]
                    metadata = vector_data.get("metadata", {})

                    # Validate dimension
                    if len(vector) != collection["dimension"]:
                        raise ValueError(f"Vector dimension {len(vector)} does not match collection dimension {collection['dimension']}")

                    collection["vectors"][vector_id] = {
                        "vector": vector,
                        "metadata": metadata
                    }

            elif self.backend == "chromadb":
                collection = self.client.get_collection(collection_name)
                ids = [v["id"] for v in vectors_data]
                embeddings = [v["vector"] for v in vectors_data]
                metadatas = [v.get("metadata", {}) for v in vectors_data]

                collection.add(
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )

            return [{
                "type": "text",
                "text": f"Successfully added {len(vectors_data)} vectors to collection '{collection_name}'"
            }]

        except Exception as e:
            raise Exception(f"Failed to add vectors to collection '{collection_name}': {str(e)}")

    async def _search_vectors(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        collection_name = args["collection_name"]
        query_vector = args.get("query_vector")
        query_text = args.get("query_text")
        top_k = args.get("top_k", 10)
        include_metadata = args.get("include_metadata", True)
        filter_criteria = args.get("filter")

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        # Handle text queries by generating embeddings
        if query_text and not query_vector:
            query_vector = await self._embed_text(query_text)
            if not query_vector:
                raise ValueError("Failed to generate embedding for query text")

        if not query_vector:
            raise ValueError("Either query_vector or query_text must be provided")

        try:
            results = []

            if self.backend == "in_memory":
                collection = self.client[collection_name]
                metric = collection["distance_metric"]

                # Calculate similarities (simplified implementation)
                similarities = []
                for vector_id, vector_data in collection["vectors"].items():
                    stored_vector = vector_data["vector"]
                    similarity = self._calculate_similarity(query_vector, stored_vector, metric)
                    similarities.append((vector_id, similarity, vector_data))

                # Sort by similarity (higher is better for cosine, lower for euclidean)
                similarities.sort(key=lambda x: x[1], reverse=(metric == "cosine"))

                for vector_id, similarity, vector_data in similarities[:top_k]:
                    result = {
                        "id": vector_id,
                        "score": similarity
                    }
                    if include_metadata:
                        result["metadata"] = vector_data["metadata"]
                    results.append(result)

            elif self.backend == "chromadb":
                collection = self.client.get_collection(collection_name)
                search_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    include=["metadatas", "distances"] if include_metadata else ["distances"]
                )

                for i, (id_val, distance) in enumerate(zip(
                    search_results["ids"][0],
                    search_results["distances"][0]
                )):
                    result = {
                        "id": id_val,
                        "score": 1.0 - distance if collection.metadata.get("distance_metric") == "cosine" else distance
                    }
                    if include_metadata and "metadatas" in search_results:
                        result["metadata"] = search_results["metadatas"][0][i]
                    results.append(result)

            return [{
                "type": "text",
                "text": f"Search Results for collection '{collection_name}':\n\n" +
                       "\n".join([f"• ID: {r['id']}, Score: {r['score']:.3f}{f', Metadata: {r.get(\"metadata\", {})}' if include_metadata else ''}"
                                 for r in results])
            }]

        except Exception as e:
            raise Exception(f"Failed to search vectors in collection '{collection_name}': {str(e)}")

    async def _generate_embedding(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate embeddings for text."""
        text = args.get("text")
        texts = args.get("texts", [])

        if text and texts:
            raise ValueError("Provide either 'text' or 'texts', not both")
        elif text:
            texts = [text]
        elif not texts:
            raise ValueError("Either 'text' or 'texts' must be provided")

        try:
            embeddings = []
            for t in texts:
                embedding = await self._embed_text(t)
                if embedding:
                    embeddings.append(embedding)
                else:
                    embeddings.append(None)

            return [{
                "type": "text",
                "text": f"Generated {len(embeddings)} embeddings:\n\n" +
                       "\n".join([f"Text: {texts[i][:50]}...\nEmbedding: [{embeddings[i][0]:.3f}, {embeddings[i][1]:.3f}, ...] ({len(embeddings[i])} dims)"
                                 if embeddings[i] else f"Text: {texts[i][:50]}...\nEmbedding: Failed"
                                 for i in range(len(texts))])
            }]

        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using configured embedder."""
        embedder_config = self.config.get("embedder", {})

        if not embedder_config:
            # Simple mock embedding for testing
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # Generate a 128-dimensional mock embedding
            embedding = []
            for i in range(128):
                embedding.append(((hash_int >> (i % 32)) & 1) * 2.0 - 1.0)  # -1 to 1
            return embedding

        embedder_type = embedder_config.get("type")

        try:
            if embedder_type == "openai":
                import openai
                client = openai.OpenAI(api_key=embedder_config.get("api_key"))
                response = client.embeddings.create(
                    input=text,
                    model=embedder_config.get("model", "text-embedding-ada-002")
                )
                return response.data[0].embedding

            elif embedder_type == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                model_name = embedder_config.get("model", "all-MiniLM-L6-v2")
                model = SentenceTransformer(model_name)
                embedding = model.encode(text)
                return embedding.tolist()

            elif embedder_type == "cohere":
                import cohere
                client = cohere.Client(api_key=embedder_config["api_key"])
                response = client.embed(
                    texts=[text],
                    model=embedder_config.get("model", "embed-english-v2.0")
                )
                return response.embeddings[0]

            else:
                logger.warning(f"Unknown embedder type: {embedder_type}")
                return None

        except ImportError as e:
            logger.warning(f"Embedder dependencies not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _calculate_similarity(self, vec1: List[float], vec2: List[float], metric: str) -> float:
        """Calculate similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        if metric == "cosine":
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0
        elif metric == "euclidean":
            return -np.linalg.norm(v1 - v2)  # Negative for descending sort
        elif metric == "dot_product":
            return np.dot(v1, v2)
        else:
            return 0.0

    async def _delete_vectors(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delete vectors from a collection."""
        collection_name = args["collection_name"]
        vector_ids = args["vector_ids"]

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        try:
            if self.backend == "in_memory":
                collection = self.client[collection_name]
                deleted_count = 0
                for vector_id in vector_ids:
                    if vector_id in collection["vectors"]:
                        del collection["vectors"][vector_id]
                        deleted_count += 1

            elif self.backend == "chromadb":
                collection = self.client.get_collection(collection_name)
                collection.delete(ids=vector_ids)
                deleted_count = len(vector_ids)

            return [{
                "type": "text",
                "text": f"Successfully deleted {deleted_count} vectors from collection '{collection_name}'"
            }]

        except Exception as e:
            raise Exception(f"Failed to delete vectors from collection '{collection_name}': {str(e)}")

    async def _list_collections(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List all collections."""
        detailed = args.get("detailed", False)

        try:
            collections_info = []
            collection_names = await self._get_collections_list()

            for name in collection_names:
                info = {"name": name}
                if detailed:
                    details = await self._get_collection_details(name)
                    info.update(details)
                collections_info.append(info)

            return [{
                "type": "text",
                "text": f"Collections ({len(collections_info)} total):\n\n" +
                       "\n".join([f"• {info['name']}" +
                                 (f" - {info.get('vector_count', 'Unknown')} vectors, {info.get('dimension', 'Unknown')}D"
                                  if detailed else "")
                                 for info in collections_info])
            }]

        except Exception as e:
            raise Exception(f"Failed to list collections: {str(e)}")

    async def _get_collection_info(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get information about a specific collection."""
        collection_name = args["collection_name"]

        try:
            details = await self._get_collection_details(collection_name)

            return [{
                "type": "text",
                "text": f"Collection '{collection_name}' Information:\n\n" +
                       f"• Dimension: {details.get('dimension', 'Unknown')}\n" +
                       f"• Vectors: {details.get('vector_count', 'Unknown')}\n" +
                       f"• Distance Metric: {details.get('distance_metric', 'Unknown')}\n" +
                       f"• Created: {details.get('created_at', 'Unknown')}\n" +
                       f"• Backend: {self.backend}"
            }]

        except Exception as e:
            raise Exception(f"Failed to get collection info for '{collection_name}': {str(e)}")

    async def _batch_embed_and_store(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Embed texts and store them in a collection."""
        collection_name = args["collection_name"]
        documents = args["documents"]

        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        try:
            # Generate embeddings for all documents
            texts = [doc["text"] for doc in documents]
            embeddings = []

            for text in texts:
                embedding = await self._embed_text(text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    raise ValueError(f"Failed to generate embedding for text: {text[:50]}...")

            # Prepare vectors data
            vectors_data = []
            for i, doc in enumerate(documents):
                vectors_data.append({
                    "id": doc["id"],
                    "vector": embeddings[i],
                    "metadata": {
                        **doc.get("metadata", {}),
                        "text": doc["text"],
                        "embedded_at": datetime.utcnow().isoformat()
                    }
                })

            # Add to collection
            add_args = {
                "collection_name": collection_name,
                "vectors": vectors_data
            }

            result = await self._add_vectors(add_args)

            return [{
                "type": "text",
                "text": f"Successfully embedded and stored {len(documents)} documents in collection '{collection_name}'"
            }]

        except Exception as e:
            raise Exception(f"Failed to batch embed and store documents: {str(e)}")

    async def _get_collections_list(self) -> List[str]:
        """Get list of all collections."""
        try:
            if self.backend == "in_memory":
                return list(self.client.keys())
            elif self.backend == "chromadb":
                return [c.name for c in self.client.list_collections()]
            elif self.backend == "pinecone":
                return self.client.list_indexes()
            elif self.backend == "qdrant":
                collections = self.client.get_collections()
                return [c.name for c in collections.collections]
            else:
                return list(self.collections.keys())
        except Exception as e:
            logger.warning(f"Failed to get collections list: {e}")
            return list(self.collections.keys())

    async def _get_collection_details(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection."""
        if collection_name in self.collections:
            return self.collections[collection_name]

        # Try to get from backend
        try:
            if self.backend == "in_memory":
                collection = self.client.get(collection_name, {})
                return {
                    "dimension": collection.get("dimension"),
                    "vector_count": len(collection.get("vectors", {})),
                    "distance_metric": collection.get("distance_metric"),
                    "created_at": collection.get("created_at")
                }
            elif self.backend == "chromadb":
                collection = self.client.get_collection(collection_name)
                return {
                    "dimension": collection.metadata.get("dimension"),
                    "vector_count": collection.count(),
                    "distance_metric": collection.metadata.get("distance_metric")
                }
        except Exception as e:
            logger.warning(f"Failed to get collection details from backend: {e}")

        return {"error": "Collection details not available"}

    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "type": self.backend,
            "description": self.SUPPORTED_BACKENDS.get(self.backend, {}).get("description", "Unknown"),
            "config": {k: v for k, v in self.config.items() if k not in ["api_key", "password"]},
            "collections_count": len(self.collections)
        }
