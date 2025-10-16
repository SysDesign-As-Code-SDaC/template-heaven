"""
Search Server for MCP Middleware.

This server provides AI assistants with search capabilities across various
data sources including web search, document search, and structured data queries.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
import aiohttp
from urllib.parse import urlparse, quote_plus
from datetime import datetime
import re

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class SearchServer(BaseMCPServer):
    """
    MCP server for search operations.

    Provides tools for web search, document search, database queries,
    and structured data retrieval.
    """

    def __init__(self, name: str, config: Dict[str, Any], auth: Optional[Dict[str, Any]] = None):
        super().__init__(name, config, auth)
        self.session: Optional[aiohttp.ClientSession] = None
        self.search_engines = {
            "google": "https://www.googleapis.com/customsearch/v1",
            "bing": "https://api.bing.microsoft.com/v7.0/search",
            "duckduckgo": "https://api.duckduckgo.com/",
            "searx": None  # Configurable
        }

    async def initialize(self) -> bool:
        """Initialize the search server."""
        try:
            # Validate configuration
            self._validate_config()

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)

            logger.info(f"Search server '{self.name}' initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize search server '{self.name}': {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the search server."""
        try:
            if self.session:
                await self.session.close()
                self.session = None

            logger.info(f"Search server '{self.name}' shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down search server '{self.name}': {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            health_status = {
                "status": "healthy" if self.session else "unhealthy",
                "session_active": self.session is not None,
                "search_engines_configured": len(self._get_configured_engines()) > 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            return health_status

        except Exception as e:
            logger.error(f"Health check failed for search server '{self.name}': {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available search tools."""
        return [
            {
                "name": "web_search",
                "description": "Search the web using configured search engines",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["google", "bing", "duckduckgo", "searx"],
                            "description": "Search engine to use",
                            "default": "duckduckgo"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10
                        },
                        "language": {
                            "type": "string",
                            "description": "Language for search results",
                            "default": "en"
                        },
                        "safe_search": {
                            "type": "boolean",
                            "description": "Enable safe search filtering",
                            "default": true
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "document_search",
                "description": "Search within documents and files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to search in"
                        },
                        "file_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["txt", "md", "pdf", "docx", "html", "json", "xml"]
                            },
                            "description": "File types to include in search"
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Case sensitive search",
                            "default": false
                        },
                        "regex": {
                            "type": "boolean",
                            "description": "Use regex pattern matching",
                            "default": false
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "structured_search",
                "description": "Search structured data sources (databases, APIs)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query or filter criteria"
                        },
                        "data_source": {
                            "type": "string",
                            "description": "Data source identifier (database, API endpoint)"
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to search in"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Additional filter criteria"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 100
                        }
                    },
                    "required": ["query", "data_source"]
                }
            },
            {
                "name": "semantic_search",
                "description": "Perform semantic similarity search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for semantic matching"
                        },
                        "documents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of documents to search through"
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Similarity threshold (0-1)",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.7
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query", "documents"]
                }
            },
            {
                "name": "multi_source_search",
                "description": "Search across multiple sources simultaneously",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["web", "document", "structured"]},
                                    "config": {"type": "object"}
                                }
                            },
                            "description": "List of search sources with configurations"
                        },
                        "combine_results": {
                            "type": "boolean",
                            "description": "Combine and deduplicate results",
                            "default": true
                        }
                    },
                    "required": ["query", "sources"]
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a search tool."""
        try:
            if tool_name == "web_search":
                return await self._execute_web_search(arguments)
            elif tool_name == "document_search":
                return await self._execute_document_search(arguments)
            elif tool_name == "structured_search":
                return await self._execute_structured_search(arguments)
            elif tool_name == "semantic_search":
                return await self._execute_semantic_search(arguments)
            elif tool_name == "multi_source_search":
                return await self._execute_multi_source_search(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing search tool '{tool_name}': {e}")
            return [{
                "type": "text",
                "text": f"Error executing {tool_name}: {str(e)}"
            }]

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available search resources."""
        resources = []

        # Add search history as resources
        resources.append({
            "uri": f"search://{self.name}/history",
            "mimeType": "application/json",
            "description": "Search history and cached results"
        })

        # Add configured search indexes
        if "indexes" in self.config:
            for index_name, index_config in self.config["indexes"].items():
                resources.append({
                    "uri": f"search://{self.name}/index/{index_name}",
                    "mimeType": "application/json",
                    "description": index_config.get("description", f"Search index: {index_name}")
                })

        return resources

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a search resource."""
        try:
            if not uri.startswith("search://"):
                raise ValueError(f"Invalid search resource URI: {uri}")

            # Parse URI: search://server_name/resource_type/resource_id
            parts = uri[9:].split("/", 2)  # Remove "search://" prefix
            if len(parts) < 2:
                raise ValueError(f"Invalid search resource URI format: {uri}")

            server_name = parts[0]
            resource_type = parts[1]

            if server_name != self.name:
                raise ValueError(f"Resource belongs to different server: {server_name}")

            if resource_type == "history":
                # Return search history
                history = self._get_search_history()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(history, indent=2)
                    }]
                }

            elif resource_type == "index" and len(parts) > 2:
                index_name = parts[2]
                index_data = self._get_index_data(index_name)
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(index_data, indent=2)
                    }]
                }

            else:
                raise ValueError(f"Unknown resource type: {resource_type}")

        except Exception as e:
            logger.error(f"Error reading search resource '{uri}': {e}")
            raise

    def _validate_config(self):
        """Validate search server configuration."""
        # Validate search engine configurations
        search_engines = self.config.get("search_engines", {})

        for engine_name, engine_config in search_engines.items():
            if engine_name not in self.search_engines:
                raise ValueError(f"Unsupported search engine: {engine_name}")

            if engine_name in ["google", "bing"] and "api_key" not in engine_config:
                raise ValueError(f"API key required for {engine_name} search engine")

        # Validate timeout
        if "timeout" in self.config:
            if not isinstance(self.config["timeout"], (int, float)) or self.config["timeout"] <= 0:
                raise ValueError("Timeout must be a positive number")

    def _get_configured_engines(self) -> List[str]:
        """Get list of configured search engines."""
        configured = []
        search_engines = self.config.get("search_engines", {})

        for engine_name in self.search_engines.keys():
            if engine_name in search_engines or engine_name == "duckduckgo":
                configured.append(engine_name)

        return configured

    async def _execute_web_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute web search."""
        query = args["query"]
        engine = args.get("engine", "duckduckgo")
        num_results = args.get("num_results", 10)
        language = args.get("language", "en")
        safe_search = args.get("safe_search", True)

        # Check if engine is configured
        if engine not in self._get_configured_engines():
            raise ValueError(f"Search engine '{engine}' is not configured")

        # Execute search based on engine
        if engine == "duckduckgo":
            results = await self._search_duckduckgo(query, num_results, language, safe_search)
        elif engine == "google":
            results = await self._search_google(query, num_results, language, safe_search)
        elif engine == "bing":
            results = await self._search_bing(query, num_results, language, safe_search)
        elif engine == "searx":
            results = await self._search_searx(query, num_results, language, safe_search)
        else:
            raise ValueError(f"Unsupported search engine: {engine}")

        # Format results
        formatted_results = []
        for i, result in enumerate(results[:num_results], 1):
            formatted_results.append(f"{i}. **{result['title']}**\n   {result['url']}\n   {result['snippet']}\n")

        return [{
            "type": "text",
            "text": f"Web Search Results for '{query}' using {engine}:\n\n" + "\n".join(formatted_results)
        }]

    async def _search_duckduckgo(self, query: str, num_results: int, language: str, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }

        if language != "en":
            params["kl"] = language

        url = "https://api.duckduckgo.com/"
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []

                # Extract instant answer if available
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "Instant Answer"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("AbstractText", "")
                    })

                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:num_results]:
                    if "Text" in topic and "FirstURL" in topic:
                        results.append({
                            "title": topic.get("Text", "").split(" - ")[0],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })

                return results
            else:
                raise Exception(f"DuckDuckGo API error: {response.status}")

    async def _search_google(self, query: str, num_results: int, language: str, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        engine_config = self.config.get("search_engines", {}).get("google", {})
        api_key = engine_config.get("api_key")
        search_engine_id = engine_config.get("search_engine_id")

        if not api_key or not search_engine_id:
            raise ValueError("Google search requires api_key and search_engine_id")

        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": min(num_results, 10),  # Google limits to 10 per request
            "safe": "active" if safe_search else "off"
        }

        if language != "en":
            params["lr"] = f"lang_{language}"

        url = "https://www.googleapis.com/customsearch/v1"
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []

                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })

                return results
            else:
                raise Exception(f"Google API error: {response.status}")

    async def _search_bing(self, query: str, num_results: int, language: str, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using Bing Web Search API."""
        engine_config = self.config.get("search_engines", {}).get("bing", {})
        api_key = engine_config.get("api_key")

        if not api_key:
            raise ValueError("Bing search requires api_key")

        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {
            "q": query,
            "count": min(num_results, 50),  # Bing limits to 50
            "safeSearch": "Strict" if safe_search else "Off"
        }

        if language != "en":
            params["mkt"] = language

        url = "https://api.bing.microsoft.com/v7.0/search"
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []

                for item in data.get("webPages", {}).get("value", []):
                    results.append({
                        "title": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", "")
                    })

                return results
            else:
                raise Exception(f"Bing API error: {response.status}")

    async def _search_searx(self, query: str, num_results: int, language: str, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using SearX instance."""
        engine_config = self.config.get("search_engines", {}).get("searx", {})
        base_url = engine_config.get("base_url")

        if not base_url:
            raise ValueError("SearX search requires base_url configuration")

        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "lang": language
        }

        if safe_search:
            params["safesearch"] = "1"

        url = f"{base_url}/search"
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []

                for result in data.get("results", [])[:num_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", "")
                    })

                return results
            else:
                raise Exception(f"SearX API error: {response.status}")

    async def _execute_document_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute document search."""
        query = args["query"]
        file_paths = args.get("file_paths", [])
        file_types = args.get("file_types", [])
        case_sensitive = args.get("case_sensitive", False)
        use_regex = args.get("regex", False)

        results = []

        # If no file paths specified, search in configured directories
        if not file_paths:
            search_dirs = self.config.get("search_directories", [])
            for search_dir in search_dirs:
                # This would require filesystem access - simplified for now
                results.extend(await self._search_directory(search_dir, query, file_types, case_sensitive, use_regex))

        # Search specified files
        for file_path in file_paths:
            file_results = await self._search_file(file_path, query, case_sensitive, use_regex)
            results.extend(file_results)

        # Format results
        formatted_results = []
        for result in results[:50]:  # Limit results
            formatted_results.append(f"**{result['file']}** (line {result['line']}): {result['match']}")

        return [{
            "type": "text",
            "text": f"Document Search Results for '{query}':\n\n" + "\n".join(formatted_results)
        }]

    async def _execute_structured_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute structured data search."""
        query = args["query"]
        data_source = args["data_source"]
        fields = args.get("fields", [])
        filters = args.get("filters", {})
        limit = args.get("limit", 100)

        # This would integrate with database or API endpoints
        # Simplified implementation for now
        results = await self._search_structured_data(data_source, query, fields, filters, limit)

        return [{
            "type": "text",
            "text": f"Structured Search Results from '{data_source}':\n\n{json.dumps(results, indent=2)}"
        }]

    async def _execute_semantic_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute semantic similarity search."""
        query = args["query"]
        documents = args["documents"]
        threshold = args.get("threshold", 0.7)
        max_results = args.get("max_results", 10)

        # This would require NLP/similarity libraries
        # Simplified implementation for now
        results = []

        for i, doc in enumerate(documents):
            # Mock similarity calculation
            similarity = self._calculate_similarity(query, doc)
            if similarity >= threshold:
                results.append({
                    "document_index": i,
                    "similarity": similarity,
                    "content": doc[:200] + "..." if len(doc) > 200 else doc
                })

        # Sort by similarity and limit results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:max_results]

        return [{
            "type": "text",
            "text": f"Semantic Search Results (threshold: {threshold}):\n\n" +
                   "\n".join([f"Document {r['document_index']}: {r['similarity']:.3f}\n  {r['content']}\n"
                             for r in results])
        }]

    async def _execute_multi_source_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute multi-source search."""
        query = args["query"]
        sources = args["sources"]
        combine_results = args.get("combine_results", True)

        all_results = []

        # Execute search on each source
        for source in sources:
            source_type = source["type"]
            source_config = source.get("config", {})

            try:
                if source_type == "web":
                    results = await self._execute_web_search({**source_config, "query": query})
                elif source_type == "document":
                    results = await self._execute_document_search({**source_config, "query": query})
                elif source_type == "structured":
                    results = await self._execute_structured_search({**source_config, "query": query})
                else:
                    continue

                all_results.extend(results)

            except Exception as e:
                logger.warning(f"Error searching source {source_type}: {e}")

        if combine_results:
            # Combine and deduplicate results
            combined = self._combine_search_results(all_results)
            return combined
        else:
            return all_results

    def _calculate_similarity(self, query: str, document: str) -> float:
        """Calculate simple text similarity (placeholder for real NLP)."""
        # Very basic similarity calculation
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def _search_directory(self, directory: str, query: str, file_types: List[str],
                               case_sensitive: bool, use_regex: bool) -> List[Dict[str, Any]]:
        """Search files in a directory (placeholder implementation)."""
        # This would require actual filesystem access
        # For now, return empty results
        return []

    async def _search_file(self, file_path: str, query: str, case_sensitive: bool, use_regex: bool) -> List[Dict[str, Any]]:
        """Search within a specific file (placeholder implementation)."""
        # This would require actual file reading
        # For now, return empty results
        return []

    async def _search_structured_data(self, data_source: str, query: str, fields: List[str],
                                    filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Search structured data (placeholder implementation)."""
        # This would integrate with databases or APIs
        # For now, return mock results
        return [
            {"id": 1, "title": f"Mock result for {query}", "score": 0.95},
            {"id": 2, "title": f"Another result for {query}", "score": 0.87}
        ]

    def _combine_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and deduplicate search results."""
        # Simple deduplication based on text content
        seen = set()
        combined = []

        for result in results:
            text = result.get("text", "")
            if text not in seen:
                seen.add(text)
                combined.append(result)

        return combined

    def _get_search_history(self) -> Dict[str, Any]:
        """Get search history (placeholder)."""
        return {
            "total_searches": 0,
            "recent_searches": [],
            "cached_results": {}
        }

    def _get_index_data(self, index_name: str) -> Dict[str, Any]:
        """Get index data (placeholder)."""
        indexes = self.config.get("indexes", {})
        return indexes.get(index_name, {"error": "Index not found"})
