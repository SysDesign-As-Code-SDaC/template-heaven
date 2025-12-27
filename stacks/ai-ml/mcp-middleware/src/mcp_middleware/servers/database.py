"""
Database MCP Server.

This server provides database operations through the MCP protocol,
allowing AI assistants to query, insert, update, and manage database records.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class DatabaseServer(BaseMCPServer):
    """MCP server for database operations."""

    def __init__(self, name: str, config: Dict[str, Any], auth: Dict[str, Any]):
        super().__init__(name, config, auth)
        self.database_url = config.get("database_url")
        self.allowed_operations = config.get("allowed_operations", ["select"])
        self.max_rows = config.get("max_rows", 1000)
        self.timeout = config.get("timeout", 30)

        self.engine = None
        self.session_factory = None
        self.metadata = None

    @classmethod
    def validate_config(cls, config: Dict[str, Any]):
        """Validate database server configuration."""
        if not config.get("database_url"):
            raise ValueError("database_url is required")

        allowed_ops = config.get("allowed_operations", [])
        valid_ops = ["select", "insert", "update", "delete"]
        for op in allowed_ops:
            if op not in valid_ops:
                raise ValueError(f"Invalid operation: {op}")

    async def initialize(self):
        """Initialize the database server."""
        try:
            # Create database engine
            self.engine = create_engine(
                self.database_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"connect_timeout": 10}
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)

            # Load metadata
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)

            self.initialized = True
            logger.info(f"Database server {self.name} initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database server {self.name}: {e}")
            raise

    async def shutdown(self):
        """Shutdown the database server."""
        if self.engine:
            self.engine.dispose()
        self.initialized = False
        logger.info(f"Database server {self.name} shutdown")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            return {
                "healthy": True,
                "timestamp": datetime.utcnow().isoformat(),
                "server": self.name,
                "tables_count": len(self.metadata.tables) if self.metadata else 0
            }
        except Exception as e:
            return {
                "healthy": False,
                "timestamp": datetime.utcnow().isoformat(),
                "server": self.name,
                "error": str(e)
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available database tools."""
        tools = []

        if "select" in self.allowed_operations:
            tools.append({
                "name": "execute_query",
                "description": "Execute a SELECT query and return results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query to execute"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return",
                            "default": 100
                        }
                    },
                    "required": ["query"]
                }
            })

            tools.append({
                "name": "list_tables",
                "description": "List all tables in the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "string",
                            "description": "Schema name to filter tables"
                        }
                    }
                }
            })

            tools.append({
                "name": "describe_table",
                "description": "Get table schema and metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to describe"
                        }
                    },
                    "required": ["table_name"]
                }
            })

        if "insert" in self.allowed_operations:
            tools.append({
                "name": "insert_record",
                "description": "Insert a new record into a table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to insert into"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data to insert as key-value pairs"
                        }
                    },
                    "required": ["table_name", "data"]
                }
            })

        if "update" in self.allowed_operations:
            tools.append({
                "name": "update_records",
                "description": "Update records in a table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to update"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data to update as key-value pairs"
                        },
                        "where": {
                            "type": "object",
                            "description": "WHERE conditions as key-value pairs"
                        }
                    },
                    "required": ["table_name", "data", "where"]
                }
            })

        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a database tool."""
        try:
            if tool_name == "execute_query":
                return await self._execute_query(arguments)
            elif tool_name == "list_tables":
                return await self._list_tables(arguments)
            elif tool_name == "describe_table":
                return await self._describe_table(arguments)
            elif tool_name == "insert_record":
                return await self._insert_record(arguments)
            elif tool_name == "update_records":
                return await self._update_records(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            return self._handle_error(f"call_tool_{tool_name}", e)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List database resources."""
        resources = []

        if self.metadata:
            for table_name in self.metadata.tables.keys():
                resources.append({
                    "uri": f"db://{self.name}/{table_name}",
                    "mimeType": "application/sql",
                    "description": f"Database table: {table_name}"
                })

        return resources

    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Read a database resource."""
        if not resource_uri.startswith(f"db://{self.name}/"):
            raise ValueError(f"Resource URI not within database scope: {resource_uri}")

        table_name = resource_uri.replace(f"db://{self.name}/", "")

        if table_name not in self.metadata.tables:
            raise ValueError(f"Table not found: {table_name}")

        # Return table schema as resource content
        table = self.metadata.tables[table_name]
        schema_info = {
            "table_name": table_name,
            "columns": [
                {
                    "name": col.name,
                    "type": str(col.type),
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "default": str(col.default) if col.default else None
                }
                for col in table.columns
            ]
        }

        return {
            "contents": [
                {
                    "uri": resource_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(schema_info, indent=2)
                }
            ]
        }

    async def _execute_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a SELECT query."""
        if "select" not in self.allowed_operations:
            raise PermissionError("SELECT operation not allowed")

        query = arguments["query"]
        limit = min(arguments.get("limit", 100), self.max_rows)

        # Basic SQL injection protection - only allow SELECT
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        try:
            with self.session_factory() as session:
                # Add LIMIT if not present
                if "LIMIT" not in query.upper():
                    query = f"{query} LIMIT {limit}"

                result = session.execute(text(query))
                rows = result.fetchall()

                # Convert to list of dicts
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in rows]

                return {
                    "query": query,
                    "rows_returned": len(data),
                    "columns": list(columns),
                    "data": data
                }

        except SQLAlchemyError as e:
            raise ValueError(f"Query execution failed: {str(e)}")

    async def _list_tables(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List database tables."""
        schema = arguments.get("schema")

        tables = []
        if self.metadata:
            for table_name, table in self.metadata.tables.items():
                if schema and table.schema != schema:
                    continue

                tables.append({
                    "name": table_name,
                    "schema": table.schema,
                    "columns_count": len(table.columns)
                })

        return {
            "tables": tables,
            "count": len(tables),
            "schema_filter": schema
        }

    async def _describe_table(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Describe table schema."""
        table_name = arguments["table_name"]

        if table_name not in self.metadata.tables:
            raise ValueError(f"Table not found: {table_name}")

        table = self.metadata.tables[table_name]

        columns = []
        for col in table.columns:
            columns.append({
                "name": col.name,
                "type": str(col.type),
                "nullable": col.nullable,
                "primary_key": col.primary_key,
                "default": str(col.default) if col.default else None,
                "autoincrement": col.autoincrement,
            })

        indexes = []
        if hasattr(table, 'indexes'):
            for idx in table.indexes:
                indexes.append({
                    "name": idx.name,
                    "columns": [col.name for col in idx.columns],
                    "unique": idx.unique
                })

        return {
            "table_name": table_name,
            "schema": table.schema,
            "columns": columns,
            "indexes": indexes,
            "primary_keys": [col.name for col in table.primary_key.columns]
        }

    async def _insert_record(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a record into a table."""
        if "insert" not in self.allowed_operations:
            raise PermissionError("INSERT operation not allowed")

        table_name = arguments["table_name"]
        data = arguments["data"]

        if table_name not in self.metadata.tables:
            raise ValueError(f"Table not found: {table_name}")

        try:
            with self.session_factory() as session:
                table = self.metadata.tables[table_name]

                # Validate columns exist
                table_columns = {col.name for col in table.columns}
                invalid_columns = set(data.keys()) - table_columns
                if invalid_columns:
                    raise ValueError(f"Invalid columns: {invalid_columns}")

                # Execute insert
                stmt = table.insert().values(**data)
                result = session.execute(stmt)
                session.commit()

                return {
                    "table_name": table_name,
                    "inserted": True,
                    "row_id": result.inserted_primary_key[0] if result.inserted_primary_key else None
                }

        except SQLAlchemyError as e:
            raise ValueError(f"Insert failed: {str(e)}")

    async def _update_records(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Update records in a table."""
        if "update" not in self.allowed_operations:
            raise PermissionError("UPDATE operation not allowed")

        table_name = arguments["table_name"]
        data = arguments["data"]
        where_conditions = arguments["where"]

        if table_name not in self.metadata.tables:
            raise ValueError(f"Table not found: {table_name}")

        try:
            with self.session_factory() as session:
                table = self.metadata.tables[table_name]

                # Build WHERE clause
                where_clause = []
                for key, value in where_conditions.items():
                    if key in table.columns:
                        where_clause.append(table.columns[key] == value)
                    else:
                        raise ValueError(f"Invalid WHERE column: {key}")

                # Execute update
                stmt = table.update().where(*where_clause).values(**data)
                result = session.execute(stmt)
                session.commit()

                return {
                    "table_name": table_name,
                    "updated": True,
                    "rows_affected": result.rowcount
                }

        except SQLAlchemyError as e:
            raise ValueError(f"Update failed: {str(e)}")
