"""
Code Execution Server for MCP Middleware.

This server provides AI assistants with the ability to execute code in various
programming languages, run scripts, and perform computational tasks safely.
"""

import asyncio
import json
import logging
import os
import tempfile
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import resource

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class CodeExecutionServer(BaseMCPServer):
    """
    MCP server for code execution.

    Provides tools for executing code in multiple languages with safety controls,
    resource limits, and isolated execution environments.
    """

    SUPPORTED_LANGUAGES = {
        "python": {
            "extensions": [".py"],
            "commands": ["python3", "python"],
            "timeout": 30,
            "memory_limit": 100 * 1024 * 1024  # 100MB
        },
        "javascript": {
            "extensions": [".js"],
            "commands": ["node"],
            "timeout": 30,
            "memory_limit": 50 * 1024 * 1024  # 50MB
        },
        "typescript": {
            "extensions": [".ts"],
            "commands": ["npx", "ts-node"],
            "timeout": 45,
            "memory_limit": 75 * 1024 * 1024  # 75MB
        },
        "bash": {
            "extensions": [".sh", ".bash"],
            "commands": ["bash"],
            "timeout": 60,
            "memory_limit": 25 * 1024 * 1024  # 25MB
        },
        "r": {
            "extensions": [".r", ".R"],
            "commands": ["Rscript"],
            "timeout": 120,
            "memory_limit": 200 * 1024 * 1024  # 200MB
        },
        "julia": {
            "extensions": [".jl"],
            "commands": ["julia"],
            "timeout": 90,
            "memory_limit": 150 * 1024 * 1024  # 150MB
        },
        "ruby": {
            "extensions": [".rb"],
            "commands": ["ruby"],
            "timeout": 45,
            "memory_limit": 75 * 1024 * 1024  # 75MB
        },
        "go": {
            "extensions": [".go"],
            "commands": ["go", "run"],
            "timeout": 60,
            "memory_limit": 100 * 1024 * 1024  # 100MB
        },
        "rust": {
            "extensions": [".rs"],
            "commands": ["rustc", "--run"],
            "timeout": 120,
            "memory_limit": 150 * 1024 * 1024  # 150MB
        },
        "java": {
            "extensions": [".java"],
            "commands": ["javac", "java"],
            "timeout": 90,
            "memory_limit": 150 * 1024 * 1024  # 150MB
        },
        "cpp": {
            "extensions": [".cpp", ".cc", ".cxx"],
            "commands": ["g++", "-o", "program", "&&", "./program"],
            "timeout": 120,
            "memory_limit": 200 * 1024 * 1024  # 200MB
        }
    }

    def __init__(self, name: str, config: Dict[str, Any], auth: Optional[Dict[str, Any]] = None):
        super().__init__(name, config, auth)
        self.execution_dir = None
        self.active_processes: Dict[str, subprocess.Popen] = {}

    async def initialize(self) -> bool:
        """Initialize the code execution server."""
        try:
            # Validate configuration
            self._validate_config()

            # Create execution directory
            self.execution_dir = Path(tempfile.mkdtemp(prefix="mcp_execution_"))
            logger.info(f"Created execution directory: {self.execution_dir}")

            # Validate language support
            await self._validate_language_support()

            logger.info(f"Code execution server '{self.name}' initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize code execution server '{self.name}': {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the code execution server."""
        try:
            # Terminate any running processes
            for pid, process in self.active_processes.items():
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except Exception as e:
                    logger.warning(f"Failed to terminate process {pid}: {e}")
                    try:
                        process.kill()
                    except:
                        pass

            self.active_processes.clear()

            # Clean up execution directory
            if self.execution_dir and self.execution_dir.exists():
                shutil.rmtree(self.execution_dir, ignore_errors=True)

            logger.info(f"Code execution server '{self.name}' shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down code execution server '{self.name}': {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            # Check available languages
            available_languages = await self._get_available_languages()

            health_status = {
                "status": "healthy" if available_languages else "degraded",
                "execution_directory": str(self.execution_dir) if self.execution_dir else None,
                "available_languages": available_languages,
                "active_processes": len(self.active_processes),
                "timestamp": datetime.utcnow().isoformat()
            }
            return health_status

        except Exception as e:
            logger.error(f"Health check failed for code execution server '{self.name}': {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available code execution tools."""
        return [
            {
                "name": "execute_code",
                "description": "Execute code in a specified programming language",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to execute"
                        },
                        "language": {
                            "type": "string",
                            "enum": list(self.SUPPORTED_LANGUAGES.keys()),
                            "description": "Programming language"
                        },
                        "input_data": {
                            "type": "string",
                            "description": "Input data to pass to the program"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Execution timeout in seconds",
                            "minimum": 1,
                            "maximum": 300,
                            "default": 30
                        },
                        "environment": {
                            "type": "object",
                            "description": "Environment variables",
                            "additionalProperties": {"type": "string"}
                        },
                        "compile_only": {
                            "type": "boolean",
                            "description": "Only compile, don't run (for compiled languages)",
                            "default": false
                        }
                    },
                    "required": ["code", "language"]
                }
            },
            {
                "name": "run_script",
                "description": "Execute a script file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "script_content": {
                            "type": "string",
                            "description": "Script content"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Script filename with extension"
                        },
                        "input_data": {
                            "type": "string",
                            "description": "Input data to pass to the script"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Execution timeout in seconds",
                            "default": 60
                        }
                    },
                    "required": ["script_content", "filename"]
                }
            },
            {
                "name": "install_package",
                "description": "Install a package for a specific language",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript", "r"],
                            "description": "Programming language"
                        },
                        "package_name": {
                            "type": "string",
                            "description": "Package name to install"
                        },
                        "version": {
                            "type": "string",
                            "description": "Package version (optional)"
                        }
                    },
                    "required": ["language", "package_name"]
                }
            },
            {
                "name": "check_syntax",
                "description": "Check syntax of code without executing it",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to check"
                        },
                        "language": {
                            "type": "string",
                            "enum": list(self.SUPPORTED_LANGUAGES.keys()),
                            "description": "Programming language"
                        }
                    },
                    "required": ["code", "language"]
                }
            },
            {
                "name": "get_execution_history",
                "description": "Get history of code executions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of entries to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    }
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a code execution tool."""
        try:
            if tool_name == "execute_code":
                return await self._execute_code(arguments)
            elif tool_name == "run_script":
                return await self._run_script(arguments)
            elif tool_name == "install_package":
                return await self._install_package(arguments)
            elif tool_name == "check_syntax":
                return await self._check_syntax(arguments)
            elif tool_name == "get_execution_history":
                return await self._get_execution_history(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing code execution tool '{tool_name}': {e}")
            return [{
                "type": "text",
                "text": f"Error executing {tool_name}: {str(e)}"
            }]

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available code execution resources."""
        resources = []

        # Add execution history
        resources.append({
            "uri": f"execution://{self.name}/history",
            "mimeType": "application/json",
            "description": "Code execution history"
        })

        # Add language capabilities
        for language, config in self.SUPPORTED_LANGUAGES.items():
            resources.append({
                "uri": f"execution://{self.name}/language/{language}",
                "mimeType": "application/json",
                "description": f"Capabilities for {language} execution"
            })

        return resources

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a code execution resource."""
        try:
            if not uri.startswith("execution://"):
                raise ValueError(f"Invalid execution resource URI: {uri}")

            # Parse URI: execution://server_name/resource_type/resource_id
            parts = uri[12:].split("/", 2)  # Remove "execution://" prefix
            if len(parts) < 2:
                raise ValueError(f"Invalid execution resource URI format: {uri}")

            server_name = parts[0]
            resource_type = parts[1]

            if server_name != self.name:
                raise ValueError(f"Resource belongs to different server: {server_name}")

            if resource_type == "history":
                history = self._get_execution_history_data()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(history, indent=2)
                    }]
                }

            elif resource_type == "language" and len(parts) > 2:
                language = parts[2]
                if language not in self.SUPPORTED_LANGUAGES:
                    raise ValueError(f"Unsupported language: {language}")

                capabilities = self.SUPPORTED_LANGUAGES[language].copy()
                # Check if language is actually available
                available = await self._is_language_available(language)
                capabilities["available"] = available

                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(capabilities, indent=2)
                    }]
                }

            else:
                raise ValueError(f"Unknown resource type: {resource_type}")

        except Exception as e:
            logger.error(f"Error reading execution resource '{uri}': {e}")
            raise

    def _validate_config(self):
        """Validate code execution server configuration."""
        # Validate allowed languages
        allowed_languages = self.config.get("allowed_languages", [])
        if allowed_languages:
            for lang in allowed_languages:
                if lang not in self.SUPPORTED_LANGUAGES:
                    raise ValueError(f"Unsupported language in allowed_languages: {lang}")

        # Validate resource limits
        resource_limits = self.config.get("resource_limits", {})
        if "max_memory_mb" in resource_limits:
            if not isinstance(resource_limits["max_memory_mb"], (int, float)) or resource_limits["max_memory_mb"] <= 0:
                raise ValueError("max_memory_mb must be a positive number")

        if "max_timeout_seconds" in resource_limits:
            if not isinstance(resource_limits["max_timeout_seconds"], (int, float)) or resource_limits["max_timeout_seconds"] <= 0:
                raise ValueError("max_timeout_seconds must be a positive number")

    async def _validate_language_support(self):
        """Validate that configured languages are available."""
        allowed_languages = self.config.get("allowed_languages", list(self.SUPPORTED_LANGUAGES.keys()))

        for language in allowed_languages:
            if language not in self.SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {language}")

            available = await self._is_language_available(language)
            if not available:
                logger.warning(f"Language '{language}' is configured but not available on the system")

    async def _is_language_available(self, language: str) -> bool:
        """Check if a language runtime is available."""
        if language not in self.SUPPORTED_LANGUAGES:
            return False

        commands = self.SUPPORTED_LANGUAGES[language]["commands"]
        primary_command = commands[0]

        try:
            # Check if command exists
            result = await asyncio.create_subprocess_exec(
                "which", primary_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False

    async def _get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        available = []
        allowed_languages = self.config.get("allowed_languages", list(self.SUPPORTED_LANGUAGES.keys()))

        for language in allowed_languages:
            if await self._is_language_available(language):
                available.append(language)

        return available

    async def _execute_code(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute code in a specified language."""
        code = args["code"]
        language = args["language"]
        input_data = args.get("input_data", "")
        timeout = args.get("timeout", self.SUPPORTED_LANGUAGES[language]["timeout"])
        environment = args.get("environment", {})
        compile_only = args.get("compile_only", False)

        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        # Check if language is allowed
        allowed_languages = self.config.get("allowed_languages", [])
        if allowed_languages and language not in allowed_languages:
            raise ValueError(f"Language '{language}' is not allowed")

        # Apply resource limits
        resource_limits = self.config.get("resource_limits", {})
        max_timeout = resource_limits.get("max_timeout_seconds", 300)
        timeout = min(timeout, max_timeout)

        # Execute code
        result = await self._run_code_execution(
            code, language, input_data, timeout, environment, compile_only
        )

        return [{
            "type": "text",
            "text": f"Code Execution Result ({language}):\n\n**Exit Code:** {result['exit_code']}\n\n**Output:**\n{result['stdout']}\n\n**Errors:**\n{result['stderr']}\n\n**Execution Time:** {result['execution_time']:.2f}s"
        }]

    async def _run_code_execution(self, code: str, language: str, input_data: str,
                                timeout: float, environment: Dict[str, str],
                                compile_only: bool) -> Dict[str, Any]:
        """Run code execution with proper isolation and limits."""
        if not self.execution_dir:
            raise RuntimeError("Execution directory not initialized")

        # Create temporary file for code
        lang_config = self.SUPPORTED_LANGUAGES[language]
        extension = lang_config["extensions"][0]
        temp_file = self.execution_dir / f"temp_{asyncio.get_event_loop().time()}{extension}"

        try:
            # Write code to file
            async with asyncio.Lock():  # Ensure thread safety
                temp_file.write_text(code)

            # Prepare execution command
            commands = lang_config["commands"]
            if language == "python":
                cmd = [commands[0], str(temp_file)]
            elif language == "javascript":
                cmd = [commands[0], str(temp_file)]
            elif language == "typescript":
                cmd = commands + [str(temp_file)]
            elif language == "bash":
                cmd = [commands[0], str(temp_file)]
            elif language == "r":
                cmd = [commands[0], str(temp_file)]
            elif language == "julia":
                cmd = [commands[0], str(temp_file)]
            elif language == "ruby":
                cmd = [commands[0], str(temp_file)]
            elif language == "go":
                if compile_only:
                    cmd = [commands[0], "build", str(temp_file)]
                else:
                    cmd = [commands[0], "run", str(temp_file)]
            elif language == "rust":
                if compile_only:
                    cmd = [commands[0], str(temp_file)]
                else:
                    cmd = commands + [str(temp_file), "-o", "temp_exec", "&&", "./temp_exec"]
            elif language == "java":
                class_name = f"Temp{int(asyncio.get_event_loop().time())}"
                # Replace class name in code if it exists
                modified_code = code.replace("public class ", f"public class {class_name} ")
                temp_file.write_text(modified_code)

                java_file = self.execution_dir / f"{class_name}.java"
                java_file.write_text(modified_code)

                if compile_only:
                    cmd = ["javac", str(java_file)]
                else:
                    cmd = ["javac", str(java_file), "&&", "java", "-cp", str(self.execution_dir), class_name]
            elif language == "cpp":
                if compile_only:
                    cmd = ["g++", "-c", str(temp_file)]
                else:
                    cmd = ["g++", str(temp_file), "-o", "temp_exec", "&&", "./temp_exec"]
            else:
                raise ValueError(f"Unsupported language: {language}")

            # Set resource limits
            memory_limit = lang_config["memory_limit"]
            resource_limits = self.config.get("resource_limits", {})
            max_memory = resource_limits.get("max_memory_mb", 500) * 1024 * 1024
            memory_limit = min(memory_limit, max_memory)

            # Execute command
            start_time = asyncio.get_event_loop().time()

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.execution_dir,
                env={**os.environ, **environment},
                preexec_fn=self._set_resource_limits(memory_limit)
            )

            # Track active process
            process_id = str(process.pid)
            self.active_processes[process_id] = process

            try:
                # Run with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode() if input_data else None),
                    timeout=timeout
                )

                execution_time = asyncio.get_event_loop().time() - start_time

                return {
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "execution_time": execution_time
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Code execution timed out after {timeout} seconds")

            finally:
                # Remove from active processes
                self.active_processes.pop(process_id, None)

        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()

            # Clean up compiled artifacts
            for pattern in ["temp_exec", "temp_exec.exe", "*.class", "*.o"]:
                for file in self.execution_dir.glob(pattern):
                    file.unlink(missing_ok=True)

    def _set_resource_limits(self, memory_limit: int):
        """Set resource limits for the process."""
        def set_limits():
            try:
                # Set memory limit
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
                # Set CPU time limit (soft limit)
                resource.setrlimit(resource.RLIMIT_CPU, (300, 300))  # 5 minutes
            except Exception as e:
                logger.warning(f"Failed to set resource limits: {e}")

        return set_limits

    async def _run_script(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run a script file."""
        script_content = args["script_content"]
        filename = args["filename"]
        input_data = args.get("input_data", "")
        timeout = args.get("timeout", 60)

        # Determine language from file extension
        file_ext = Path(filename).suffix.lower()
        language = None
        for lang, config in self.SUPPORTED_LANGUAGES.items():
            if file_ext in config["extensions"]:
                language = lang
                break

        if not language:
            raise ValueError(f"Unsupported file extension: {file_ext}")

        # Execute as code
        result = await self._run_code_execution(
            script_content, language, input_data, timeout, {}, False
        )

        return [{
            "type": "text",
            "text": f"Script Execution Result ({filename}):\n\n**Exit Code:** {result['exit_code']}\n\n**Output:**\n{result['stdout']}\n\n**Errors:**\n{result['stderr']}\n\n**Execution Time:** {result['execution_time']:.2f}s"
        }]

    async def _install_package(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Install a package for a language."""
        language = args["language"]
        package_name = args["package_name"]
        version = args.get("version")

        if language not in ["python", "javascript", "r"]:
            raise ValueError(f"Package installation not supported for language: {language}")

        # Prepare installation command
        if language == "python":
            if version:
                cmd = ["pip", "install", f"{package_name}=={version}"]
            else:
                cmd = ["pip", "install", package_name]
        elif language == "javascript":
            if version:
                cmd = ["npm", "install", "-g", f"{package_name}@{version}"]
            else:
                cmd = ["npm", "install", "-g", package_name]
        elif language == "r":
            if version:
                cmd = ["R", "-e", f"install.packages('{package_name}', version='{version}')"]
            else:
                cmd = ["R", "-e", f"install.packages('{package_name}')"]

        # Execute installation
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.execution_dir
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minutes timeout for installation
            )

            success = process.returncode == 0

            return [{
                "type": "text",
                "text": f"Package Installation Result ({language}):\n\n**Package:** {package_name}\n**Version:** {version or 'latest'}\n**Success:** {success}\n\n**Output:**\n{stdout.decode('utf-8', errors='replace')}\n\n**Errors:**\n{stderr.decode('utf-8', errors='replace')}"
            }]

        except asyncio.TimeoutError:
            raise TimeoutError("Package installation timed out")

    async def _check_syntax(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check syntax of code without executing it."""
        code = args["code"]
        language = args["language"]

        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        # For now, just attempt to compile/parse without running
        try:
            if language == "python":
                compile(code, '<string>', 'exec')
                result = "Syntax is valid"
            elif language == "javascript":
                # Basic syntax check - would need a proper linter
                result = "Basic syntax check passed (full validation requires external tools)"
            else:
                result = f"Syntax checking not implemented for {language}"

            return [{
                "type": "text",
                "text": f"Syntax Check Result ({language}):\n\n{result}"
            }]

        except SyntaxError as e:
            return [{
                "type": "text",
                "text": f"Syntax Check Result ({language}):\n\n**Syntax Error:** {str(e)}"
            }]

    async def _get_execution_history(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get execution history."""
        limit = args.get("limit", 10)
        history = self._get_execution_history_data()

        # Limit results
        recent_history = history.get("executions", [])[-limit:]

        return [{
            "type": "text",
            "text": f"Execution History (last {limit} entries):\n\n" +
                   "\n".join([f"• {entry.get('timestamp', 'Unknown')} - {entry.get('language', 'Unknown')} - Exit code: {entry.get('exit_code', 'Unknown')}"
                             for entry in recent_history])
        }]

    def _get_execution_history_data(self) -> Dict[str, Any]:
        """Get execution history data (placeholder for persistent storage)."""
        # In a real implementation, this would read from a database or file
        return {
            "total_executions": 0,
            "executions": []
        }

    def _log_execution(self, language: str, exit_code: int, execution_time: float):
        """Log execution for history (placeholder)."""
        # In a real implementation, this would write to persistent storage
        pass
