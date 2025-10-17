"""
Monitoring and observability for MCP SDK Template.

Provides metrics collection, health checks, logging, and performance monitoring
for MCP server and client applications.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import psutil
import os

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client.core import CollectorRegistry

from .config import MonitoringSettings
from .exceptions import MCPSDKError


logger = logging.getLogger(__name__)


class MonitoringManager:
    """
    Monitoring and observability management.
    
    Provides metrics collection, health checks, performance monitoring,
    and system resource tracking.
    """
    
    def __init__(self, settings: MonitoringSettings):
        """
        Initialize monitoring manager.
        
        Args:
            settings: Monitoring configuration settings
        """
        self.settings = settings
        self.registry = CollectorRegistry()
        self._start_time = time.time()
        self._initialized = False
        
        # Metrics
        self._setup_metrics()
        
        # Performance tracking
        self._request_times = deque(maxlen=1000)
        self._error_counts = defaultdict(int)
        self._operation_counts = defaultdict(int)
        
        # Health check results
        self._health_checks = {}
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # Application info
        self.app_info = Info(
            'mcp_app_info',
            'Application information',
            registry=self.registry
        )
        
        # Request metrics
        self.request_count = Counter(
            'mcp_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'mcp_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Tool execution metrics
        self.tool_executions = Counter(
            'mcp_tool_executions_total',
            'Total number of tool executions',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        self.tool_duration = Histogram(
            'mcp_tool_duration_seconds',
            'Tool execution duration in seconds',
            ['tool_name'],
            registry=self.registry
        )
        
        # Resource metrics
        self.resource_reads = Counter(
            'mcp_resource_reads_total',
            'Total number of resource reads',
            ['resource_uri', 'status'],
            registry=self.registry
        )
        
        # Prompt metrics
        self.prompt_generations = Counter(
            'mcp_prompt_generations_total',
            'Total number of prompt generations',
            ['prompt_name', 'status'],
            registry=self.registry
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'mcp_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.connection_errors = Counter(
            'mcp_connection_errors_total',
            'Total number of connection errors',
            ['error_type'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'mcp_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'mcp_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'mcp_disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=self.registry
        )
        
        # Custom metrics
        self.custom_metrics = {}
    
    async def initialize(self) -> None:
        """
        Initialize monitoring manager.
        
        Raises:
            MCPSDKError: If initialization fails
        """
        try:
            logger.info("Initializing monitoring manager...")
            
            # Set application info
            self.app_info.info({
                'version': '1.0.0',
                'name': 'MCP SDK Template',
                'environment': os.getenv('ENVIRONMENT', 'development')
            })
            
            # Start background tasks
            if self.settings.enabled:
                asyncio.create_task(self._update_system_metrics())
                asyncio.create_task(self._perform_health_checks())
            
            self._initialized = True
            logger.info("Monitoring manager initialization complete")
            
        except Exception as e:
            logger.error(f"Monitoring manager initialization failed: {e}")
            raise MCPSDKError(f"Monitoring initialization failed: {str(e)}")
    
    async def _update_system_metrics(self) -> None:
        """Update system metrics periodically."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.labels(path='/').set(disk.used)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Failed to update system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_checks(self) -> None:
        """Perform periodic health checks."""
        while True:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self._health_checks['cpu'] = {
                    'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 95 else 'critical',
                    'value': cpu_percent,
                    'timestamp': datetime.utcnow()
                }
                
                self._health_checks['memory'] = {
                    'status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 95 else 'critical',
                    'value': memory.percent,
                    'timestamp': datetime.utcnow()
                }
                
                self._health_checks['disk'] = {
                    'status': 'healthy' if disk.percent < 80 else 'warning' if disk.percent < 95 else 'critical',
                    'value': disk.percent,
                    'timestamp': datetime.utcnow()
                }
                
                await asyncio.sleep(self.settings.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Record request metrics.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: Response status code
            duration: Request duration in seconds
        """
        if not self.settings.enabled:
            return
        
        # Record metrics
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Track performance
        self._request_times.append(duration)
        self._operation_counts[f"{method}:{endpoint}"] += 1
        
        if status_code >= 400:
            self._error_counts[f"{method}:{endpoint}"] += 1
    
    def record_tool_execution(
        self,
        tool_name: str,
        duration: float,
        success: bool
    ) -> None:
        """
        Record tool execution metrics.
        
        Args:
            tool_name: Name of the tool
            duration: Execution duration in seconds
            success: Whether execution was successful
        """
        if not self.settings.enabled:
            return
        
        status = 'success' if success else 'error'
        
        self.tool_executions.labels(
            tool_name=tool_name,
            status=status
        ).inc()
        
        self.tool_duration.labels(tool_name=tool_name).observe(duration)
    
    def record_resource_read(
        self,
        resource_uri: str,
        success: bool
    ) -> None:
        """
        Record resource read metrics.
        
        Args:
            resource_uri: Resource URI
            success: Whether read was successful
        """
        if not self.settings.enabled:
            return
        
        status = 'success' if success else 'error'
        
        self.resource_reads.labels(
            resource_uri=resource_uri,
            status=status
        ).inc()
    
    def record_prompt_generation(
        self,
        prompt_name: str,
        success: bool
    ) -> None:
        """
        Record prompt generation metrics.
        
        Args:
            prompt_name: Name of the prompt
            success: Whether generation was successful
        """
        if not self.settings.enabled:
            return
        
        status = 'success' if success else 'error'
        
        self.prompt_generations.labels(
            prompt_name=prompt_name,
            status=status
        ).inc()
    
    def record_connection_error(self, error_type: str) -> None:
        """
        Record connection error.
        
        Args:
            error_type: Type of connection error
        """
        if not self.settings.enabled:
            return
        
        self.connection_errors.labels(error_type=error_type).inc()
    
    def set_active_connections(self, count: int) -> None:
        """
        Set number of active connections.
        
        Args:
            count: Number of active connections
        """
        if not self.settings.enabled:
            return
        
        self.active_connections.set(count)
    
    def create_custom_metric(
        self,
        name: str,
        metric_type: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> None:
        """
        Create a custom metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric (counter, gauge, histogram)
            description: Metric description
            labels: Optional list of label names
        """
        if not self.settings.enabled:
            return
        
        labels = labels or []
        
        if metric_type == 'counter':
            metric = Counter(name, description, labels, registry=self.registry)
        elif metric_type == 'gauge':
            metric = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == 'histogram':
            metric = Histogram(name, description, labels, registry=self.registry)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self.custom_metrics[name] = metric
    
    def get_metrics(self) -> str:
        """
        Get Prometheus metrics in text format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry).decode('utf-8')
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Health status information
        """
        uptime = self.get_uptime()
        
        # Calculate overall health
        overall_status = 'healthy'
        if any(check['status'] == 'critical' for check in self._health_checks.values()):
            overall_status = 'critical'
        elif any(check['status'] == 'warning' for check in self._health_checks.values()):
            overall_status = 'warning'
        
        # Calculate performance metrics
        avg_response_time = 0
        if self._request_times:
            avg_response_time = sum(self._request_times) / len(self._request_times)
        
        total_requests = sum(self._operation_counts.values())
        total_errors = sum(self._error_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'status': overall_status,
            'uptime_seconds': uptime,
            'uptime_human': self._format_uptime(uptime),
            'health_checks': self._health_checks,
            'performance': {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate_percent': round(error_rate, 2),
                'avg_response_time_ms': round(avg_response_time * 1000, 2),
                'top_endpoints': self._get_top_endpoints()
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_uptime(self) -> float:
        """
        Get server uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self._start_time
    
    def _format_uptime(self, seconds: float) -> str:
        """
        Format uptime in human-readable format.
        
        Args:
            seconds: Uptime in seconds
            
        Returns:
            Formatted uptime string
        """
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _get_top_endpoints(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top endpoints by request count.
        
        Args:
            limit: Maximum number of endpoints to return
            
        Returns:
            List of endpoint statistics
        """
        sorted_endpoints = sorted(
            self._operation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                'endpoint': endpoint,
                'requests': count,
                'errors': self._error_counts.get(endpoint, 0),
                'error_rate': round(
                    (self._error_counts.get(endpoint, 0) / count * 100) if count > 0 else 0,
                    2
                )
            }
            for endpoint, count in sorted_endpoints[:limit]
        ]
    
    async def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        try:
            logger.info("Cleaning up monitoring manager...")
            # No specific cleanup needed for Prometheus metrics
            self._initialized = False
            logger.info("Monitoring manager cleanup complete")
            
        except Exception as e:
            logger.error(f"Monitoring cleanup failed: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if monitoring manager is initialized."""
        return self._initialized
