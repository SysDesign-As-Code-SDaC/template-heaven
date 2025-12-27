# Upstream Source Attribution

## Python MCP SDK Template

This template is based on the official Model Context Protocol Python SDK and incorporates best practices from the MCP ecosystem.

### Primary Sources

- **Official MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
  - License: MIT
  - Version: 1.18.0+
  - Core MCP protocol implementation

### Key Dependencies

- **FastAPI**: https://github.com/tiangolo/fastapi
  - License: MIT
  - Web framework for building APIs

- **SQLAlchemy**: https://github.com/sqlalchemy/sqlalchemy
  - License: MIT
  - Database ORM and toolkit

- **Pydantic**: https://github.com/pydantic/pydantic
  - License: MIT
  - Data validation and settings management

- **Prometheus Client**: https://github.com/prometheus/client_python
  - License: Apache 2.0
  - Metrics collection and monitoring

### Template Enhancements

This template extends the official MCP SDK with:

1. **Production-ready architecture** with containerization
2. **Database integration** with PostgreSQL and Redis
3. **Authentication and authorization** with JWT and RBAC
4. **Monitoring and observability** with Prometheus metrics
5. **Comprehensive testing** with pytest and coverage
6. **Development tools** with Jupyter, database admin, and debugging
7. **Security best practices** with input validation and audit logging
8. **Documentation and examples** for easy adoption

### License Compliance

All upstream dependencies are properly licensed and compatible with this template's MIT license. The template maintains proper attribution and follows open source best practices.

### Contributing Back

Contributions to this template that could benefit the broader MCP community are encouraged to be contributed back to the official MCP Python SDK repository.
