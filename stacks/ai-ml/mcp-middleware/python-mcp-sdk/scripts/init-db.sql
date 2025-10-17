-- Database initialization script for MCP SDK Template
-- This script sets up the initial database structure

-- Create databases for different services
CREATE DATABASE IF NOT EXISTS mcp_db;
CREATE DATABASE IF NOT EXISTS template_db;
CREATE DATABASE IF NOT EXISTS validation_db;
CREATE DATABASE IF NOT EXISTS generation_db;
CREATE DATABASE IF NOT EXISTS analysis_db;
CREATE DATABASE IF NOT EXISTS user_db;
CREATE DATABASE IF NOT EXISTS sync_db;

-- Create users with appropriate permissions
CREATE USER IF NOT EXISTS mcp_user WITH PASSWORD 'mcp_pass';
CREATE USER IF NOT EXISTS template_user WITH PASSWORD 'template_pass';
CREATE USER IF NOT EXISTS validation_user WITH PASSWORD 'validation_pass';
CREATE USER IF NOT EXISTS generation_user WITH PASSWORD 'generation_pass';
CREATE USER IF NOT EXISTS analysis_user WITH PASSWORD 'analysis_pass';
CREATE USER IF NOT EXISTS user_user WITH PASSWORD 'user_pass';
CREATE USER IF NOT EXISTS sync_user WITH PASSWORD 'sync_pass';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE mcp_db TO mcp_user;
GRANT ALL PRIVILEGES ON DATABASE template_db TO template_user;
GRANT ALL PRIVILEGES ON DATABASE validation_db TO validation_user;
GRANT ALL PRIVILEGES ON DATABASE generation_db TO generation_user;
GRANT ALL PRIVILEGES ON DATABASE analysis_db TO analysis_user;
GRANT ALL PRIVILEGES ON DATABASE user_db TO user_user;
GRANT ALL PRIVILEGES ON DATABASE sync_db TO sync_user;

-- Connect to main database and create tables
\c mcp_db;

-- Create basic tables for MCP operations
CREATE TABLE IF NOT EXISTS mcp_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS mcp_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    input_schema JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mcp_resources (
    id SERIAL PRIMARY KEY,
    uri VARCHAR(500) UNIQUE NOT NULL,
    name VARCHAR(255),
    description TEXT,
    mime_type VARCHAR(100),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mcp_prompts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    arguments JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_mcp_sessions_user_id ON mcp_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_sessions_last_activity ON mcp_sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX IF NOT EXISTS idx_mcp_resources_uri ON mcp_resources(uri);
CREATE INDEX IF NOT EXISTS idx_mcp_prompts_name ON mcp_prompts(name);

-- Insert default data
INSERT INTO mcp_tools (name, description, input_schema) VALUES
('echo', 'Echo tool for testing MCP functionality', '{"type": "object", "properties": {"message": {"type": "string", "description": "Message to echo"}}}'),
('health', 'Get server health status', '{"type": "object", "properties": {}}')
ON CONFLICT (name) DO NOTHING;

INSERT INTO mcp_resources (uri, name, description, mime_type, content) VALUES
('config://server', 'Server Configuration', 'Current server configuration', 'text/plain', 'Server configuration will be populated at runtime')
ON CONFLICT (uri) DO NOTHING;

INSERT INTO mcp_prompts (name, description, arguments) VALUES
('help', 'Get help information about the MCP server', '[{"name": "topic", "description": "Help topic", "required": false}]')
ON CONFLICT (name) DO NOTHING;
