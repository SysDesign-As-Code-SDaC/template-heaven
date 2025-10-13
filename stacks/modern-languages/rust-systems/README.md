# Rust Systems Programming Template

A production-ready Rust template for systems programming, featuring memory safety, zero-cost abstractions, and high-performance applications for 2025.

## 🚀 Features

- **Memory Safety** - Zero-cost memory safety without garbage collection
- **Concurrency** - Fearless concurrency with ownership and borrowing
- **Performance** - C/C++ level performance with modern language features
- **WebAssembly** - Compile to WASM for web and edge computing
- **Async Programming** - High-performance async/await with Tokio
- **Systems Programming** - Low-level system access and control
- **Cross-Platform** - Compile to multiple architectures and platforms
- **Package Management** - Cargo for dependency management
- **Testing** - Built-in testing framework
- **Documentation** - Integrated documentation generation

## 📋 Prerequisites

- Rust 1.70+
- Cargo
- Git

## 🛠️ Quick Start

### 1. Create New Rust Project

```bash
cargo new my-rust-system
cd my-rust-system
```

### 2. Add Dependencies

```toml
# Cargo.toml
[package]
name = "my-rust-system"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
reqwest = { version = "0.11", features = ["json"] }
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres"] }
```

### 3. Run Project

```bash
# Development
cargo run

# Release build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── main.rs                # Application entry point
│   ├── lib.rs                 # Library interface
│   ├── config/                # Configuration management
│   │   ├── mod.rs
│   │   └── settings.rs
│   ├── handlers/              # Request handlers
│   │   ├── mod.rs
│   │   ├── health.rs
│   │   └── api.rs
│   ├── services/              # Business logic
│   │   ├── mod.rs
│   │   ├── database.rs
│   │   └── external.rs
│   ├── models/                # Data models
│   │   ├── mod.rs
│   │   ├── user.rs
│   │   └── response.rs
│   ├── middleware/            # Middleware components
│   │   ├── mod.rs
│   │   ├── auth.rs
│   │   └── logging.rs
│   ├── utils/                 # Utility functions
│   │   ├── mod.rs
│   │   ├── crypto.rs
│   │   └── validation.rs
│   └── error.rs               # Error handling
├── tests/                     # Integration tests
├── benches/                   # Benchmark tests
├── examples/                  # Example implementations
├── docs/                      # Documentation
├── Cargo.toml                 # Package configuration
├── Cargo.lock                 # Dependency lock file
└── README.md                  # Project documentation
```

## 🔧 Available Scripts

```bash
# Development
cargo run                      # Run application
cargo run --release           # Run optimized build
cargo run --bin my-binary     # Run specific binary

# Building
cargo build                   # Build debug version
cargo build --release         # Build release version
cargo build --target wasm32-unknown-unknown # Build for WebAssembly

# Testing
cargo test                    # Run all tests
cargo test --lib              # Run library tests
cargo test --bins             # Run binary tests
cargo test --doc              # Run documentation tests
cargo test -- --nocapture     # Show output from tests

# Code Quality
cargo fmt                     # Format code
cargo clippy                  # Lint code
cargo audit                   # Security audit
cargo outdated                # Check for outdated dependencies

# Documentation
cargo doc                     # Generate documentation
cargo doc --open              # Generate and open documentation
cargo doc --no-deps           # Generate docs without dependencies
```

## 🦀 Rust Systems Programming Examples

### Async HTTP Server

```rust
// src/main.rs
use tokio::net::TcpListener;
use std::sync::Arc;
use anyhow::Result;

mod config;
mod handlers;
mod services;
mod models;
mod middleware;
mod utils;
mod error;

use config::Settings;
use handlers::api::ApiHandler;
use middleware::logging::LoggingMiddleware;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Load configuration
    let settings = Settings::new()?;
    
    // Initialize services
    let database = services::database::Database::new(&settings.database_url).await?;
    let external_api = services::external::ExternalApi::new(&settings.api_key);
    
    // Create shared state
    let state = Arc::new(AppState {
        database,
        external_api,
        settings: settings.clone(),
    });
    
    // Create handlers
    let api_handler = ApiHandler::new(state.clone());
    
    // Create middleware
    let logging_middleware = LoggingMiddleware::new();
    
    // Start server
    let listener = TcpListener::bind(&settings.server_address).await?;
    tracing::info!("Server listening on {}", settings.server_address);
    
    loop {
        let (stream, addr) = listener.accept().await?;
        let handler = api_handler.clone();
        let middleware = logging_middleware.clone();
        
        tokio::spawn(async move {
            if let Err(e) = middleware.handle_request(stream, addr, handler).await {
                tracing::error!("Error handling request: {}", e);
            }
        });
    }
}

#[derive(Clone)]
pub struct AppState {
    pub database: services::database::Database,
    pub external_api: services::external::ExternalApi,
    pub settings: Settings,
}
```

### Configuration Management

```rust
// src/config/settings.rs
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub server_address: String,
    pub database_url: String,
    pub api_key: String,
    pub log_level: String,
    pub max_connections: u32,
    pub timeout_seconds: u64,
}

impl Settings {
    pub fn new() -> anyhow::Result<Self> {
        let server_address = env::var("SERVER_ADDRESS")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string());
        
        let database_url = env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://localhost/myapp".to_string());
        
        let api_key = env::var("API_KEY")
            .expect("API_KEY environment variable is required");
        
        let log_level = env::var("LOG_LEVEL")
            .unwrap_or_else(|_| "info".to_string());
        
        let max_connections = env::var("MAX_CONNECTIONS")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100);
        
        let timeout_seconds = env::var("TIMEOUT_SECONDS")
            .unwrap_or_else(|_| "30".to_string())
            .parse()
            .unwrap_or(30);
        
        Ok(Settings {
            server_address,
            database_url,
            api_key,
            log_level,
            max_connections,
            timeout_seconds,
        })
    }
}
```

### Database Service

```rust
// src/services/database.rs
use sqlx::{PgPool, Row};
use anyhow::Result;
use crate::models::user::User;

pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        
        // Run migrations
        sqlx::migrate!("./migrations").run(&pool).await?;
        
        Ok(Database { pool })
    }
    
    pub async fn create_user(&self, user: &User) -> Result<User> {
        let row = sqlx::query!(
            r#"
            INSERT INTO users (id, name, email, created_at)
            VALUES ($1, $2, $3, $4)
            RETURNING id, name, email, created_at
            "#,
            user.id,
            user.name,
            user.email,
            user.created_at
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(User {
            id: row.id,
            name: row.name,
            email: row.email,
            created_at: row.created_at,
        })
    }
    
    pub async fn get_user(&self, id: &str) -> Result<Option<User>> {
        let row = sqlx::query!(
            "SELECT id, name, email, created_at FROM users WHERE id = $1",
            id
        )
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(row) = row {
            Ok(Some(User {
                id: row.id,
                name: row.name,
                email: row.email,
                created_at: row.created_at,
            }))
        } else {
            Ok(None)
        }
    }
    
    pub async fn list_users(&self, limit: i64, offset: i64) -> Result<Vec<User>> {
        let rows = sqlx::query!(
            "SELECT id, name, email, created_at FROM users LIMIT $1 OFFSET $2",
            limit,
            offset
        )
        .fetch_all(&self.pool)
        .await?;
        
        let users = rows
            .into_iter()
            .map(|row| User {
                id: row.id,
                name: row.name,
                email: row.email,
                created_at: row.created_at,
            })
            .collect();
        
        Ok(users)
    }
    
    pub async fn update_user(&self, user: &User) -> Result<User> {
        let row = sqlx::query!(
            r#"
            UPDATE users 
            SET name = $2, email = $3
            WHERE id = $1
            RETURNING id, name, email, created_at
            "#,
            user.id,
            user.name,
            user.email
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(User {
            id: row.id,
            name: row.name,
            email: row.email,
            created_at: row.created_at,
        })
    }
    
    pub async fn delete_user(&self, id: &str) -> Result<bool> {
        let result = sqlx::query!(
            "DELETE FROM users WHERE id = $1",
            id
        )
        .execute(&self.pool)
        .await?;
        
        Ok(result.rows_affected() > 0)
    }
}
```

### API Handlers

```rust
// src/handlers/api.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::models::user::User;
use crate::models::response::ApiResponse;
use crate::AppState;

#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    pub email: String,
}

#[derive(Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
}

#[derive(Deserialize)]
pub struct ListUsersQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

pub struct ApiHandler {
    state: std::sync::Arc<AppState>,
}

impl ApiHandler {
    pub fn new(state: std::sync::Arc<AppState>) -> Self {
        ApiHandler { state }
    }
    
    pub fn router(self) -> Router {
        Router::new()
            .route("/health", get(Self::health_check))
            .route("/users", post(Self::create_user))
            .route("/users/:id", get(Self::get_user))
            .route("/users/:id", put(Self::update_user))
            .route("/users/:id", delete(Self::delete_user))
            .route("/users", get(Self::list_users))
            .with_state(self.state)
    }
    
    pub async fn health_check() -> Json<ApiResponse<HashMap<String, String>>> {
        let mut data = HashMap::new();
        data.insert("status".to_string(), "healthy".to_string());
        data.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        
        Json(ApiResponse {
            success: true,
            data: Some(data),
            error: None,
        })
    }
    
    pub async fn create_user(
        State(state): State<std::sync::Arc<AppState>>,
        Json(payload): Json<CreateUserRequest>,
    ) -> Result<Json<ApiResponse<User>>, StatusCode> {
        let user = User {
            id: Uuid::new_v4().to_string(),
            name: payload.name,
            email: payload.email,
            created_at: chrono::Utc::now(),
        };
        
        match state.database.create_user(&user).await {
            Ok(created_user) => Ok(Json(ApiResponse {
                success: true,
                data: Some(created_user),
                error: None,
            })),
            Err(e) => {
                tracing::error!("Failed to create user: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
    
    pub async fn get_user(
        State(state): State<std::sync::Arc<AppState>>,
        Path(id): Path<String>,
    ) -> Result<Json<ApiResponse<User>>, StatusCode> {
        match state.database.get_user(&id).await {
            Ok(Some(user)) => Ok(Json(ApiResponse {
                success: true,
                data: Some(user),
                error: None,
            })),
            Ok(None) => Err(StatusCode::NOT_FOUND),
            Err(e) => {
                tracing::error!("Failed to get user: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
    
    pub async fn update_user(
        State(state): State<std::sync::Arc<AppState>>,
        Path(id): Path<String>,
        Json(payload): Json<UpdateUserRequest>,
    ) -> Result<Json<ApiResponse<User>>, StatusCode> {
        // Get existing user
        let existing_user = match state.database.get_user(&id).await {
            Ok(Some(user)) => user,
            Ok(None) => return Err(StatusCode::NOT_FOUND),
            Err(e) => {
                tracing::error!("Failed to get user: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        };
        
        // Update user
        let updated_user = User {
            id: existing_user.id,
            name: payload.name.unwrap_or(existing_user.name),
            email: payload.email.unwrap_or(existing_user.email),
            created_at: existing_user.created_at,
        };
        
        match state.database.update_user(&updated_user).await {
            Ok(user) => Ok(Json(ApiResponse {
                success: true,
                data: Some(user),
                error: None,
            })),
            Err(e) => {
                tracing::error!("Failed to update user: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
    
    pub async fn delete_user(
        State(state): State<std::sync::Arc<AppState>>,
        Path(id): Path<String>,
    ) -> Result<Json<ApiResponse<()>>, StatusCode> {
        match state.database.delete_user(&id).await {
            Ok(true) => Ok(Json(ApiResponse {
                success: true,
                data: None,
                error: None,
            })),
            Ok(false) => Err(StatusCode::NOT_FOUND),
            Err(e) => {
                tracing::error!("Failed to delete user: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
    
    pub async fn list_users(
        State(state): State<std::sync::Arc<AppState>>,
        Query(params): Query<ListUsersQuery>,
    ) -> Result<Json<ApiResponse<Vec<User>>>, StatusCode> {
        let limit = params.limit.unwrap_or(10);
        let offset = params.offset.unwrap_or(0);
        
        match state.database.list_users(limit, offset).await {
            Ok(users) => Ok(Json(ApiResponse {
                success: true,
                data: Some(users),
                error: None,
            })),
            Err(e) => {
                tracing::error!("Failed to list users: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}
```

### Error Handling

```rust
// src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("External API error: {0}")]
    ExternalApi(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    Configuration(#[from] config::ConfigError),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Authentication error: {0}")]
    Authentication(String),
    
    #[error("Authorization error: {0}")]
    Authorization(String),
    
    #[error("Internal server error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn status_code(&self) -> axum::http::StatusCode {
        match self {
            AppError::Database(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ExternalApi(_) => axum::http::StatusCode::BAD_GATEWAY,
            AppError::Serialization(_) => axum::http::StatusCode::BAD_REQUEST,
            AppError::Configuration(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Validation(_) => axum::http::StatusCode::BAD_REQUEST,
            AppError::Authentication(_) => axum::http::StatusCode::UNAUTHORIZED,
            AppError::Authorization(_) => axum::http::StatusCode::FORBIDDEN,
            AppError::Internal(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
```

### WebAssembly Integration

```rust
// src/wasm/mod.rs
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub struct WasmProcessor {
    data: Vec<f64>,
}

#[wasm_bindgen]
impl WasmProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmProcessor {
        WasmProcessor {
            data: Vec::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn add_data(&mut self, value: f64) {
        self.data.push(value);
    }
    
    #[wasm_bindgen]
    pub fn process_data(&self) -> JsValue {
        let result = self.data.iter().sum::<f64>() / self.data.len() as f64;
        JsValue::from_f64(result)
    }
    
    #[wasm_bindgen]
    pub fn get_data(&self) -> JsValue {
        JsValue::from_serde(&self.data).unwrap()
    }
}

#[wasm_bindgen]
pub fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[wasm_bindgen]
pub fn process_json(json_str: &str) -> String {
    #[derive(Deserialize, Serialize)]
    struct Data {
        values: Vec<f64>,
    }
    
    match serde_json::from_str::<Data>(json_str) {
        Ok(data) => {
            let sum: f64 = data.values.iter().sum();
            let avg = sum / data.values.len() as f64;
            
            let result = Data {
                values: data.values.into_iter().map(|v| v * avg).collect(),
            };
            
            serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
        }
        Err(_) => "{}".to_string(),
    }
}
```

### Performance Benchmarking

```rust
// benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use my_rust_system::utils::crypto::hash_data;
use my_rust_system::services::database::Database;

fn benchmark_hash_function(c: &mut Criterion) {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    c.bench_function("hash_data", |b| {
        b.iter(|| hash_data(black_box(&data)))
    });
}

fn benchmark_database_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let database = rt.block_on(async {
        Database::new("postgresql://localhost/test").await.unwrap()
    });
    
    c.bench_function("database_query", |b| {
        b.to_async(&rt).iter(|| async {
            database.list_users(black_box(10), black_box(0)).await
        })
    });
}

criterion_group!(benches, benchmark_hash_function, benchmark_database_operations);
criterion_main!(benches);
```

## 🧪 Testing

```rust
// tests/integration_tests.rs
use my_rust_system::services::database::Database;
use my_rust_system::models::user::User;
use uuid::Uuid;

#[tokio::test]
async fn test_create_user() {
    let database = Database::new("postgresql://localhost/test").await.unwrap();
    
    let user = User {
        id: Uuid::new_v4().to_string(),
        name: "Test User".to_string(),
        email: "test@example.com".to_string(),
        created_at: chrono::Utc::now(),
    };
    
    let created_user = database.create_user(&user).await.unwrap();
    assert_eq!(created_user.name, user.name);
    assert_eq!(created_user.email, user.email);
}

#[tokio::test]
async fn test_get_user() {
    let database = Database::new("postgresql://localhost/test").await.unwrap();
    
    let user = User {
        id: Uuid::new_v4().to_string(),
        name: "Test User".to_string(),
        email: "test@example.com".to_string(),
        created_at: chrono::Utc::now(),
    };
    
    let created_user = database.create_user(&user).await.unwrap();
    let retrieved_user = database.get_user(&created_user.id).await.unwrap().unwrap();
    
    assert_eq!(retrieved_user.id, created_user.id);
    assert_eq!(retrieved_user.name, created_user.name);
}
```

## 📚 Learning Resources

- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Async Book](https://rust-lang.github.io/async-book/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)

## 🔗 Upstream Source

- **Repository**: [rust-lang/rust](https://github.com/rust-lang/rust)
- **Cargo**: [rust-lang/cargo](https://github.com/rust-lang/cargo)
- **Tokio**: [tokio-rs/tokio](https://github.com/tokio-rs/tokio)
- **License**: MIT/Apache-2.0
