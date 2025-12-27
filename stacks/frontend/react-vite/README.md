# React Vite Frontend Template

A comprehensive React frontend template with TypeScript, Vite, modern tooling, and production-ready deployment. This template follows all gold standard practices for automated repo management, testing, documentation, and deployment.

## 🚀 Features

### Core Features
- **React 18** with modern hooks and concurrent features
- **TypeScript** for type-safe development
- **Vite** for lightning-fast development and building
- **React Router** for client-side routing
- **Tailwind CSS** for utility-first styling
- **Zustand** for lightweight state management
- **React Query** for server state management
- **React Hook Form + Zod** for form validation
- **Comprehensive testing** with Vitest and Playwright

### Development & Quality Assurance
- **Hot Module Replacement** for instant updates
- **ESLint + Prettier** for code quality and formatting
- **TypeScript strict mode** for type safety
- **Vitest** for fast unit testing
- **Playwright** for E2E testing
- **Storybook** for component development
- **Bundle analyzer** for optimization
- **Size limits** for performance monitoring

### Production & Deployment
- **Optimized builds** with code splitting and tree shaking
- **Progressive Web App** support with service workers
- **Docker containerization** for consistent deployment
- **automation pipelines** examples (GitHub Actions disabled)
- **Performance monitoring** with Lighthouse
- **SEO optimization** with React Helmet
- **Accessibility** compliance (WCAG guidelines)

### Developer Experience
- **Comprehensive Makefile** with 50+ development commands
- **Pre-commit hooks** for automatic quality checks
- **Conventional commits** with commitizen
- **Semantic versioning** with automated releases
- **Interactive development** with Storybook
- **API integration** with automatic type generation
- **Error boundaries** and comprehensive error handling

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Testing](#-testing)
- [Building & Deployment](#-building--deployment)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm 9+ (or yarn/pnpm)
- Git

### Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd react-vite-frontend
   ```

2. **Install dependencies**
   ```bash
   make install-dev
   ```

3. **Setup environment**
   ```bash
   make env  # Creates .env from template
   # Edit .env with your API endpoints and configuration
   ```

4. **Start development server**
   ```bash
   make dev
   ```

The application will be available at `http://localhost:3000` with hot reload enabled.

### Docker Setup (Recommended)

```bash
# Start development with Docker
make docker-dev

# Or build and run production container
make docker-build
make docker-run
```

## 📁 Project Structure

```
react-vite-frontend/
├── src/                          # Source code
│   ├── components/               # Reusable UI components
│   │   ├── ui/                  # Base UI components
│   │   ├── forms/               # Form components
│   │   └── layout/              # Layout components
│   ├── pages/                   # Page components
│   │   ├── auth/                # Authentication pages
│   │   ├── dashboard/           # Dashboard pages
│   │   └── public/              # Public pages
│   ├── hooks/                   # Custom React hooks
│   ├── utils/                   # Utility functions
│   ├── services/                # API services and clients
│   ├── types/                   # TypeScript type definitions
│   ├── context/                 # React context providers
│   ├── styles/                  # Global styles and Tailwind config
│   ├── constants/               # Application constants
│   └── lib/                     # Third-party library configurations
├── tests/                       # Test files
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── public/                      # Static assets
│   ├── icons/                   # PWA icons
│   ├── images/                  # Static images
│   └── manifest.json           # PWA manifest
├── docs/                        # Documentation
├── scripts/                     # Build and utility scripts
├── .github/workflows/          # automation pipelines
├── .cursor/rules/              # AI coding agent rules
├── package.json                # Dependencies and scripts
├── vite.config.ts              # Vite configuration
├── tailwind.config.js          # Tailwind CSS configuration
├── tsconfig.json               # TypeScript configuration
├── .eslintrc.js               # ESLint configuration
├── .prettierrc                # Prettier configuration
├── playwright.config.ts       # Playwright E2E configuration
├── vitest.config.ts           # Vitest configuration
├── Dockerfile                 # Docker containerization
├── Makefile                   # Development automation
└── README.md                  # This file
```

## 🛠️ Development

### Development Commands

```bash
# Start development server
make dev

# Start with custom host/port
make dev-host

# Run linting
make lint

# Fix linting issues
make lint-fix

# Format code
make format

# Check formatting
make format-check

# Type checking
make type-check

# Run security checks
make security
```

### Code Quality

The template includes comprehensive code quality tooling:

- **ESLint** with TypeScript and React rules
- **Prettier** for consistent code formatting
- **TypeScript** with strict type checking
- **Pre-commit hooks** for automatic quality checks
- **Husky** for git hook management

### Component Development

```bash
# Create new component
make component-create name=Button

# Create new page
make page-create name=Dashboard

# Create custom hook
make hook-create name=useAuth

# Start Storybook
make storybook
```

## 🧪 Testing

### Testing Strategy

The template includes multiple layers of testing:

- **Unit Tests**: Component and utility function testing with Vitest
- **Integration Tests**: API integration and component interaction testing
- **E2E Tests**: Full user workflow testing with Playwright

### Running Tests

```bash
# Run all tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e          # End-to-end tests only

# Run tests with coverage
make test-coverage

# Run tests in watch mode
make test-watch

# Run E2E tests with UI
make e2e-ui
```

### Test Configuration

- **Vitest** for fast unit testing with React Testing Library
- **Playwright** for cross-browser E2E testing
- **MSW** for API mocking in tests
- **Coverage reporting** with detailed reports

## 🏗️ Building & Deployment

### Build Commands

```bash
# Production build
make build

# Development build
make build-dev

# Build with bundle analysis
make build-analyze

# Check bundle size
make size
```

### Deployment Options

#### Vercel (Recommended)
```bash
npm i -g vercel
make deploy-prod
```

#### Docker
```bash
make docker-build
make docker-run
```

#### Static Hosting
```bash
make build
# Deploy dist/ folder to any static hosting service
```

### Environment Configuration

Create `.env` file with:

```bash
# API Configuration
VITE_API_URL=https://api.example.com
VITE_API_TIMEOUT=10000

# Authentication
VITE_AUTH_REDIRECT_URL=https://app.example.com/auth/callback

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_PWA=true

# Third-party Services
VITE_GOOGLE_ANALYTICS_ID=GA_MEASUREMENT_ID
VITE_SENTRY_DSN=https://sentry-dsn
```

## ⚙️ Configuration

### Vite Configuration

Advanced Vite configuration includes:
- **Path aliases** for clean imports
- **Environment variables** handling
- **PWA support** with service workers
- **Bundle optimization** and code splitting
- **Development proxy** for API calls

### Tailwind CSS

Custom Tailwind configuration with:
- **Design tokens** for consistent theming
- **Custom utilities** for common patterns
- **Dark mode** support
- **Responsive breakpoints** customization

### TypeScript

Strict TypeScript configuration with:
- **Path mapping** for absolute imports
- **Strict mode** enabled
- **ESLint integration** for additional rules
- **Declaration files** generation

## 📊 Performance

### Performance Monitoring

```bash
# Analyze bundle
make analyze

# Check size limits
make size

# Lighthouse performance audit
make performance-check

# Accessibility audit
make accessibility-check

# SEO audit
make seo-check
```

### Optimization Features

- **Code splitting** with dynamic imports
- **Lazy loading** for routes and components
- **Image optimization** with responsive images
- **Font loading** optimization
- **Critical CSS** inlining
- **Service worker** for caching

### Bundle Analysis

The template includes bundle analysis tools to:
- **Identify large dependencies**
- **Monitor bundle size growth**
- **Optimize chunk splitting**
- **Tree shake unused code**

## 🐳 Docker Development

### Development Container

```bash
# Run development environment
make docker-dev

# View container logs
make compose-logs

# Stop containers
make compose-down
```

### Production Container

```bash
# Build production image
make docker-build

# Run production container
make docker-run
```

## 📱 Progressive Web App

### PWA Features

- **Service Worker** for offline functionality
- **Web App Manifest** for installation
- **Push notifications** support
- **Background sync** for data synchronization

### PWA Commands

```bash
# Generate PWA assets
make pwa-generate

# Validate PWA configuration
make pwa-validate
```

## 🤖 AI Coding Agent Support

This template includes comprehensive AI coding agent support:

- **Cursor Rules**: Located in `.cursor/rules/`
- **Component Generation**: AI-assisted component creation
- **Code Review**: Automated code quality checks
- **Testing**: AI-generated test cases
- **Documentation**: Auto-generated component docs

### AI Development Setup

```bash
# AI context is automatically configured
# Cursor will use the rules in .cursor/rules/
```

## 📈 Advanced Features

### State Management

- **Zustand** for global state with TypeScript support
- **React Query** for server state management
- **Context API** for theme and user preferences

### Form Handling

- **React Hook Form** for performant form management
- **Zod** for schema validation
- **Custom form components** with validation

### API Integration

- **Axios** with interceptors for authentication
- **React Query** for caching and synchronization
- **Automatic TypeScript types** generation from API

### Error Handling

- **Error boundaries** for graceful error handling
- **Toast notifications** for user feedback
- **Logging** with structured error reporting

## 🎨 Styling

### Design System

- **Tailwind CSS** for utility-first styling
- **Custom design tokens** for consistent theming
- **Dark mode** support with CSS variables
- **Responsive design** with mobile-first approach

### Component Library

- **Headless UI** for accessible components
- **Radix UI** for unstyled, accessible components
- **Lucide React** for consistent iconography
- **Custom component variants** with Tailwind

## 🔧 Available Commands

Run `make help` to see all available commands:

```bash
make help
```

### Development Workflow

```bash
# Complete setup
make setup

# Development cycle
make dev          # Start development server
make test         # Run tests
make lint         # Check code quality
make build        # Build for production

# Quality assurance
make check        # Run all checks
make ci          # Run automation pipeline locally
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests and checks**
   ```bash
   make check
   make test
   ```
5. **Create conventional commit**
   ```bash
   make commit
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- **TypeScript** for all new code
- **Component composition** over inheritance
- **Custom hooks** for shared logic
- **Atomic design** for component organization
- **Accessibility first** approach
- **Performance conscious** development

### Code Standards

- **ESLint** configuration with React and TypeScript rules
- **Prettier** for consistent formatting
- **Conventional commits** for clear git history
- **Semantic versioning** for releases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **React** for the excellent frontend framework
- **Vite** for the blazing fast build tool
- **Tailwind CSS** for the utility-first CSS framework
- **TypeScript** for type-safe JavaScript development
- **Vitest** for fast and reliable testing
- **Playwright** for cross-browser testing
- **Storybook** for component development

---

**Built with ❤️ using Template Heaven's gold standard practices**

For more information, visit our [GitHub repository](https://github.com/template-heaven/templateheaven).
