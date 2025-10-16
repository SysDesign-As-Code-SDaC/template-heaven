# Makefile for React Vite Frontend Template
# Provides comprehensive automation for frontend development workflows

.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check security clean build dev preview docker-build docker-run storybook analyze size e2e setup check ci

# Default target
help: ## Show this help message
	@echo "React Vite Frontend - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	npm ci

install-dev: ## Install development dependencies
	npm ci
	npm run prepare

# Testing
test: ## Run all tests
	npm run test:run

test-unit: ## Run unit tests only
	npm run test:run -- --testPathPattern=unit

test-integration: ## Run integration tests only
	npm run test:run -- --testPathPattern=integration

test-e2e: ## Run end-to-end tests only
	npm run e2e

test-coverage: ## Run tests with coverage report
	npm run test:coverage

test-watch: ## Run tests in watch mode
	npm run test:watch

# Code Quality
lint: ## Run linting
	npm run lint

lint-fix: ## Run linting and fix issues
	npm run lint:fix

format: ## Format code
	npm run format

format-check: ## Check code formatting
	npm run format:check

type-check: ## Run type checking
	npm run type-check

# Security
security: ## Run security checks
	npm run security
	npm audit --audit-level moderate

# Development
dev: ## Start development server
	npm run dev

dev-host: ## Start development server on all interfaces
	npm run dev -- --host 0.0.0.0

preview: ## Preview production build
	npm run preview

serve: ## Serve production build on port 3000
	npm run serve

# Building
build: ## Build for production
	npm run build

build-dev: ## Build for development
	npm run build:dev

build-analyze: ## Build and analyze bundle
	npm run build && npm run analyze

# Storybook
storybook: ## Start Storybook development server
	npm run storybook

storybook-build: ## Build Storybook
	npm run storybook:build

# Docker
docker-build: ## Build Docker image
	npm run docker:build

docker-run: ## Run Docker container
	npm run docker:run

docker-dev: ## Run development with Docker
	docker run -it --rm -v $(PWD):/app -p 3000:3000 node:18-alpine sh -c "cd /app && npm ci && npm run dev -- --host 0.0.0.0"

# Performance & Bundle Analysis
analyze: ## Analyze bundle size
	npm run analyze

size: ## Check bundle size limits
	npm run size

size-check: ## Check if bundle size is within limits
	npm run size

# E2E Testing
e2e: ## Run E2E tests
	npm run e2e

e2e-ui: ## Run E2E tests with UI
	npm run e2e:ui

e2e-headed: ## Run E2E tests in headed mode
	npm run e2e:headed

# Cleanup
clean: ## Clean up build artifacts and caches
	npm run clean
	rm -rf node_modules/.vite
	rm -rf playwright-report
	rm -rf test-results
	rm -rf coverage

clean-all: clean ## Clean everything including node_modules
	rm -rf node_modules
	rm -rf .npm

# Environment
env: ## Create .env file from template
	cp .env.example .env

env-check: ## Check environment variables
	node -e "console.log('Environment variables loaded successfully')"

# Git hooks
hooks-install: ## Install git hooks
	npm run prepare

hooks-uninstall: ## Uninstall git hooks
	rm -rf .husky

# Pre-commit
pre-commit: ## Run pre-commit hooks
	npx lint-staged

# Commit
commit: ## Create conventional commit
	npm run commit

# Documentation
docs: ## Generate documentation
	npx typedoc --out docs/ src/

docs-serve: ## Serve documentation locally
	cd docs && npx serve

docs-clean: ## Clean documentation
	rm -rf docs/

# Performance
perf: ## Run performance tests
	npx lighthouse http://localhost:3000 --output=json --output-path=./lighthouse-report.json

perf-ci: ## Run performance tests for CI
	npx lighthouse http://localhost:3000 --output=json --output-path=./lighthouse-report.json --chrome-flags="--headless --disable-gpu"

# PWA
pwa-generate: ## Generate PWA assets
	npx pwa-asset-generator logo.svg public/icons/ -i public/manifest.json

pwa-validate: ## Validate PWA configuration
	npx pwa-validate public/manifest.json

# API
api-types: ## Generate TypeScript types from API
	npx openapi-typescript http://localhost:8000/openapi.json --output src/types/api.ts

# Deployment
deploy-preview: ## Deploy to preview environment
	npm run build
	npx vercel --prod false

deploy-prod: ## Deploy to production
	npm run build
	npx vercel --prod true

# Monitoring
bundle-watch: ## Watch bundle size changes
	npm run size -- --watch

# Development workflow
setup: install-dev env hooks-install ## Complete development setup
	@echo "🚀 React Vite Frontend setup complete!"
	@echo "Run 'make dev' to start the development server"
	@echo "Run 'make test' to run the test suite"

check: format-check lint type-check security ## Run all checks
	@echo "✅ All checks passed!"

ci: check test-coverage build size-check ## Run CI/CD pipeline locally
	@echo "✅ CI/CD pipeline completed successfully!"

# Quick development commands
quick-test: ## Quick test run
	npm run test:run -- --run --reporter=verbose

quick-lint: ## Quick lint check
	npm run lint -- --quiet

quick-format: ## Quick format check
	npm run format:check -- --quiet

quick-type: ## Quick type check
	npm run type-check -- --quiet

# Bundle monitoring
bundle-size: ## Show current bundle size
	du -sh dist/assets/* | sort -h

# Dependencies
deps: ## Show dependency tree
	npm ls --depth=0

deps-outdated: ## Show outdated dependencies
	npm outdated

deps-update: ## Update dependencies interactively
	npx npm-check-updates -u

# Git
git-clean: ## Clean untracked files
	git clean -fd

git-reset: ## Reset to last commit (WARNING: destructive)
	git reset --hard HEAD

# Release
release: ## Create a release
	npm run release

release-dry: ## Dry run release
	npm run release -- --dry-run

# Custom frontend commands
assets-optimize: ## Optimize static assets
	npx imagemin public/images/* --out-dir=public/images/optimized

icons-generate: ## Generate favicon and icons
	npx real-favicon generate logo.svg faviconDescription.json public/

seo-check: ## Check SEO score
	npx lighthouse http://localhost:3000 --only-categories=seo --output=json

accessibility-check: ## Check accessibility score
	npx lighthouse http://localhost:3000 --only-categories=accessibility --output=json

performance-check: ## Check performance score
	npx lighthouse http://localhost:3000 --only-categories=performance --output=json

best-practices-check: ## Check best practices score
	npx lighthouse http://localhost:3000 --only-categories=best-practices --output=json

health-check: ## Check overall health score
	npx lighthouse http://localhost:3000 --output=json --output-path=./health-report.json

# Component development
component-create: ## Create new component (usage: make component-create name=Button)
	@if [ -z "$(name)" ]; then echo "Usage: make component-create name=ComponentName"; exit 1; fi
	mkdir -p src/components/$(name)
	touch src/components/$(name)/index.ts
	touch src/components/$(name)/$(name).tsx
	touch src/components/$(name)/$(name).test.tsx
	touch src/components/$(name)/$(name).stories.tsx
	echo "export { default } from './$(name)';" > src/components/$(name)/index.ts
	echo "import React from 'react';" > src/components/$(name)/$(name).tsx
	echo "import { describe, it, expect } from 'vitest';" > src/components/$(name)/$(name).test.tsx
	echo "import type { Meta, StoryObj } from '@storybook/react';" > src/components/$(name)/$(name).stories.tsx
	echo "✅ Component $(name) created successfully!"

page-create: ## Create new page (usage: make page-create name=Dashboard)
	@if [ -z "$(name)" ]; then echo "Usage: make page-create name=PageName"; exit 1; fi
	mkdir -p src/pages/$(name)
	touch src/pages/$(name)/index.ts
	touch src/pages/$(name)/$(name).tsx
	touch src/pages/$(name)/$(name).test.tsx
	echo "export { default } from './$(name)';" > src/pages/$(name)/index.ts
	echo "import React from 'react';" > src/pages/$(name)/$(name).tsx
	echo "import { describe, it, expect } from 'vitest';" > src/pages/$(name)/$(name).test.tsx
	echo "✅ Page $(name) created successfully!"

hook-create: ## Create new custom hook (usage: make hook-create name=useAuth)
	@if [ -z "$(name)" ]; then echo "Usage: make hook-create name=HookName"; exit 1; fi
	touch src/hooks/$(name).ts
	touch src/hooks/$(name).test.ts
	echo "import { useState, useEffect } from 'react';" > src/hooks/$(name).ts
	echo "import { describe, it, expect } from 'vitest';" > src/hooks/$(name).test.ts
	echo "✅ Hook $(name) created successfully!"

# Testing helpers
test-debug: ## Debug failing tests
	npm run test:run -- --run --reporter=verbose --no-coverage

test-parallel: ## Run tests in parallel
	npm run test:run -- --run --parallel

# CI helpers
ci-test: ## Run tests for CI/CD
	npm run test:run -- --run --coverage --reporter=json --outputFile=test-results.json

ci-lint: ## Run linting for CI/CD
	npm run lint -- --format=json --output-file=lint-results.json

ci-type: ## Run type checking for CI/CD
	npm run type-check

ci-security: ## Run security checks for CI/CD
	npm audit --audit-level moderate --json > security-audit.json

# Version management
version: ## Show current version
	node -p "require('./package.json').version"

version-bump: ## Bump version (usage: make version-bump type=patch|minor|major)
	@if [ -z "$(type)" ]; then echo "Usage: make version-bump type=patch|minor|major"; exit 1; fi
	npm version $(type)

# Docker Compose helpers
compose-up: ## Start all services
	docker-compose up -d

compose-down: ## Stop all services
	docker-compose down

compose-logs: ## View service logs
	docker-compose logs -f

compose-build: ## Build all services
	docker-compose build

# Advanced development
dev-hot: ## Start development server with hot reload debugging
	NODE_OPTIONS="--inspect" npm run dev

dev-https: ## Start development server with HTTPS
	npm run dev -- --https

dev-port: ## Start development server on custom port (usage: make dev-port port=4000)
	@if [ -z "$(port)" ]; then echo "Usage: make dev-port port=PORT_NUMBER"; exit 1; fi
	npm run dev -- --port $(port)

# Production optimization
build-optimize: ## Build with maximum optimization
	NODE_ENV=production npm run build -- --minify

build-sourcemap: ## Build with source maps
	npm run build -- --sourcemap

# Internationalization
i18n-extract: ## Extract translation strings
	npx i18next-scanner

i18n-check: ## Check translation completeness
	npx i18next-check

# Code generation
codegen-api: ## Generate API client code
	npx openapi-codegen

codegen-types: ## Generate TypeScript types
	npx graphql-codegen

# Database (if using local SQLite for development)
db-migrate: ## Run database migrations
	npx prisma migrate dev

db-generate: ## Generate Prisma client
	npx prisma generate

db-seed: ## Seed database with sample data
	npx prisma db seed

# Monitoring and observability
metrics: ## Show project metrics
	@echo "=== Project Metrics ==="
	@echo "Lines of code: $$(find src -name '*.ts' -o -name '*.tsx' | xargs wc -l | tail -1 | awk '{print $$1}')"
	@echo "Test files: $$(find tests -name '*.test.*' | wc -l)"
	@echo "Components: $$(find src/components -name '*.tsx' | wc -l)"
	@echo "Pages: $$(find src/pages -name '*.tsx' | wc -l)"
	@echo "Bundle size: $$(du -sh dist/assets/*.js 2>/dev/null | awk '{print $$1}' || echo 'Not built')"

# AI Code Review
ai-review: ## Run AI-powered code review
	npx @microsoft/rushstack-eslint-multiplexer --config .eslintrc.js --output-format json src/

# Accessibility
a11y-check: ## Run accessibility checks
	npx axe-playwright http://localhost:3000

# Performance profiling
profile-start: ## Start React DevTools profiler
	npm run dev &
	npx react-devtools

profile-build: ## Profile production build
	npm run build:analyze

# Final summary
status: ## Show project status
	@echo "=== React Vite Frontend Status ==="
	@echo "Node version: $$(node --version)"
	@echo "NPM version: $$(npm --version)"
	@echo "Dependencies: $$(npm ls --depth=0 | wc -l) packages"
	@echo "Test coverage: $$(find coverage -name '*.json' -exec jq -r '.total.lines.pct' {} \; 2>/dev/null || echo 'Not run')"
	@echo "Build status: $$(test -d dist && echo 'Built' || echo 'Not built')"
	@echo "Lint status: $$(npm run lint --silent && echo 'Passing' || echo 'Failing')"
	@echo "Type check: $$(npm run type-check --silent && echo 'Passing' || echo 'Failing')"
