# Turborepo Monorepo Template

A high-performance monorepo template using Turborepo for build system and task orchestration, featuring modern tooling and best practices for 2025.

## 🚀 Features

- **Turborepo** - High-performance build system for JavaScript/TypeScript
- **pnpm Workspaces** - Fast, disk space efficient package manager
- **TypeScript** - Type safety across all packages
- **Shared Configuration** - ESLint, Prettier, and TypeScript configs
- **Independent Packages** - Apps and packages with separate dependencies
- **Incremental Builds** - Only rebuild what changed
- **Remote Caching** - Share build cache across team and CI
- **Docker Support** - Containerized development and deployment
- **CI/CD Ready** - GitHub Actions workflows included

## 📋 Prerequisites

- Node.js 18+
- pnpm 8+
- Git

## 🛠️ Quick Start

### 1. Create New Monorepo

```bash
npx create-turbo@latest my-monorepo
cd my-monorepo
```

### 2. Install Dependencies

```bash
pnpm install
```

### 3. Start Development

```bash
# Start all apps in development mode
pnpm dev

# Start specific app
pnpm dev --filter=web

# Build all packages
pnpm build
```

## 📁 Project Structure

```
├── apps/
│   ├── web/                    # Next.js web application
│   ├── docs/                   # Documentation site
│   └── mobile/                 # React Native app
├── packages/
│   ├── ui/                     # Shared UI components
│   ├── config/                 # Shared configurations
│   │   ├── eslint-config/      # ESLint configuration
│   │   ├── typescript-config/  # TypeScript configuration
│   │   └── tailwind-config/    # Tailwind CSS configuration
│   ├── utils/                  # Shared utilities
│   ├── database/               # Database schemas and migrations
│   └── api/                    # Shared API utilities
├── tools/
│   ├── eslint-config/          # Custom ESLint rules
│   └── scripts/                # Build and deployment scripts
├── turbo.json                  # Turborepo configuration
├── package.json                # Root package.json
└── pnpm-workspace.yaml         # pnpm workspace configuration
```

## 🔧 Available Scripts

```bash
# Development
pnpm dev                        # Start all apps in development
pnpm dev --filter=web          # Start specific app
pnpm dev --filter=...web       # Start web and its dependencies

# Building
pnpm build                      # Build all packages
pnpm build --filter=web        # Build specific package
pnpm build --filter=...web     # Build web and its dependencies

# Testing
pnpm test                       # Run all tests
pnpm test --filter=ui          # Test specific package
pnpm lint                       # Lint all packages
pnpm type-check                 # Type check all packages

# Package Management
pnpm add <package> --filter=web # Add dependency to specific package
pnpm add <package> -w          # Add dependency to workspace root
pnpm remove <package> --filter=web # Remove dependency from package

# Database
pnpm db:generate               # Generate Prisma client
pnpm db:push                   # Push schema to database
pnpm db:migrate                # Run database migrations
```

## ⚡ Turborepo Configuration

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "!.next/cache/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    },
    "lint": {
      "dependsOn": ["^build"]
    },
    "type-check": {
      "dependsOn": ["^build"]
    },
    "clean": {
      "cache": false
    }
  }
}
```

## 📦 Package Configuration

### Root Package.json

```json
{
  "name": "my-monorepo",
  "private": true,
  "scripts": {
    "build": "turbo build",
    "dev": "turbo dev",
    "lint": "turbo lint",
    "test": "turbo test",
    "type-check": "turbo type-check",
    "clean": "turbo clean",
    "format": "prettier --write \"**/*.{ts,tsx,md}\""
  },
  "devDependencies": {
    "@turbo/gen": "^1.10.12",
    "prettier": "^3.0.0",
    "turbo": "latest",
    "typescript": "^5.0.0"
  },
  "packageManager": "pnpm@8.0.0",
  "engines": {
    "node": ">=18"
  }
}
```

### App Package.json

```json
{
  "name": "web",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "build": "next build",
    "dev": "next dev",
    "lint": "next lint",
    "start": "next start"
  },
  "dependencies": {
    "@repo/ui": "workspace:*",
    "@repo/utils": "workspace:*",
    "next": "14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@repo/eslint-config": "workspace:*",
    "@repo/typescript-config": "workspace:*",
    "@types/node": "^20.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "eslint": "^8.0.0",
    "typescript": "^5.0.0"
  }
}
```

## 🎨 Shared UI Components

```typescript
// packages/ui/src/Button.tsx
import React from 'react';
import { cn } from '@repo/utils';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  children,
  className,
  onClick,
}) => {
  return (
    <button
      className={cn(
        'rounded-md font-medium transition-colors',
        {
          'bg-blue-600 text-white hover:bg-blue-700': variant === 'primary',
          'bg-gray-200 text-gray-900 hover:bg-gray-300': variant === 'secondary',
          'border border-gray-300 hover:bg-gray-50': variant === 'outline',
          'px-3 py-1.5 text-sm': size === 'sm',
          'px-4 py-2 text-base': size === 'md',
          'px-6 py-3 text-lg': size === 'lg',
        },
        className
      )}
      onClick={onClick}
    >
      {children}
    </button>
  );
};
```

## 🗄️ Shared Database Package

```typescript
// packages/database/src/index.ts
import { PrismaClient } from '@prisma/client';

export const prisma = new PrismaClient({
  log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
});

export * from '@prisma/client';
```

```prisma
// packages/database/prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
  output   = "../src/generated"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  posts     Post[]
}

model Post {
  id        String   @id @default(cuid())
  title     String
  content   String?
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}
```

## 🧪 Testing Strategy

```typescript
// packages/ui/src/__tests__/Button.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '../Button';

describe('Button', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    
    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('applies variant styles', () => {
    render(<Button variant="secondary">Secondary</Button>);
    const button = screen.getByText('Secondary');
    expect(button).toHaveClass('bg-gray-200');
  });
});
```

## 🚀 Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json pnpm-lock.yaml* ./
RUN corepack enable pnpm && pnpm i --frozen-lockfile

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the application
RUN corepack enable pnpm && pnpm build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/apps/web/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

### CI/CD with GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
    
    - name: Setup pnpm
      uses: pnpm/action-setup@v2
      with:
        version: 8
    
    - name: Get pnpm store directory
      shell: bash
      run: |
        echo "STORE_PATH=$(pnpm store path --silent)" >> $GITHUB_ENV
    
    - name: Setup pnpm cache
      uses: actions/cache@v3
      with:
        path: ${{ env.STORE_PATH }}
        key: ${{ runner.os }}-pnpm-store-${{ hashFiles('**/pnpm-lock.yaml') }}
        restore-keys: |
          ${{ runner.os }}-pnpm-store-
    
    - name: Install dependencies
      run: pnpm install --frozen-lockfile
    
    - name: Build packages
      run: pnpm build
    
    - name: Run tests
      run: pnpm test
    
    - name: Run linting
      run: pnpm lint
```

## 📚 Learning Resources

- [Turborepo Documentation](https://turbo.build/repo/docs)
- [pnpm Workspaces](https://pnpm.io/workspaces)
- [Monorepo Best Practices](https://monorepo.tools/)
- [Turborepo Examples](https://github.com/vercel/turbo/tree/main/examples)

## 🔗 Upstream Source

- **Repository**: [vercel/turbo](https://github.com/vercel/turbo)
- **Documentation**: [turbo.build](https://turbo.build/)
- **License**: MPL-2.0
