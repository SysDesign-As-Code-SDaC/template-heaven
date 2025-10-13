# React + Vite Template

A modern, fast React application template using Vite for build tooling and development.

## 🚀 Features

- **React 18** with latest features
- **Vite** for lightning-fast development
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **ESLint & Prettier** for code quality
- **Vitest** for testing
- **React Router** for routing
- **PWA** support (optional)

## 📋 Prerequisites

- Node.js 18+
- npm, yarn, or pnpm

## 🛠️ Quick Start

### 1. Create New Project

```bash
npm create vite@latest my-app -- --template react-ts
cd my-app
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Start Development Server

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## 📁 Project Structure

```
├── src/
│   ├── components/         # Reusable components
│   ├── pages/             # Page components
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # Utility functions
│   ├── styles/            # Global styles
│   ├── types/             # TypeScript type definitions
│   ├── App.tsx            # Main App component
│   ├── main.tsx           # Application entry point
│   └── vite-env.d.ts      # Vite type definitions
├── public/                # Static assets
└── tests/                 # Test files
```

## 🔧 Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run lint:fix     # Fix ESLint issues
npm run format       # Format code with Prettier
npm run test         # Run tests with Vitest
npm run test:ui      # Run tests with UI
npm run test:coverage # Run tests with coverage
```

## ⚡ Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    open: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
```

## 🎨 Styling with Tailwind CSS

```typescript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          900: '#1e3a8a',
        },
      },
    },
  },
  plugins: [],
}
```

## 🧪 Testing with Vitest

```typescript
// tests/App.test.tsx
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import App from '../src/App'

describe('App', () => {
  it('renders correctly', () => {
    render(<App />)
    expect(screen.getByText(/Vite \+ React/i)).toBeInTheDocument()
  })
})
```

## 🚀 Deployment

### Vercel

```bash
npm run build
# Deploy dist/ folder to Vercel
```

### Netlify

```bash
npm run build
# Deploy dist/ folder to Netlify
```

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 📚 Learning Resources

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Vitest Documentation](https://vitest.dev/)

## 🔗 Upstream Source

- **Repository**: [vitejs/vite](https://github.com/vitejs/vite)
- **Template**: [vitejs/vite/tree/main/packages/create-vite/template-react](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react)
- **Documentation**: [vitejs.dev](https://vitejs.dev/)
- **License**: MIT
