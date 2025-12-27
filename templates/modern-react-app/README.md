# {{ project_name | title }} - Modern React Application

{{ project_description }}

## 🚀 Features

This template provides a modern, production-ready React application with:

- **⚡ Vite** - Lightning-fast build tool and dev server
- **⚛️ React 18** - Latest React with concurrent features
- **📘 TypeScript** - Full type safety and IntelliSense
- **🎨 Tailwind CSS** - Utility-first CSS framework
- **🧪 Vitest** - Fast unit testing with Jest-compatible API
- **🔍 ESLint & Prettier** - Code quality and formatting
- **📦 pnpm** - Fast, disk space efficient package manager
- **🐳 Docker** - Containerized development and production
- **🔄 GitHub Actions**: automation examples (disabled)
- **📱 PWA Ready** - Progressive Web App capabilities
- **♿ Accessibility** - WCAG 2.1 AA compliant components
- **🌐 i18n Ready** - Internationalization support
- **📊 Analytics** - Built-in analytics and monitoring

## 🛠️ Tech Stack

- **Frontend**: React 18, TypeScript, Vite
- **Styling**: Tailwind CSS, Headless UI
- **Testing**: Vitest, Testing Library, Playwright
- **Linting**: ESLint, Prettier, TypeScript
- **Build**: Vite, Rollup
- **Package Manager**: pnpm
- **Deployment**: Docker, Vercel/Netlify ready

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm
- Git

### Installation

```bash
# Clone and setup
git clone <your-repo-url>
cd {{ project_name }}
pnpm install

# Start development server
pnpm dev

# Open http://localhost:5173
```

### Docker

```bash
# Development with Docker
docker-compose up --build

# Production build
docker build -t {{ project_name }} .
docker run -p 3000:3000 {{ project_name }}
```

## 📁 Project Structure

```
{{ project_name }}/
├── public/                       # Static assets
│   ├── favicon.ico
│   ├── manifest.json
│   └── robots.txt
├── src/                          # Source code
│   ├── components/               # Reusable components
│   │   ├── ui/                  # Base UI components
│   │   ├── forms/               # Form components
│   │   └── layout/              # Layout components
│   ├── pages/                   # Page components
│   ├── hooks/                   # Custom React hooks
│   ├── services/                # API services
│   ├── store/                   # State management
│   ├── utils/                   # Utility functions
│   ├── types/                   # TypeScript type definitions
│   ├── constants/               # Application constants
│   ├── assets/                  # Images, icons, etc.
│   ├── styles/                  # Global styles
│   ├── locales/                 # i18n translations
│   ├── App.tsx                  # Main App component
│   ├── main.tsx                 # Application entry point
│   └── vite-env.d.ts           # Vite type definitions
├── tests/                       # Test files
│   ├── components/              # Component tests
│   ├── pages/                   # Page tests
│   ├── utils/                   # Utility tests
│   └── e2e/                     # End-to-end tests
├── docs/                        # Documentation
├── .github/                     # Automation examples (GitHub Actions disabled)
├── docker/                      # Docker configurations
├── package.json                 # Dependencies and scripts
├── vite.config.ts              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
├── tsconfig.json               # TypeScript configuration
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local development
└── README.md                   # This file
```

## 🛠️ Development

### Available Scripts

```bash
# Development
pnpm dev              # Start development server
pnpm build            # Build for production
pnpm preview          # Preview production build

# Testing
pnpm test             # Run unit tests
pnpm test:ui          # Run tests with UI
pnpm test:coverage    # Run tests with coverage
pnpm test:e2e         # Run end-to-end tests

# Code Quality
pnpm lint             # Run ESLint
pnpm lint:fix         # Fix ESLint issues
pnpm format           # Format code with Prettier
pnpm type-check       # Run TypeScript type checking

# Build & Deploy
pnpm build            # Build for production
pnpm analyze          # Analyze bundle size
pnpm deploy           # Deploy to production
```

### Code Quality

```bash
# Run all quality checks
pnpm quality

# Pre-commit hooks (if using husky)
pnpm prepare
```

### Testing

```bash
# Unit tests
pnpm test

# Tests with coverage
pnpm test:coverage

# E2E tests
pnpm test:e2e

# Test specific file
pnpm test Button.test.tsx
```

## 🎨 Styling

This template uses **Tailwind CSS** for styling with a custom design system:

### Design Tokens

```typescript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          900: '#1e3a8a',
        },
        // ... more colors
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
}
```

### Component Styling

```tsx
// Example component with Tailwind
import { cn } from '@/utils/cn'

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  className?: string
  children: React.ReactNode
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  className,
  children,
  ...props
}) => {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center rounded-md font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        'disabled:pointer-events-none disabled:opacity-50',
        {
          'bg-primary text-primary-foreground hover:bg-primary/90': variant === 'primary',
          'bg-secondary text-secondary-foreground hover:bg-secondary/80': variant === 'secondary',
          'border border-input bg-background hover:bg-accent': variant === 'outline',
        },
        {
          'h-9 px-3 text-sm': size === 'sm',
          'h-10 px-4 py-2': size === 'md',
          'h-11 px-8 text-lg': size === 'lg',
        },
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
}
```

## 🧪 Testing

### Unit Testing with Vitest

```typescript
// tests/components/Button.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { Button } from '@/components/ui/Button'

describe('Button', () => {
  it('renders with correct text', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByRole('button', { name: 'Click me' })).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn()
    render(<Button onClick={handleClick}>Click me</Button>)
    
    fireEvent.click(screen.getByRole('button'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('applies correct variant classes', () => {
    render(<Button variant="secondary">Secondary</Button>)
    const button = screen.getByRole('button')
    expect(button).toHaveClass('bg-secondary')
  })
})
```

### E2E Testing with Playwright

```typescript
// tests/e2e/homepage.spec.ts
import { test, expect } from '@playwright/test'

test('homepage loads correctly', async ({ page }) => {
  await page.goto('/')
  
  await expect(page).toHaveTitle(/{{ project_name | title }}/)
  await expect(page.locator('h1')).toContainText('Welcome')
})

test('navigation works', async ({ page }) => {
  await page.goto('/')
  
  await page.click('text=About')
  await expect(page).toHaveURL('/about')
})
```

## 🚀 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production deployment
vercel --prod
```

### Netlify

```bash
# Build command
pnpm build

# Publish directory
dist

# Environment variables
VITE_API_URL=https://api.example.com
```

### Docker Production

```bash
# Build production image
docker build -f docker/Dockerfile.prod -t {{ project_name }}:prod .

# Run with production settings
docker run -d \
  --name {{ project_name }} \
  -p 3000:3000 \
  -e NODE_ENV=production \
  {{ project_name }}:prod
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```bash
# API Configuration
VITE_API_URL=http://localhost:8000/api
VITE_API_TIMEOUT=10000

# Analytics
VITE_GA_TRACKING_ID=GA-XXXXXXXXX
VITE_SENTRY_DSN=https://...

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_PWA=true

# Development
VITE_DEBUG=true
```

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    host: true,
  },
  build: {
    target: 'esnext',
    sourcemap: true,
  },
})
```

## 📱 PWA Features

This template includes Progressive Web App capabilities:

- **Service Worker** for offline functionality
- **Web App Manifest** for installability
- **Responsive Design** for all devices
- **Fast Loading** with optimized assets
- **Offline Support** with caching strategies

## ♿ Accessibility

Built with accessibility in mind:

- **WCAG 2.1 AA** compliant components
- **Keyboard Navigation** support
- **Screen Reader** friendly
- **Focus Management** for modals and forms
- **Color Contrast** compliance
- **Semantic HTML** structure

## 🌐 Internationalization

Ready for multiple languages:

```typescript
// src/locales/en.json
{
  "common": {
    "welcome": "Welcome to {{ project_name | title }}",
    "loading": "Loading...",
    "error": "Something went wrong"
  }
}

// Usage in components
import { useTranslation } from 'react-i18next'

const HomePage = () => {
  const { t } = useTranslation()
  
  return (
    <div>
      <h1>{t('common.welcome')}</h1>
    </div>
  )
}
```

## 📊 Analytics & Monitoring

Built-in analytics and monitoring:

- **Google Analytics** integration
- **Sentry** error tracking
- **Performance monitoring** with Web Vitals
- **User behavior** tracking
- **Custom events** and metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `pnpm quality`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Standards

- **TypeScript** for all new code
- **Tests** for all new functionality
- **Accessibility** compliance
- **Performance** optimization
- **Documentation** updates

## 📄 License

This project is licensed under the {{ license }} License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **React Team** for the amazing framework
- **Vite Team** for the fast build tool
- **Tailwind CSS** for the utility-first CSS
- **Template Heaven** for the modern template

---

**Built with ❤️ using Template Heaven Modern React Template**

*This template provides a solid foundation for building modern, scalable React applications with best practices and production-ready features.*
