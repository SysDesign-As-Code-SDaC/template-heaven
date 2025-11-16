# {{ project_name | title }} - Next.js Fullstack Application

{{ project_description }}

## 🚀 Features

This template provides a production-ready Next.js fullstack application with:

- **⚡ Next.js 14** - Latest Next.js with App Router and React Server Components
- **📘 TypeScript** - Full type safety throughout the stack
- **🎨 Tailwind CSS** - Utility-first CSS with custom design system
- **🗄️ Prisma** - Type-safe database ORM with migrations
- **🔐 NextAuth.js** - Complete authentication solution
- **📊 tRPC** - End-to-end typesafe APIs
- **🧪 Testing** - Jest, Testing Library, and Playwright
- **📝 ESLint & Prettier** - Code quality and formatting
- **🐳 Docker** - Containerized development and production
-- **🔄 GitHub Actions**: automation examples (disabled)
- **📱 Responsive Design** - Mobile-first responsive design
- **♿ Accessibility** - WCAG 2.1 AA compliant
- **🌐 i18n Ready** - Internationalization support
- **📈 Analytics** - Built-in analytics and monitoring
- **🛡️ Security** - Security headers, CSRF protection, input validation

## 🛠️ Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes, tRPC, Prisma
- **Database**: PostgreSQL (with SQLite for development)
- **Authentication**: NextAuth.js with multiple providers
- **Styling**: Tailwind CSS, Headless UI, Framer Motion
- **Testing**: Jest, Testing Library, Playwright
- **Deployment**: Vercel, Docker, Kubernetes ready
- **Monitoring**: Vercel Analytics, Sentry

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- pnpm (recommended) or npm
- PostgreSQL (for production)
- Git

### Installation

```bash
# Clone and setup
git clone <your-repo-url>
cd {{ project_name }}
pnpm install

# Setup environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Setup database
pnpm db:push
pnpm db:seed

# Start development server
pnpm dev

# Open http://localhost:3000
```

### Docker

```bash
# Development with Docker Compose
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
│   └── images/
├── src/                          # Source code
│   ├── app/                      # Next.js App Router
│   │   ├── (auth)/              # Auth route group
│   │   │   ├── login/
│   │   │   └── register/
│   │   ├── (dashboard)/         # Dashboard route group
│   │   │   ├── dashboard/
│   │   │   └── profile/
│   │   ├── api/                 # API routes
│   │   │   ├── auth/
│   │   │   ├── trpc/
│   │   │   └── health/
│   │   ├── globals.css          # Global styles
│   │   ├── layout.tsx           # Root layout
│   │   ├── page.tsx             # Home page
│   │   └── loading.tsx          # Loading UI
│   ├── components/              # Reusable components
│   │   ├── ui/                  # Base UI components
│   │   ├── forms/               # Form components
│   │   ├── layout/              # Layout components
│   │   └── providers/           # Context providers
│   ├── lib/                     # Utility libraries
│   │   ├── auth.ts              # Auth configuration
│   │   ├── db.ts                # Database connection
│   │   ├── trpc.ts              # tRPC configuration
│   │   ├── utils.ts             # Utility functions
│   │   └── validations.ts       # Zod schemas
│   ├── server/                  # Server-side code
│   │   ├── api/                 # tRPC routers
│   │   ├── auth.ts              # Auth utilities
│   │   └── db/                  # Database queries
│   ├── styles/                  # Additional styles
│   └── types/                   # TypeScript types
├── prisma/                      # Database schema and migrations
│   ├── migrations/
│   ├── schema.prisma
│   └── seed.ts
├── tests/                       # Test files
│   ├── __mocks__/
│   ├── components/
│   ├── pages/
│   └── e2e/
├── docs/                        # Documentation
├── docker/                      # Docker configurations
├── .github/                     # Automation examples (GitHub Actions disabled)
├── package.json                 # Dependencies and scripts
├── next.config.js               # Next.js configuration
├── tailwind.config.js           # Tailwind configuration
├── tsconfig.json                # TypeScript configuration
├── Dockerfile                   # Container image
├── docker-compose.yml           # Local development
└── README.md                    # This file
```

## 🛠️ Development

### Available Scripts

```bash
# Development
pnpm dev              # Start development server
pnpm build            # Build for production
pnpm start            # Start production server
pnpm preview          # Preview production build

# Database
pnpm db:push          # Push schema changes to database
pnpm db:migrate       # Run database migrations
pnpm db:seed          # Seed database with sample data
pnpm db:studio        # Open Prisma Studio
pnpm db:generate      # Generate Prisma client

# Testing
pnpm test             # Run unit tests
pnpm test:watch       # Run tests in watch mode
pnpm test:coverage    # Run tests with coverage
pnpm test:e2e         # Run end-to-end tests
pnpm test:e2e:ui      # Run E2E tests with UI

# Code Quality
pnpm lint             # Run ESLint
pnpm lint:fix         # Fix ESLint issues
pnpm format           # Format code with Prettier
pnpm type-check       # Run TypeScript type checking

# Build & Deploy
pnpm build            # Build for production
pnpm analyze          # Analyze bundle size
pnpm deploy           # Deploy to Vercel
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

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```bash
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/{{ project_name }}"
DIRECT_URL="postgresql://user:password@localhost:5432/{{ project_name }}"

# NextAuth.js
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key-here"

# OAuth Providers (optional)
GOOGLE_CLIENT_ID="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"
GITHUB_CLIENT_ID="your-github-client-id"
GITHUB_CLIENT_SECRET="your-github-client-secret"

# Email (optional)
EMAIL_SERVER_HOST="smtp.gmail.com"
EMAIL_SERVER_PORT=587
EMAIL_SERVER_USER="your-email@gmail.com"
EMAIL_SERVER_PASSWORD="your-app-password"
EMAIL_FROM="noreply@{{ project_name }}.com"

# Analytics (optional)
NEXT_PUBLIC_GA_ID="GA-XXXXXXXXX"
SENTRY_DSN="https://your-sentry-dsn@sentry.io/project-id"

# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS="true"
NEXT_PUBLIC_ENABLE_SENTRY="true"
```

### Next.js Configuration

```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost', 'example.com'],
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig
```

## 🗄️ Database

### Prisma Schema

```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id            String    @id @default(cuid())
  email         String    @unique
  emailVerified DateTime?
  name          String?
  image         String?
  accounts      Account[]
  sessions      Session[]
  posts         Post[]
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  @@map("users")
}

model Account {
  id                String  @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String? @db.Text
  access_token      String? @db.Text
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String? @db.Text
  session_state     String?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
  @@map("accounts")
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@map("sessions")
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

  @@map("posts")
}
```

### Database Operations

```bash
# Generate Prisma client
pnpm db:generate

# Push schema changes
pnpm db:push

# Create and apply migration
pnpm db:migrate

# Seed database
pnpm db:seed

# Open Prisma Studio
pnpm db:studio
```

## 🔐 Authentication

### NextAuth.js Configuration

```typescript
// src/lib/auth.ts
import { NextAuthOptions } from "next-auth"
import { PrismaAdapter } from "@next-auth/prisma-adapter"
import GoogleProvider from "next-auth/providers/google"
import GitHubProvider from "next-auth/providers/github"
import CredentialsProvider from "next-auth/providers/credentials"
import { db } from "@/lib/db"
import { compare } from "bcryptjs"

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(db),
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }),
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        const user = await db.user.findUnique({
          where: { email: credentials.email }
        })

        if (!user || !user.password) {
          return null
        }

        const isPasswordValid = await compare(credentials.password, user.password)

        if (!isPasswordValid) {
          return null
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
        }
      }
    })
  ],
  session: {
    strategy: "jwt"
  },
  pages: {
    signIn: "/auth/signin",
    signUp: "/auth/signup",
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id
      }
      return token
    },
    async session({ session, token }) {
      if (token) {
        session.user.id = token.id as string
      }
      return session
    }
  }
}
```

## 📊 tRPC API

### Router Definition

```typescript
// src/server/api/root.ts
import { createTRPCRouter } from "@/server/api/trpc"
import { postRouter } from "./routers/post"
import { userRouter } from "./routers/user"

export const appRouter = createTRPCRouter({
  post: postRouter,
  user: userRouter,
})

export type AppRouter = typeof appRouter
```

### Example Router

```typescript
// src/server/api/routers/post.ts
import { z } from "zod"
import { createTRPCRouter, protectedProcedure, publicProcedure } from "@/server/api/trpc"

export const postRouter = createTRPCRouter({
  getAll: publicProcedure.query(({ ctx }) => {
    return ctx.db.post.findMany({
      include: {
        author: true,
      },
    })
  }),

  create: protectedProcedure
    .input(z.object({
      title: z.string(),
      content: z.string().optional(),
    }))
    .mutation(({ ctx, input }) => {
      return ctx.db.post.create({
        data: {
          title: input.title,
          content: input.content,
          authorId: ctx.session.user.id,
        },
      })
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(({ ctx, input }) => {
      return ctx.db.post.delete({
        where: { id: input.id },
      })
    }),
})
```

### Client Usage

```typescript
// src/components/PostList.tsx
import { api } from "@/lib/trpc"

export function PostList() {
  const { data: posts, isLoading } = api.post.getAll.useQuery()

  if (isLoading) return <div>Loading...</div>

  return (
    <div>
      {posts?.map((post) => (
        <div key={post.id}>
          <h3>{post.title}</h3>
          <p>{post.content}</p>
          <p>By {post.author.name}</p>
        </div>
      ))}
    </div>
  )
}
```

## 🎨 Styling

### Tailwind Configuration

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          900: '#1e3a8a',
        },
        secondary: {
          50: '#f8fafc',
          500: '#64748b',
          900: '#0f172a',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
```

### Component Styling

```tsx
// src/components/ui/Button.tsx
import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "underline-offset-4 hover:underline text-primary",
      },
      size: {
        default: "h-10 py-2 px-4",
        sm: "h-9 px-3 rounded-md",
        lg: "h-11 px-8 rounded-md",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return (
    <button
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}
```

## 🧪 Testing

### Unit Testing with Jest

```typescript
// tests/components/Button.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { Button } from '@/components/ui/Button'

describe('Button', () => {
  it('renders with correct text', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByRole('button', { name: 'Click me' })).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = jest.fn()
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

test('authentication flow', async ({ page }) => {
  await page.goto('/auth/signin')
  
  await page.fill('input[name="email"]', 'test@example.com')
  await page.fill('input[name="password"]', 'password123')
  await page.click('button[type="submit"]')
  
  await expect(page).toHaveURL('/dashboard')
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

### Docker Production

```dockerfile
# Dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json pnpm-lock.yaml* ./
RUN corepack enable pnpm && pnpm i --frozen-lockfile

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN corepack enable pnpm && pnpm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

### Environment Variables for Production

```bash
# Production environment variables
DATABASE_URL="postgresql://user:password@db:5432/{{ project_name }}_prod"
NEXTAUTH_URL="https://{{ project_name }}.vercel.app"
NEXTAUTH_SECRET="your-production-secret-key"
GOOGLE_CLIENT_ID="your-production-google-client-id"
GOOGLE_CLIENT_SECRET="your-production-google-client-secret"
```

## 📊 Analytics & Monitoring

### Vercel Analytics

```typescript
// src/app/layout.tsx
import { Analytics } from '@vercel/analytics/react'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
```

### Sentry Integration

```typescript
// src/lib/sentry.ts
import * as Sentry from "@sentry/nextjs"

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 1.0,
})
```

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
- **Database migrations** for schema changes

## 📄 License

This project is licensed under the {{ license }} License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Next.js Team** for the amazing framework
- **Vercel Team** for the deployment platform
- **Prisma Team** for the database toolkit
- **tRPC Team** for the typesafe APIs
- **Template Heaven** for the fullstack template

---

**Built with ❤️ using Template Heaven Next.js Fullstack Template**

*This template provides a complete foundation for building modern, scalable fullstack applications with Next.js, TypeScript, and best practices.*
