# T3 Stack Template

A production-ready fullstack TypeScript template using the T3 Stack (Next.js, TypeScript, tRPC, Prisma, NextAuth.js, and Tailwind CSS).

## 🚀 What's Included

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **tRPC** - End-to-end typesafe APIs
- **Prisma** - Database ORM with migrations
- **NextAuth.js** - Authentication
- **Tailwind CSS** - Utility-first CSS framework
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **Husky** - Git hooks
- **TypeScript Strict Mode** - Enhanced type checking

## 📋 Prerequisites

- Node.js 18+ 
- npm, yarn, or pnpm
- PostgreSQL, MySQL, or SQLite database
- Git

## 🛠️ Quick Start

### 1. Create New Project

```bash
# Using the template
npx create-t3-app@latest my-app

# Or clone this template
git clone <this-repo> my-app
cd my-app
```

### 2. Install Dependencies

```bash
npm install
# or
yarn install
# or
pnpm install
```

### 3. Environment Setup

Copy the environment file and configure:

```bash
cp .env.example .env
```

Required environment variables:

```env
# Database
DATABASE_URL="postgresql://username:password@localhost:5432/myapp"

# NextAuth.js
NEXTAUTH_SECRET="your-secret-key"
NEXTAUTH_URL="http://localhost:3000"

# OAuth Providers (optional)
GOOGLE_CLIENT_ID=""
GOOGLE_CLIENT_SECRET=""
GITHUB_CLIENT_ID=""
GITHUB_CLIENT_SECRET=""
```

### 4. Database Setup

```bash
# Generate Prisma client
npx prisma generate

# Run database migrations
npx prisma db push

# (Optional) Seed the database
npx prisma db seed
```

### 5. Start Development Server

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
├── src/
│   ├── app/                 # Next.js App Router
│   │   ├── api/            # API routes
│   │   ├── auth/           # Authentication pages
│   │   └── (pages)/        # Application pages
│   ├── components/         # Reusable UI components
│   ├── lib/               # Utility functions
│   │   ├── auth.ts        # NextAuth configuration
│   │   ├── db.ts          # Database connection
│   │   └── utils.ts       # Utility functions
│   ├── server/            # Server-side code
│   │   ├── api/           # tRPC routers
│   │   └── db/            # Database queries
│   └── styles/            # Global styles
├── prisma/                # Database schema and migrations
├── public/                # Static assets
└── docs/                  # Documentation
```

## 🔧 Available Scripts

```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run lint:fix     # Fix ESLint issues
npm run format       # Format code with Prettier

# Database
npm run db:generate  # Generate Prisma client
npm run db:push      # Push schema to database
npm run db:migrate   # Run database migrations
npm run db:seed      # Seed database with sample data
npm run db:studio    # Open Prisma Studio

# Testing
npm run test         # Run tests
npm run test:watch   # Run tests in watch mode
npm run test:coverage # Run tests with coverage
```

## 🗄️ Database

This template uses Prisma as the database ORM. The schema is defined in `prisma/schema.prisma`.

### Example Schema

```prisma
// This is your Prisma schema file,
// learn more about it in the docs: https://pris.ly/d/prisma-schema

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id            String    @id @default(cuid())
  name          String?
  email         String    @unique
  emailVerified DateTime?
  image         String?
  accounts      Account[]
  sessions      Session[]
  posts         Post[]
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
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
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)
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

## 🔐 Authentication

This template includes NextAuth.js for authentication with support for:

- Email/Password
- OAuth providers (Google, GitHub, etc.)
- Database sessions
- Type-safe authentication

### Usage Example

```typescript
import { useSession } from "next-auth/react";

export default function Profile() {
  const { data: session, status } = useSession();

  if (status === "loading") return <p>Loading...</p>;
  if (status === "unauthenticated") return <p>Access Denied</p>;

  return (
    <div>
      <h1>Profile</h1>
      <p>Welcome, {session?.user?.name}!</p>
    </div>
  );
}
```

## 🌐 API with tRPC

tRPC provides end-to-end type safety for your APIs.

### Example Router

```typescript
// src/server/api/routers/posts.ts
import { z } from "zod";
import { createTRPCRouter, publicProcedure, protectedProcedure } from "~/server/api/trpc";

export const postsRouter = createTRPCRouter({
  getAll: publicProcedure.query(({ ctx }) => {
    return ctx.db.post.findMany({
      include: { author: true },
    });
  }),

  create: protectedProcedure
    .input(z.object({ title: z.string(), content: z.string() }))
    .mutation(({ ctx, input }) => {
      return ctx.db.post.create({
        data: {
          title: input.title,
          content: input.content,
          authorId: ctx.session.user.id,
        },
      });
    }),
});
```

### Client Usage

```typescript
import { api } from "~/utils/api";

export default function Posts() {
  const { data: posts, isLoading } = api.posts.getAll.useQuery();

  if (isLoading) return <div>Loading...</div>;

  return (
    <div>
      {posts?.map((post) => (
        <div key={post.id}>
          <h2>{post.title}</h2>
          <p>{post.content}</p>
          <p>By {post.author.name}</p>
        </div>
      ))}
    </div>
  );
}
```

## 🎨 Styling

This template uses Tailwind CSS for styling. Key features:

- Utility-first CSS framework
- Responsive design utilities
- Dark mode support
- Custom color palette
- Component-based styling

### Example Component

```typescript
import { cn } from "~/lib/utils";

interface ButtonProps {
  variant?: "primary" | "secondary";
  size?: "sm" | "md" | "lg";
  children: React.ReactNode;
  className?: string;
}

export function Button({ 
  variant = "primary", 
  size = "md", 
  children, 
  className 
}: ButtonProps) {
  return (
    <button
      className={cn(
        "rounded-md font-medium transition-colors",
        {
          "bg-blue-600 text-white hover:bg-blue-700": variant === "primary",
          "bg-gray-200 text-gray-900 hover:bg-gray-300": variant === "secondary",
          "px-3 py-1.5 text-sm": size === "sm",
          "px-4 py-2 text-base": size === "md",
          "px-6 py-3 text-lg": size === "lg",
        },
        className
      )}
    >
      {children}
    </button>
  );
}
```

## 🧪 Testing

This template includes testing setup with:

- Jest for unit testing
- React Testing Library for component testing
- Playwright for end-to-end testing

### Running Tests

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Test coverage
npm run test:coverage
```

## 🚀 Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy!

### Docker

```dockerfile
FROM node:18-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS builder
WORKDIR /app
COPY . .
COPY --from=deps /app/node_modules ./node_modules
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
```

## 📚 Learning Resources

- [T3 Stack Documentation](https://create.t3.gg/)
- [Next.js Documentation](https://nextjs.org/docs)
- [tRPC Documentation](https://trpc.io/docs)
- [Prisma Documentation](https://www.prisma.io/docs)
- [NextAuth.js Documentation](https://next-auth.js.org/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This template is based on the T3 Stack and follows the same licensing terms.

## 🔗 Upstream Source

- **Repository**: [t3-oss/create-t3-app](https://github.com/t3-oss/create-t3-app)
- **Documentation**: [create.t3.gg](https://create.t3.gg/)
- **License**: MIT

---

Built with ❤️ using the T3 Stack
