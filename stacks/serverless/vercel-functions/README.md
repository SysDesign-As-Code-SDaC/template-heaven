# Vercel Serverless Functions Template

A production-ready serverless application template using Vercel Functions, featuring modern edge computing, API routes, and full-stack deployment for 2025.

## 🚀 Features

- **Vercel Functions** - Serverless API endpoints
- **Next.js 14** - Full-stack React framework
- **Edge Runtime** - Global edge computing
- **TypeScript** - Type-safe development
- **Prisma** - Database ORM with edge support
- **Vercel KV** - Redis-compatible edge storage
- **Vercel Postgres** - Serverless PostgreSQL
- **Vercel Blob** - File storage and CDN
- **Cron Jobs** - Scheduled functions
- **Webhooks** - Event-driven functions
- **Middleware** - Request/response processing
- **Analytics** - Built-in performance monitoring

## 📋 Prerequisites

- Node.js 18+
- Vercel account
- Git

## 🛠️ Quick Start

### 1. Create New Project

```bash
npx create-next-app@latest my-serverless-app --typescript --tailwind --eslint --app
cd my-serverless-app
```

### 2. Install Vercel CLI

```bash
npm install -g vercel
```

### 3. Environment Setup

```bash
cp .env.example .env.local
```

Configure your environment variables:

```env
# Database
DATABASE_URL="postgresql://user:pass@localhost:5432/myapp"
DIRECT_URL="postgresql://user:pass@localhost:5432/myapp"

# Vercel Services
KV_REST_API_URL="https://your-kv-instance.vercel-storage.com"
KV_REST_API_TOKEN="your-kv-token"
BLOB_READ_WRITE_TOKEN="your-blob-token"

# External APIs
OPENAI_API_KEY="your-openai-key"
STRIPE_SECRET_KEY="your-stripe-key"

# Application
NEXTAUTH_SECRET="your-secret"
NEXTAUTH_URL="http://localhost:3000"
```

### 4. Deploy to Vercel

```bash
vercel login
vercel
```

## 📁 Project Structure

```
├── app/                        # Next.js App Router
│   ├── api/                   # API routes (serverless functions)
│   │   ├── auth/              # Authentication endpoints
│   │   ├── users/             # User management
│   │   ├── webhooks/          # Webhook handlers
│   │   └── cron/              # Scheduled functions
│   ├── (dashboard)/           # Dashboard pages
│   ├── globals.css            # Global styles
│   ├── layout.tsx             # Root layout
│   └── page.tsx               # Home page
├── lib/                       # Utility functions
│   ├── auth.ts                # Authentication logic
│   ├── db.ts                  # Database connection
│   ├── kv.ts                  # Vercel KV operations
│   └── utils.ts               # Utility functions
├── components/                # React components
│   ├── ui/                    # UI components
│   └── forms/                 # Form components
├── middleware.ts              # Next.js middleware
├── vercel.json                # Vercel configuration
└── prisma/                    # Database schema
    ├── schema.prisma
    └── migrations/
```

## 🔧 Available Scripts

```bash
# Development
npm run dev                    # Start development server
npm run build                  # Build for production
npm run start                  # Start production server
npm run lint                   # Run ESLint
npm run type-check             # Type check

# Database
npm run db:generate            # Generate Prisma client
npm run db:push                # Push schema to database
npm run db:migrate             # Run migrations
npm run db:seed                # Seed database

# Deployment
vercel dev                     # Local development with Vercel
vercel build                   # Build for Vercel
vercel deploy                  # Deploy to Vercel
vercel logs                    # View function logs
```

## 🌐 API Routes (Serverless Functions)

### User Management API

```typescript
// app/api/users/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { kv } from '@/lib/kv';

export async function GET(request: NextRequest) {
  try {
    // Check cache first
    const cachedUsers = await kv.get('users:list');
    if (cachedUsers) {
      return NextResponse.json(JSON.parse(cachedUsers));
    }

    // Fetch from database
    const users = await prisma.user.findMany({
      select: {
        id: true,
        name: true,
        email: true,
        createdAt: true,
      },
    });

    // Cache for 5 minutes
    await kv.setex('users:list', 300, JSON.stringify(users));

    return NextResponse.json(users);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch users' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name, email } = body;

    const user = await prisma.user.create({
      data: { name, email },
    });

    // Invalidate cache
    await kv.del('users:list');

    return NextResponse.json(user, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create user' },
      { status: 500 }
    );
  }
}
```

### Webhook Handler

```typescript
// app/api/webhooks/stripe/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { headers } from 'next/headers';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

export async function POST(request: NextRequest) {
  const body = await request.text();
  const signature = headers().get('stripe-signature')!;

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return NextResponse.json({ error: 'Invalid signature' }, { status: 400 });
  }

  switch (event.type) {
    case 'payment_intent.succeeded':
      const paymentIntent = event.data.object as Stripe.PaymentIntent;
      console.log('Payment succeeded:', paymentIntent.id);
      
      // Update database, send confirmation email, etc.
      await handlePaymentSuccess(paymentIntent);
      break;

    case 'customer.subscription.created':
      const subscription = event.data.object as Stripe.Subscription;
      console.log('Subscription created:', subscription.id);
      
      await handleSubscriptionCreated(subscription);
      break;

    default:
      console.log(`Unhandled event type: ${event.type}`);
  }

  return NextResponse.json({ received: true });
}

async function handlePaymentSuccess(paymentIntent: Stripe.PaymentIntent) {
  // Implementation for payment success
}

async function handleSubscriptionCreated(subscription: Stripe.Subscription) {
  // Implementation for subscription creation
}
```

### Cron Job Function

```typescript
// app/api/cron/cleanup/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { kv } from '@/lib/kv';

export async function GET(request: NextRequest) {
  // Verify cron secret
  const authHeader = request.headers.get('authorization');
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // Clean up old sessions
    const deletedSessions = await prisma.session.deleteMany({
      where: {
        expiresAt: {
          lt: new Date(),
        },
      },
    });

    // Clean up expired cache entries
    const keys = await kv.keys('temp:*');
    for (const key of keys) {
      const ttl = await kv.ttl(key);
      if (ttl === -1) { // No expiration set
        await kv.expire(key, 3600); // Set 1 hour expiration
      }
    }

    return NextResponse.json({
      message: 'Cleanup completed',
      deletedSessions: deletedSessions.count,
      processedKeys: keys.length,
    });
  } catch (error) {
    console.error('Cleanup failed:', error);
    return NextResponse.json(
      { error: 'Cleanup failed' },
      { status: 500 }
    );
  }
}
```

## 🗄️ Database with Prisma

```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
  directUrl = env("DIRECT_URL")
}

model User {
  id        String   @id @default(cuid())
  name      String?
  email     String   @unique
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  
  sessions  Session[]
  posts     Post[]
}

model Session {
  id        String   @id @default(cuid())
  userId    String
  expiresAt DateTime
  createdAt DateTime @default(now())
  
  user      User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@index([expiresAt])
}

model Post {
  id        String   @id @default(cuid())
  title     String
  content   String?
  published Boolean  @default(false)
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  
  author    User     @relation(fields: [authorId], references: [id])
}
```

## 🚀 Vercel Configuration

```json
// vercel.json
{
  "functions": {
    "app/api/**/*.ts": {
      "runtime": "nodejs18.x",
      "maxDuration": 30
    },
    "app/api/cron/**/*.ts": {
      "runtime": "nodejs18.x",
      "maxDuration": 60
    }
  },
  "crons": [
    {
      "path": "/api/cron/cleanup",
      "schedule": "0 2 * * *"
    }
  ],
  "env": {
    "DATABASE_URL": "@database-url",
    "KV_REST_API_URL": "@kv-rest-api-url",
    "KV_REST_API_TOKEN": "@kv-rest-api-token"
  }
}
```

## 🔒 Middleware

```typescript
// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Add security headers
  const response = NextResponse.next();
  
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');

  // Rate limiting for API routes
  if (request.nextUrl.pathname.startsWith('/api/')) {
    const ip = request.ip ?? '127.0.0.1';
    const rateLimitKey = `rate_limit:${ip}`;
    
    // This would integrate with Vercel KV for rate limiting
    // Implementation depends on your rate limiting strategy
  }

  return response;
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
```

## 📊 Analytics and Monitoring

```typescript
// lib/analytics.ts
import { Analytics } from '@vercel/analytics/react';

export function trackEvent(event: string, properties?: Record<string, any>) {
  if (typeof window !== 'undefined') {
    // Client-side tracking
    Analytics.track(event, properties);
  }
}

export function trackPageView(url: string) {
  if (typeof window !== 'undefined') {
    Analytics.page(url);
  }
}
```

## 🚀 Deployment

### Environment Variables

Set these in your Vercel dashboard:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
DIRECT_URL=postgresql://user:pass@host:5432/db

# Vercel Services
KV_REST_API_URL=https://your-kv.vercel-storage.com
KV_REST_API_TOKEN=your-kv-token
BLOB_READ_WRITE_TOKEN=your-blob-token

# External APIs
OPENAI_API_KEY=your-openai-key
STRIPE_SECRET_KEY=your-stripe-key
STRIPE_WEBHOOK_SECRET=your-webhook-secret

# Application
NEXTAUTH_SECRET=your-secret
NEXTAUTH_URL=https://your-app.vercel.app
CRON_SECRET=your-cron-secret
```

### Deployment Commands

```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod

# View logs
vercel logs

# View function logs
vercel logs --function=api/users
```

## 📚 Learning Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js API Routes](https://nextjs.org/docs/api-routes/introduction)
- [Vercel Functions](https://vercel.com/docs/functions)
- [Prisma with Vercel](https://www.prisma.io/docs/guides/deployment/deployment-guides/deploying-to-vercel)

## 🔗 Upstream Source

- **Repository**: [vercel/next.js](https://github.com/vercel/next.js)
- **Documentation**: [vercel.com/docs](https://vercel.com/docs)
- **License**: MIT
