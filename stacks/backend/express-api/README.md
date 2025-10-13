# Express.js API Template

A production-ready Express.js API template with TypeScript, authentication, validation, and testing.

## 🚀 Features

- **Express.js** with TypeScript
- **JWT Authentication** with refresh tokens
- **Input Validation** with Joi/Zod
- **Database Integration** with Prisma
- **API Documentation** with Swagger
- **Testing** with Jest and Supertest
- **Docker** support
- **Rate Limiting** and security middleware
- **Logging** with Winston
- **Environment Configuration**

## 📋 Prerequisites

- Node.js 18+
- npm, yarn, or pnpm
- PostgreSQL, MySQL, or SQLite
- Docker (optional)

## 🛠️ Quick Start

### 1. Create New Project

```bash
mkdir my-api
cd my-api
npm init -y
```

### 2. Install Dependencies

```bash
npm install express cors helmet morgan compression
npm install -D @types/express @types/cors @types/morgan @types/compression typescript ts-node nodemon
```

### 3. Environment Setup

```bash
cp .env.example .env
```

Configure your environment variables:

```env
NODE_ENV=development
PORT=3000
DATABASE_URL="postgresql://username:password@localhost:5432/myapi"
JWT_SECRET="your-jwt-secret"
JWT_REFRESH_SECRET="your-refresh-secret"
```

### 4. Database Setup

```bash
npx prisma generate
npx prisma db push
npx prisma db seed
```

### 5. Start Development Server

```bash
npm run dev
```

API will be available at [http://localhost:3000](http://localhost:3000)

## 📁 Project Structure

```
├── src/
│   ├── controllers/        # Route controllers
│   ├── middleware/         # Custom middleware
│   ├── models/            # Database models
│   ├── routes/            # API routes
│   ├── services/          # Business logic
│   ├── utils/             # Utility functions
│   ├── types/             # TypeScript types
│   ├── config/            # Configuration files
│   ├── app.ts             # Express app setup
│   └── server.ts          # Server entry point
├── prisma/                # Database schema
├── tests/                 # Test files
├── docs/                  # API documentation
└── docker/                # Docker files
```

## 🔧 Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run lint:fix     # Fix ESLint issues
npm run format       # Format code with Prettier
npm run test         # Run tests
npm run test:watch   # Run tests in watch mode
npm run test:coverage # Run tests with coverage
npm run db:generate  # Generate Prisma client
npm run db:push      # Push schema to database
npm run db:migrate   # Run database migrations
npm run db:seed      # Seed database
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
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  password  String
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

## 🔐 Authentication

```typescript
// src/middleware/auth.ts
import jwt from 'jsonwebtoken';
import { Request, Response, NextFunction } from 'express';

interface AuthRequest extends Request {
  user?: any;
}

export const authenticateToken = (req: AuthRequest, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.sendStatus(401);
  }

  jwt.verify(token, process.env.JWT_SECRET!, (err, user) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
};
```

## 📝 API Routes

```typescript
// src/routes/users.ts
import express from 'express';
import { createUser, getUsers, getUserById, updateUser, deleteUser } from '../controllers/userController';
import { authenticateToken } from '../middleware/auth';
import { validateUser } from '../middleware/validation';

const router = express.Router();

router.post('/', validateUser, createUser);
router.get('/', authenticateToken, getUsers);
router.get('/:id', authenticateToken, getUserById);
router.put('/:id', authenticateToken, validateUser, updateUser);
router.delete('/:id', authenticateToken, deleteUser);

export default router;
```

## 🧪 Testing

```typescript
// tests/users.test.ts
import request from 'supertest';
import app from '../src/app';

describe('Users API', () => {
  it('should create a new user', async () => {
    const userData = {
      name: 'John Doe',
      email: 'john@example.com',
      password: 'password123'
    };

    const response = await request(app)
      .post('/api/users')
      .send(userData)
      .expect(201);

    expect(response.body).toHaveProperty('id');
    expect(response.body.email).toBe(userData.email);
  });
});
```

## 📚 API Documentation

Swagger documentation is available at `/api-docs` when running the server.

```typescript
// src/config/swagger.ts
import swaggerJsdoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';

const options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Express API',
      version: '1.0.0',
      description: 'A simple Express API with TypeScript',
    },
    servers: [
      {
        url: 'http://localhost:3000',
        description: 'Development server',
      },
    ],
  },
  apis: ['./src/routes/*.ts'],
};

const specs = swaggerJsdoc(options);

export { specs, swaggerUi };
```

## 🚀 Deployment

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables

```env
NODE_ENV=production
PORT=3000
DATABASE_URL="postgresql://user:pass@db:5432/myapi"
JWT_SECRET="your-production-jwt-secret"
JWT_REFRESH_SECRET="your-production-refresh-secret"
```

## 📚 Learning Resources

- [Express.js Documentation](https://expressjs.com/)
- [Prisma Documentation](https://www.prisma.io/docs)
- [JWT Documentation](https://jwt.io/)
- [Swagger Documentation](https://swagger.io/docs/)

## 🔗 Upstream Source

- **Repository**: [sahat/hackathon-starter](https://github.com/sahat/hackathon-starter)
- **Documentation**: [hackathon-starter.herokuapp.com](https://hackathon-starter.herokuapp.com/)
- **License**: MIT
