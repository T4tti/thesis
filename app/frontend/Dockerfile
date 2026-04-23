# Stage 1: Install dependencies (cache-friendly)
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# Stage 2: Build Next.js app
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
RUN npm run build

# Stage 3: Production runner (standalone output)
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
ENV API_INTERNAL_URL=http://backend:8000

# Copy standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

USER node

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]


# Stage 4: Dev runner (hot-reload via bind mount in docker-compose.override.yml)
FROM node:20-alpine AS dev
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

RUN apk add --no-cache su-exec

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Bind mounts / named volumes (e.g. /app/.next, /app/node_modules) may be root-owned at runtime.
# Keep root here so entrypoint can fix perms, then drop to "node".
COPY docker-entrypoint-dev.sh /app/docker-entrypoint-dev.sh
RUN chmod +x /app/docker-entrypoint-dev.sh || true

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

ENTRYPOINT ["sh", "/app/docker-entrypoint-dev.sh"]
CMD ["npm", "run", "dev"]
