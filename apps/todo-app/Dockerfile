FROM node:20-slim AS builder
WORKDIR /app
COPY package.json tsconfig.json ./
RUN npm install
COPY server.ts ./
RUN npx tsc

FROM node:20-slim
WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY --from=builder /app/dist ./dist
COPY public ./public
EXPOSE 3000
HEALTHCHECK --interval=5s --retries=3 CMD curl -f http://localhost:3000/api/health || exit 1
CMD ["node", "dist/server.js"]
