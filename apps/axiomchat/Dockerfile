# syntax=docker/dockerfile:1
# Multi-stage build for AxiomChat: build the Vite SPA and the TS server
# separately, then assemble a slim runtime that serves the static SPA + /api.

# --- Stage 1: build the React SPA -> web/dist ------------------------------
FROM node:20-slim AS web
WORKDIR /app/web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build

# --- Stage 2: build the Express server -> dist -----------------------------
FROM node:20-slim AS server
WORKDIR /app
COPY package.json ./
RUN npm install
COPY tsconfig.json ./
COPY src/ ./src/
RUN npm run build

# --- Stage 3: slim runtime -------------------------------------------------
FROM node:20-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
COPY package.json ./
RUN npm install --omit=dev
COPY --from=server /app/dist ./dist
COPY --from=web /app/web/dist ./web/dist
EXPOSE 3000
# node-based healthcheck (the slim image has no curl).
HEALTHCHECK --interval=5s --timeout=3s --retries=5 --start-period=5s \
  CMD node -e "require('http').get('http://localhost:3000/api/health',(r)=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"
CMD ["node", "dist/server.js"]
