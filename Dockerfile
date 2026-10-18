# Single-container deployment: the Node server runs the API and WebSocket,
# spawns the C++ engine, and serves the built React client on one origin.

FROM debian:bookworm-slim AS engine
RUN apt-get update && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/cpp
COPY cpp/ ./
RUN make clean && make && make test_all

FROM node:24-bookworm-slim AS client
WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM node:24-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/server
COPY server/package.json server/package-lock.json ./
RUN npm ci --omit=dev
COPY server/*.js ./
COPY --from=engine /app/cpp/linear_regression_app /app/cpp/linear_regression_app
COPY --from=client /app/client/build /app/client/build

ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3001 \
    STATIC_DIR=/app/client/build
USER node
EXPOSE 3001
CMD ["node", "server.js"]
