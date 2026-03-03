# Stage 1: Build React frontend
FROM node:20-slim AS frontend

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Override API URL for Docker: use relative path so nginx proxies it
RUN echo "REACT_APP_API_BASE_URL=/api" > .env
RUN npm run build

# Stage 2: nginx serving static files
FROM nginx:alpine

COPY --from=frontend /frontend/build /var/www/sharkid
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
