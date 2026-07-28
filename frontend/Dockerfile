# MediaMotion frontend (Next.js, dev server for local use).
FROM node:20-bookworm-slim
ENV NODE_ENV=development
WORKDIR /app
COPY frontend/package.json /app/package.json
RUN npm install
COPY frontend/ /app/
EXPOSE 3000
CMD ["npm", "run", "dev"]
