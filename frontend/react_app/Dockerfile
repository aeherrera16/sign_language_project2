FROM node:20-alpine

WORKDIR /app

COPY frontend/react_app/package.json ./

# Install deps
RUN npm install --silent --legacy-peer-deps

COPY frontend/react_app/ /app/

EXPOSE 5173

CMD ["npm", "run", "dev"]
