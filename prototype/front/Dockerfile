FROM node:18 AS build

WORKDIR /app
COPY package*.json ./
COPY . .
RUN npm install
RUN npm run build --production

# RUN npm install -g serve

# EXPOSE 5000
# CMD ["serve", "-s", "build"]

FROM nginx:alpine

COPY ./nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
