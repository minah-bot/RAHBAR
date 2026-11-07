# syntax=docker/dockerfile:1
# Stage 1 stamps the build with the commit it came from; there is no Node toolchain because
# the UI is static HTML, so nothing but nginx and the files ships.
FROM alpine:3.20 AS stage
ARG GIT_SHA=dev
WORKDIR /site
COPY index.html .
RUN echo "$GIT_SHA" > version.txt

FROM nginxinc/nginx-unprivileged:1.27-alpine
# The tag's Alpine packages lag behind security fixes (openssl, curl, expat, zlib...). Upgrade them at build
# time as root, then return to the unprivileged user: the running container never has root.
USER root
RUN apk upgrade --no-cache
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=stage /site /usr/share/nginx/html
USER 101
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
