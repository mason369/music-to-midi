# syntax=docker/dockerfile:1.7

ARG CADDY_IMAGE=caddy:2.11.4-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648
FROM ${CADDY_IMAGE}

ARG VCS_REF=unknown
ARG BUILD_VERSION=dev

LABEL org.opencontainers.image.title="Music to MIDI gateway" \
      org.opencontainers.image.description="Authenticated same-origin web frontend and HTTPS gateway" \
      org.opencontainers.image.source="https://github.com/mason369/music-to-midi" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.licenses="MIT"

COPY web /srv/web
COPY docker/Caddyfile /etc/caddy/Caddyfile
COPY docker/gateway-entrypoint.sh /usr/local/bin/music-to-midi-gateway

RUN caddy fmt --overwrite /etc/caddy/Caddyfile \
    && PUBLIC_ADDRESS=container-build.example \
       PUBLIC_ORIGIN=https://container-build.example \
       ACME_EMAIL=container-build@example.com \
       BASIC_AUTH_USER=containercheck \
       BASIC_AUTH_HASH='$argon2id$v=19$m=47104,t=1,p=1$zJPvVe48N64JUa9MFlVhiw$b5Tznu0PxnA4TciY6qYe2BFPxncF1ePQaeNukHhH1cU' \
       MAX_REQUEST_BODY_SIZE=1MiB \
       caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile \
    && addgroup -S -g 10001 caddyapp \
    && adduser -S -D -H -u 10001 -G caddyapp caddyapp \
    && chown -R 10001:10001 /srv/web \
    && chmod 0755 /usr/local/bin/music-to-midi-gateway \
    && chmod 1777 /config/caddy /data/caddy

USER 10001:10001

EXPOSE 8080 8443 8443/udp

ENTRYPOINT ["/usr/local/bin/music-to-midi-gateway"]
