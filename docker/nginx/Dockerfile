# Nginx API gateway image.
#
# Built from the official Alpine image, which is a few megabytes rather than a
# few hundred. Nothing is compiled here, so there is no need for a multi-stage
# build; the configuration is simply copied in and validated at build time.
FROM nginx:1.27-alpine

# Replace the default configuration entirely rather than adding to it: the
# stock config includes a sample server block that would otherwise still be
# listening on port 80.
COPY nginx.conf /etc/nginx/nginx.conf

# Validate the configuration at build time, so a syntax error fails the build
# rather than the deploy.
#
# Nginx resolves upstream hostnames when it parses the config, and `ml-api`
# only exists once Docker Compose has created the network — so a plain
# `nginx -t` here fails with "host not found in upstream". We therefore
# validate a *copy* with the upstream rewritten to a literal address. Every
# directive is still checked; only the name lookup is sidestepped. The shipped
# config is untouched and resolves normally through Docker's DNS at runtime.
RUN mkdir -p /var/log/nginx \
    && sed 's|server ml-api:8000|server 127.0.0.1:8000|' /etc/nginx/nginx.conf > /tmp/validate.conf \
    && nginx -t -c /tmp/validate.conf \
    && rm /tmp/validate.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1/api/v1/health/live || exit 1

# "daemon off" keeps Nginx in the foreground so Docker can supervise it and
# deliver signals directly.
CMD ["nginx", "-g", "daemon off;"]
