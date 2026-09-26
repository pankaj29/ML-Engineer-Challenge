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
# rather than the deploy. The ml-api host does not exist at build time, and
# that is fine: the config proxies to a variable, which nginx resolves per
# request at runtime, not while parsing.
RUN mkdir -p /var/log/nginx && nginx -t

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1/api/v1/health/live || exit 1

# "daemon off" keeps Nginx in the foreground so Docker can supervise it and
# deliver signals directly.
CMD ["nginx", "-g", "daemon off;"]
