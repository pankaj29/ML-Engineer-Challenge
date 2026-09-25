# A stand-in for the artifact store, used only by the kind overlay.
#
# The base manifests fetch weights over https from object storage. There is no
# object storage on a laptop, and hostPath is forbidden by the namespace's
# restricted Pod Security Standard, so this serves the same files over HTTP
# from inside the cluster. That exercises the real network fetch path rather
# than a file:// shortcut.
#
# nginx-unprivileged, not nginx: the stock image starts as root, which the
# restricted standard also rejects.
#
# Build with models/artifacts as the context, because the repo's .dockerignore
# excludes that directory from the application build:
#
#   docker build -f k8s/overlays/kind/Dockerfile.artifact-server \
#     -t mlcv/artifact-server:kind models/artifacts
FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY --chown=101:101 *.onnx /usr/share/nginx/html/
COPY --chown=101:101 *labels*.json /usr/share/nginx/html/

EXPOSE 8080
