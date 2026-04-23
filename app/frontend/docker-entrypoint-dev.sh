#!/bin/sh
set -e

# Ensure dev cache dirs exist and are writable by the non-root "node" user.
mkdir -p /app/.next /app/node_modules
chown -R node:node /app/.next /app/node_modules 2>/dev/null || true

exec su-exec node "$@"

