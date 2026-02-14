#!/bin/bash
set -e

echo "SharkID Pre-Removal"

# Stop and disable services
if systemctl is-active --quiet sharkid-api.service; then
    echo "Stopping sharkid-api service..."
    systemctl stop sharkid-api.service
fi

if systemctl is-active --quiet sharkid-worker.service; then
    echo "Stopping sharkid-worker service..."
    systemctl stop sharkid-worker.service
fi

systemctl disable sharkid-api.service 2>/dev/null || true
systemctl disable sharkid-worker.service 2>/dev/null || true

# Remove nginx site
if [ -L /etc/nginx/sites-enabled/sharkid.conf ]; then
    echo "Removing nginx site configuration..."
    rm -f /etc/nginx/sites-enabled/sharkid.conf
    systemctl reload nginx 2>/dev/null || true
fi

echo "SharkID services stopped and disabled."
