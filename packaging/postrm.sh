#!/bin/bash
set -e

# Only remove user and data on purge, not on upgrade
if [ "$1" = "purge" ]; then
    echo "SharkID Post-Removal"

    # Remove sharkid user
    if id -u sharkid > /dev/null 2>&1; then
        echo "Removing sharkid user..."
        userdel sharkid 2>/dev/null || true
    fi

    # Remove application directories
    if [ -d /opt/sharkid ]; then
        echo "Removing /opt/sharkid..."
        rm -rf /opt/sharkid
    fi

    if [ -d /var/www/sharkid ]; then
        echo "Removing /var/www/sharkid..."
        rm -rf /var/www/sharkid
    fi
fi
