#!/bin/bash
set -e

VERSION="${1:-0.1.0}"
ARCH="amd64"
PACKAGE_NAME="sharkid"

echo "Building SharkID .deb package v${VERSION}"

# Get the project root (parent of packaging/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING_DIR="${PROJECT_ROOT}/build/staging"

# Clean and create staging directory
rm -rf "${PROJECT_ROOT}/build"
mkdir -p "${STAGING_DIR}"

echo "Staging directory: ${STAGING_DIR}"

# 1. Create directory structure
echo "Creating directory structure..."
mkdir -p "${STAGING_DIR}/opt/sharkid/backend"
mkdir -p "${STAGING_DIR}/opt/sharkid/data"
mkdir -p "${STAGING_DIR}/var/www/sharkid"
mkdir -p "${STAGING_DIR}/etc/systemd/system"
mkdir -p "${STAGING_DIR}/etc/nginx/sites-available"

# 2. Copy backend code
echo "Copying backend code..."
cp -r "${PROJECT_ROOT}/backend/app" "${STAGING_DIR}/opt/sharkid/backend/"
cp -r "${PROJECT_ROOT}/backend/worker" "${STAGING_DIR}/opt/sharkid/backend/"
cp "${PROJECT_ROOT}/backend/requirements.txt" "${STAGING_DIR}/opt/sharkid/backend/"
cp "${PROJECT_ROOT}/backend/requirements-torch.txt" "${STAGING_DIR}/opt/sharkid/backend/"
cp "${PROJECT_ROOT}/backend/seed.py" "${STAGING_DIR}/opt/sharkid/backend/"

# 3. Copy model files to expected location
echo "Setting up model files..."
# The code expects models in worker/model/, so copy them there
mkdir -p "${STAGING_DIR}/opt/sharkid/backend/worker/model"
cp "${PROJECT_ROOT}/backend/worker/efficientnet/cnn_bundle.pkl" \
   "${STAGING_DIR}/opt/sharkid/backend/worker/model/"
cp "${PROJECT_ROOT}/backend/worker/efficientnet/cnn_final.pth" \
   "${STAGING_DIR}/opt/sharkid/backend/worker/model/"

# 4. Remove unnecessary files
echo "Cleaning up unnecessary files..."
# Remove statistics directory (576MB, unused)
rm -rf "${STAGING_DIR}/opt/sharkid/backend/worker/statistics"
# Remove __pycache__ directories
find "${STAGING_DIR}/opt/sharkid/backend" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 5. Copy frontend build
echo "Copying frontend build..."
if [ -d "${PROJECT_ROOT}/frontend/build" ]; then
    cp -r "${PROJECT_ROOT}/frontend/build/"* "${STAGING_DIR}/var/www/sharkid/"
else
    echo "ERROR: Frontend build not found at ${PROJECT_ROOT}/frontend/build"
    echo "Please run 'npm run build' in the frontend directory first."
    exit 1
fi

# 6. Copy systemd service files
echo "Copying systemd service files..."
cp "${PROJECT_ROOT}/packaging/sharkid-api.service" \
   "${STAGING_DIR}/etc/systemd/system/"
cp "${PROJECT_ROOT}/packaging/sharkid-worker.service" \
   "${STAGING_DIR}/etc/systemd/system/"

# 7. Copy nginx config
echo "Copying nginx configuration..."
cp "${PROJECT_ROOT}/packaging/nginx-sharkid.conf" \
   "${STAGING_DIR}/etc/nginx/sites-available/sharkid.conf"

# 8. Copy packaging scripts
echo "Copying packaging scripts..."
mkdir -p "${STAGING_DIR}/opt/sharkid/packaging"
cp "${PROJECT_ROOT}/packaging/env.template" \
   "${STAGING_DIR}/opt/sharkid/packaging/"

# 9. Build .deb package with fpm
echo "Building .deb package with fpm..."

fpm -s dir -t deb \
    -n "${PACKAGE_NAME}" \
    -v "${VERSION}" \
    -a "${ARCH}" \
    --description "SharkID - Shark Species Identification System" \
    --url "https://github.com/yourusername/SharkMQP26" \
    --maintainer "SharkID Team" \
    --license "MIT" \
    --depends "python3 >= 3.11" \
    --depends "python3-venv" \
    --depends "python3-dev" \
    --depends "postgresql" \
    --depends "redis-server" \
    --depends "nginx" \
    --depends "build-essential" \
    --depends "libpq-dev" \
    --after-install "${PROJECT_ROOT}/packaging/postinst.sh" \
    --before-remove "${PROJECT_ROOT}/packaging/prerm.sh" \
    --after-remove "${PROJECT_ROOT}/packaging/postrm.sh" \
    --config-files /etc/nginx/sites-available/sharkid.conf \
    --config-files /etc/systemd/system/sharkid-api.service \
    --config-files /etc/systemd/system/sharkid-worker.service \
    -C "${STAGING_DIR}" \
    -p "${PROJECT_ROOT}/build/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"

echo ""
echo "Build complete!"
echo ""
echo "Package: ${PROJECT_ROOT}/build/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
ls -lh "${PROJECT_ROOT}/build/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
