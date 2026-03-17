#!/bin/bash
set -e

echo "=== SharkID Post-Installation Setup ==="

# 1. Create sharkid system user if it doesn't exist
if ! id -u sharkid > /dev/null 2>&1; then
    echo "Creating sharkid system user..."
    useradd --system --home-dir /opt/sharkid --shell /bin/bash --comment "SharkID Service User" sharkid
fi

# 2. Generate .env file if it doesn't exist (preserve on upgrades)
if [ ! -f /opt/sharkid/backend/.env ]; then
    echo "Generating .env configuration..."

    # Generate random JWT secret
    JWT_SECRET=$(openssl rand -hex 32)

    # Generate random database password
    DB_PASSWORD=$(openssl rand -hex 16)

    # Create .env from template
    sed -e "s/__JWT_SECRET__/${JWT_SECRET}/g" \
        -e "s/__DB_PASSWORD__/${DB_PASSWORD}/g" \
        /opt/sharkid/packaging/env.template > /opt/sharkid/backend/.env

    echo "DB_PASSWORD=${DB_PASSWORD}" > /opt/sharkid/.db_password
    chmod 600 /opt/sharkid/.db_password
fi

# Load DB password
if [ -f /opt/sharkid/.db_password ]; then
    source /opt/sharkid/.db_password
fi

# 3. Setup PostgreSQL database
echo "Setting up PostgreSQL database..."
sudo -u postgres psql -tc "SELECT 1 FROM pg_user WHERE usename = 'sharkid'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER sharkid WITH PASSWORD '${DB_PASSWORD}';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'sharkid'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE sharkid OWNER sharkid;"

# 4. Set ownership before creating venv (package extraction creates dirs as root)
chown -R sharkid:sharkid /opt/sharkid/backend

# 5. Create Python virtual environment
if [ ! -d /opt/sharkid/backend/.venv ]; then
    echo "Creating Python virtual environment..."
    sudo -u sharkid python3 -m venv /opt/sharkid/backend/.venv
fi

# 5. Install Python dependencies
echo "Installing Python dependencies (this may take several minutes)..."

# Install PyTorch CPU version first
echo "Installing PyTorch (CPU)..."
sudo -u sharkid /opt/sharkid/backend/.venv/bin/pip install --upgrade pip
sudo -u sharkid /opt/sharkid/backend/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
echo "Installing other dependencies..."
sudo -u sharkid /opt/sharkid/backend/.venv/bin/pip install -r /opt/sharkid/backend/requirements.txt

# 6. Create data directory
mkdir -p /opt/sharkid/data
chown -R sharkid:sharkid /opt/sharkid/data

# 7. Set ownership
chown -R sharkid:sharkid /opt/sharkid/backend
chown -R sharkid:sharkid /var/www/sharkid

# 8. Initialize database tables
echo "Initializing database tables..."
cd /opt/sharkid/backend
sudo -u sharkid /opt/sharkid/backend/.venv/bin/python -c "
from app.db import Base, engine
Base.metadata.create_all(bind=engine)
print('Database tables created successfully')
" || echo "Warning: Could not initialize database tables. You may need to run this manually."

# 9. Configure nginx
echo "Configuring nginx..."
ln -sf /etc/nginx/sites-available/sharkid.conf /etc/nginx/sites-enabled/sharkid.conf
rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
nginx -t

# 10. Reload systemd and enable services
echo "Enabling and starting services..."
systemctl daemon-reload
systemctl enable sharkid-api.service
systemctl enable sharkid-worker.service
systemctl enable nginx.service

systemctl restart nginx.service
systemctl start sharkid-api.service
systemctl start sharkid-worker.service

echo ""
echo "SharkID installation complete!"
echo ""
echo "   http://$(hostname -I | awk '{print $1}')"
