.env
``` bash
REACT_APP_BASE_URL=/api
DATABASE_URL=postgresql+psycopg2://admin:wpi_admin@localhost:5432/sharkid
```

```
[Unit]
Description=SharkID Inference Worker
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=cejason
WorkingDirectory=/home/cejason/SharkMQP26/backend
Environment="PATH=/home/cejason/SharkMQP26/backend/.venv/bin"
ExecStart=/home/cejason/SharkMQP26/backend/.venv/bin/python -m worker.worker
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```
. .venv/bin/activate && python -m uvicorn app.main:sio_asgi_app --reload --host 0.0.0.0 --port 8000
```