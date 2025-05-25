import multiprocessing
import os

# Gunicorn configuration
port = int(os.environ.get("PORT", 10000))  # Get PORT from environment variable, default to 10000
bind = f"0.0.0.0:{port}"
workers = 1  # Using 1 worker for free tier
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "speed_limit_app"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None 