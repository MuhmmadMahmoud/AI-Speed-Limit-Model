import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:10000"
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