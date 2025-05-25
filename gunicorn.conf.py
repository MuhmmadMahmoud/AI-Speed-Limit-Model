# Gunicorn configuration file
import multiprocessing

# Number of worker processes
workers = 1  # Using 1 worker to minimize memory usage
worker_class = 'sync'
worker_connections = 1000
timeout = 120  # Increased timeout for model loading
keepalive = 2

# Server socket
bind = "0.0.0.0:10000"  # Render will override this with PORT env var

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'speed_limit_app'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory management
max_requests = 1000
max_requests_jitter = 50
worker_tmp_dir = '/dev/shm'  # Use RAM for temporary files 