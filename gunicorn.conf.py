# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = 1
threads = 1
worker_class = "gthread"
timeout = 300
keepalive = 2
