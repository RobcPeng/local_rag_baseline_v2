# run.sh
#!/bin/bash
gunicorn wsgi:app -c gunicorn.conf.py