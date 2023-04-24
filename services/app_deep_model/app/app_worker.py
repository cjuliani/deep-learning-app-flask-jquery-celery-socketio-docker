import subprocess
import shlex

if __name__ == '__main__':
    # Start worker.
    cmd = "celery -A app.celery worker --loglevel=info -E"
    subprocess.call(shlex.split(cmd))  # to pass string to shell, use command-line argument as a separate list items