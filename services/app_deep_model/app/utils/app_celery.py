import os
from flask import Flask
from celery import Celery


def make_celery(app):
    """Creates the Celery app with Flask application."""
    celery = Celery(
        app.name,
        backend=app.config['RESULT_BACKEND'],
        broker=app.config['BROKER_URL'],
        result_persistent=True,
        task_result_expires = None,
        send_events = True,
        task_serializer='json',
        result_serializer = 'json',
        accept_content = ['json']
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_apps(app_name):
    # Build Flask app with Celery configuration.
    app = Flask(app_name)
    REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
    app.config['BROKER_URL'] = f'redis://{REDIS_HOST}:6379/0',
    app.config['RESULT_BACKEND'] = f'redis://{REDIS_HOST}:6379/1'

    # Build celery app.
    celery = make_celery(app)

    # Celery configuration.
    celery.conf.broker_transport_options = {'visibility_timeout': 3600 * 24}  # 24 hours
    celery.conf.worker_prefetch_multiplier = 1  # only prefetches one task at a time

    # If True, worker will remove item from queue at the
    # end of task (rather than the beginning). So, if worker
    # being killed, task is lost because still acknowledged
    # (even if it wasn't completed). acks_late setting would
    # be used when you need the task to be executed again if
    # the worker (for some reason) crashes mid-execution
    celery.conf.task_acks_late = False

    # If True, re-queue the message if the above event happens,
    # so you won't lose the task.
    celery.conf.task_reject_on_worker_lost = False
    celery.conf.task_acks_on_failure_or_timeout = True  # acknowledged even if tasks fail or time out

    # Prevent retrying to publish task message if connection loss.
    celery.conf.task_publish_retry = False

    return app, celery
