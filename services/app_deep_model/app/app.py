import os
import config
import socket
import requests
import datetime

from modules.deep_model import DeepModel
from flask import request, Response, render_template
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_socketio import SocketIO
from utils.app_celery import create_apps
from utils.database import get_database
from utils.queries import create_tables, create_connection
from utils.utils import get_class_instance_copy, get_weight_folder, process_thread
from utils.utils import cancel_active_task_from_service, convert_boolean

# Define urls to resources.
DEEP_MODEL_URL = os.environ.get("DEEP_MODEL_APP", "http://127.0.0.1:5051")
MYSQL_URL = os.environ.get("MYSQL_APP", "http://127.0.0.1:5050")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Define the app with celery.
app, celery = create_apps(app_name='deep_model')
app.secret_key = config.KEYMASTER

cors = CORS(app, origins=config.CORS_ADDR)
api = Api(app)

# Define SocketIO.
async_mode = None  # chooses automatically eventlet or gevent
socketio = SocketIO(
    app,
    async_mode=async_mode,
    logger=True,
    engineio_logger=True,
    cors_allowed_origins=config.CORS_ADDR)

# Define deep learning model and database objects.
model_inst = DeepModel()
db_train = get_database(
    host=config.HOST,
    user=config.USER,
    password=config.PASSWORD,
    show_info=True)


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", async_mode=socketio.async_mode)


@app.route('/broadcast', methods=['POST'])
def broadcast_to_client():
    """Broadcasts messages from celery task to client."""
    json_data = request.get_json(force=True)
    socketio.emit('service_log', {
        'data': json_data['data'], 'count': json_data['count'], 'service': json_data['service']})
    return {}, 200, {'ContentType': 'application/json'}


def send_message_from_service(msg, endpoint_broadcast, service_name):
    """Broadcasts messages from celery task to client."""
    requests.post(
        endpoint_broadcast,
        json={'data': msg, 'count': 0, 'service': service_name})


@app.route('/task_status', methods=['POST'])
def status():
    """Broadcasts status info to client for UI updates when
    a task terminates."""
    json_data = request.get_json(force=True)
    socketio.emit('task_status', {'status': json_data['status'], 'task_id': json_data['task_id']})
    return {}, 200, {'ContentType': 'application/json'}


@app.route('/getfiles', methods=['POST'])
def get_files_and_folders():
    """Returns folders and files from specified folder path."""
    json_data = request.get_json(force=True)
    # Get list of files in folder from server
    _, folders, files = os.walk(json_data['folder_path']).__next__()
    response = {'folders': folders, "files": files}
    return response, 200, {'ContentType': 'application/json'}


@app.route('/kill_tasks', methods=['GET'])
def purge_tasks(service_name='deepmodel'):
    """Kills all active celery tasks."""
    # Inspect active tasks.
    ins = celery.control.inspect()
    ins_act = ins.active()

    # Find the task associated to service_name process.
    for worker_name in ins_act.keys():
        # Iterate through tasks of current worker.
        try:
            tasks_meta = ins_act[worker_name]
            for meta in tasks_meta:
                task_id = meta['id']
                if service_name in task_id:
                    celery.control.revoke(task_id, terminate=True)
        except IndexError:
            pass

    return {}, 200, {'ContentType': 'application/json'}


class DeepModelTrain(Resource):
    def __init__(self):
        super(DeepModelTrain, self).__init__()

    @staticmethod
    @celery.task(bind=True, name='train_model_async')
    def run_task_async(task_self, augment, epoch_interval_saving, model_to_restore,
                       load_data_generator, solver_args):
        # Get object instance copy. This is to avoid have
        # same instance shared between services. Alternatively,
        # the main class of the instance may be inherited by
        # this class. However, the initialization of data would
        # be done every time this service is called.
        instance = get_class_instance_copy(model_inst)

        # Initiate solver
        send_message_from_service(
            'Initiating learning pipeline',
            endpoint_broadcast=DEEP_MODEL_URL + '/broadcast',
            service_name='train')
        instance.define_solver_with_data_generator(*solver_args)

        # Load variables of trained model if relevant.
        if model_to_restore:
            send_message_from_service(
                'Loading model variables',
                endpoint_broadcast=DEEP_MODEL_URL + '/broadcast',
                service_name='train')

            # Get last model weights saved from model name considered.
            weight_folder_from_model = get_weight_folder(
                model_to_restore=model_to_restore,
                path='./results/weights/')

            # Load model variables.
            instance.load_model_variables(
                model_to_restore=model_to_restore,
                weight_folder_from_model=weight_folder_from_model,
                load_data_generator=load_data_generator)

        # Run celery task with threaded message handler.
        send_message_from_service(
            'Initiating task',
            endpoint_broadcast=DEEP_MODEL_URL + '/broadcast',
            service_name='train')

        process_thread(
            'train',
            args=(augment, epoch_interval_saving),
            function=instance.train_model,
            task_id=task_self.request.id.__str__(),
            endpoint_status=DEEP_MODEL_URL + '/task_status',
            endpoint_broadcast=DEEP_MODEL_URL + '/broadcast')

        return Response({}, 200, mimetype='application/json')

    def post(self):
        json_data = request.get_json(force=True)

        if json_data['submit_button'] == 'cancel':
            # Revoke the task if the user wants to cancel the test.
            cancel_active_task_from_service(
                app, celery, service_name='deepmodel', service_type='train')
        else:
            # Store input parameters in database.
            db_train.insert_train_config(
                values=[json_data['task_id'], json_data['task_name'], json_data['model_name'],
                        json_data['model_to_restore'], json_data['epoch_interval_saving'],
                        convert_boolean(json_data['load_data_generator']),
                        convert_boolean(json_data['augmentation']),
                        str(datetime.datetime.utcnow())]
            )

            # Train the model asynchronously.
            solver_args = (json_data['model_name'], json_data['epochs'], json_data['batch_size'],
                           json_data['learning_rate'], json_data['input_size'],
                           json_data['metric_interval'], json_data['valid_step'],
                           json_data['inner_weight'], json_data['balance_train_samples_by_resampling'])

            _ = self.run_task_async.apply_async(
                args=(json_data['augmentation'], json_data['epoch_interval_saving'],
                      json_data['model_to_restore'], json_data['load_data_generator'],
                      solver_args),
                task_id=json_data['task_id'])

        return Response({}, 200, mimetype='application/json')


class DeepModelTest(Resource):
    def __init__(self):
        super(DeepModelTest, self).__init__()

    @staticmethod
    @celery.task(bind=True, name='test_model_async')
    def run_task_async(task_self, model_to_restore, solver_args):
        # Get object instance copy. This is to avoid having
        # same instance shared between services. Alternatively,
        # the main class of the instance may be inherited by
        # this class. However, the initialization of data would
        # be done every time this service is called.
        instance = get_class_instance_copy(model_inst)

        # Initiate solver
        send_message_from_service(
            'Initiating learning pipeline',
            endpoint_broadcast=DEEP_MODEL_URL + '/broadcast',
            service_name='train')
        instance.define_solver_with_data_generator(*solver_args)

        # Load variables of trained model if relevant.
        if model_to_restore:
            send_message_from_service(
                'Loading model variables',
                endpoint_broadcast=DEEP_MODEL_URL + '/broadcast',
                service_name='test')

            # Get last model weights saved from model name considered.
            weight_folder_from_model = get_weight_folder(
                model_to_restore=model_to_restore,
                path='./results/weights/')

            # Load model variables.
            instance.load_model_variables(
                model_to_restore=model_to_restore,
                weight_folder_from_model=weight_folder_from_model,
                load_data_generator=True)

        # Update state to processing.
        send_message_from_service(
            'Initiating task',
            endpoint_broadcast=DEEP_MODEL_URL + '/broadcast',
            service_name='test')

        process_thread(
            'test',
            args=(),
            function=instance.test_model,
            task_id=task_self.request.id.__str__(),
            endpoint_status=DEEP_MODEL_URL + '/task_status',
            endpoint_broadcast=DEEP_MODEL_URL + '/broadcast')

        return Response({}, 200, mimetype='application/json')

    def post(self):
        json_data = request.get_json(force=True)

        if json_data['submit_button'] == 'cancel':
            # Revoke the task if the user wants to cancel the test.
            cancel_active_task_from_service(
                app, celery, service_name='deepmodel', service_type='test')
        else:
            # Test the model.
            solver_args = ()

            _ = self.run_task_async.apply_async(
                args=(json_data['model_to_restore'], solver_args),
                task_id=json_data['task_id'])

        return Response({}, 200, mimetype='application/json')


def createdb():
    # Get MySQK connection.
    connection = create_connection(None, config.HOST, config.USER, config.PASSWORD, config.PORT, show_info=False)
    # Create database.
    connection.cursor().execute(f'CREATE DATABASE IF NOT EXISTS {config.DB_NAME}')
    # Create table to save user inputs.
    tmp = create_connection(config.DB_NAME, config.HOST, config.USER, config.PASSWORD, config.PORT, show_info=False)
    create_tables(tmp)


api.add_resource(DeepModelTrain, '/train')
api.add_resource(DeepModelTest, '/test')

if __name__ == '__main__':
    # Create the database and run the app.
    createdb()
    socketio.run(app, port=5051, host='0.0.0.0')
