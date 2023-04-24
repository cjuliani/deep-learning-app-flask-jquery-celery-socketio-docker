import os
import time
import requests
import queue
import threading
import numpy as np
import pymysql.err as error


class MessageHandler:
    def __init__(self, queue_obj=None, event=None):
        """
        Args:
            queue_obj: the queue object enqueuing items to be dequeued
                outside the function.
            event: thread event used to trigger actions outside
                this function.
        """
        self.queue = queue_obj
        self.event = event

    def __call__(self, msg, show_info, print_end=False, break_line_before=False, delay=None):
        """
        Args:
            msg (str): the message to enqueue and display.
            show_info (bool): if True, print the message.
            break_line_before (bool): if True, break a line before the
                message.
            print_end (bool): if True, successive prints without
                new lines.
            delay (float or None): used to delay the enqueuing.
                This is relevant for messages not generated in
                loops, which have no interruptions between them.
                Note: as the enqueuing might be very fast, and the
                event trigger as well, the de-queuing process may
                not have time to dequeue all messages, included the
                last one ('completed') which stop the process. If
                'completed' is not delivered (dequeued), the process
                will not stop.
        """
        # Enqueue message.
        if self.queue:
            if delay:
                time.sleep(delay)  # delay the enqueuing
            if break_line_before:
                msg = '<br>' + msg
            self.queue.put(msg)
            self.trigger_event()

        # Print message.
        if show_info:
            if not print_end:
                print(msg)
            else:
                print(msg, end='\r')

    def trigger_event(self):
        self.event.set()  # set the event (to trigger q.get action in other thread)
        time.sleep(0.001)  # gives time to dequeue and request to manage the item
        while True:
            if self.queue.empty():
                # Make sure the queue is empty before processing. This is
                # to give time to the dequeue and request to proceed the
                # enqueued item before continuing the enqueue. Note that
                # the queue accept only 1 item at a time. This is useful
                # when items are enqueued very fast - the dequeued and
                # request processes may not terminate before another
                # item enqueues again. This while loop blocks this
                # enqueuing as long as the full dequeue (of 1 item) is
                # not terminated.
                break
        self.event.clear()  # make the event unset


def process_thread(service_name, function, args, task_id, endpoint_status, endpoint_broadcast):
    """Runs function as a thread with Celery worker. The thread allows
    triggering events everytime a print/message is emitted from the internal
    core process of the function. The message handler (Queue object) first
    collects the message, then an Event is triggered to unblock the while loop
    below, such that when the event is set (flag is True), the loop broadcast
    the message to the client. The message is broadcasted by the Queue once at
    a time, and the function blocks after the Queue collects the message as long
    as the message is not broadcasted from the loop below.

    Args:
        service_name (str): name of the service whose messages are
            to be displayed client side.
        function: the function to process (task).
        args (tuple): arguments of the function.
        task_id (str): the task id.
        endpoint_status (str): the endpoint for broadcasting the status of
            the task process.
        endpoint_broadcast (str): the endpoint for broadcasting the process
            info.
    """
    # Set up event and queue in message handler.
    q = queue.Queue(maxsize=1)
    event = threading.Event()
    msg_handler = MessageHandler(q, event)

    # Create a new thread for the function to process.
    t = threading.Thread(target=function, args=(args + (msg_handler,)))

    # Start thread and dequeue messages.
    t.start()

    count = 0
    while True:
        flag = event.wait(timeout=None)
        if flag:
            # Get message from queue.
            try:
                msg = q.get()
            except Exception:
                continue

            # Break loop given last message or is the thread dies.
            if (msg == 'completed') or (not t.is_alive()):
                break

            # POST log to display on webpage.
            requests.post(
                endpoint_broadcast,
                json={'data': msg, 'count': count, 'service': service_name})
            count += 1

    # Change status of task client-side
    tmp = {'status': 'completed', 'task_id': task_id}
    requests.post(endpoint_status, json=tmp)


def execute_query(connection, query):
    """Executes query via the execute method. A query is
    passed as a string.

    Args:
        connection: the database connection.
        query (str): the queried change committed to the database.
    """
    cursor = connection.cursor()  # a cursor object to fetch results
    try:
        cursor.execute(query)
        connection.commit()  # a change committed to the database
    except error.OperationalError as e:
        raise Exception(f"The error '{e}' occurred when executing: \n{query}")


def get_weight_folder(model_to_restore, path):
    """Returns the last weight folder built while training the
    specified model to restore.

    Args:
        model_to_restore (str): name of the model to restore (whose
            variables must be loaded).
        path (str): the folder path the model to be restored.
    """
    # Process model to restore
    if not model_to_restore:
        weight_folder = ''
    else:
        # Check the FIRST weight folder inside the model folder
        _, folders, _ = os.walk(path + model_to_restore).__next__()
        weight_folder = folders[-1]  # takes last weight folder saved
    return weight_folder


def find_active_task_id(celery, service_name, service_type):
    """Finds the task associated to the learning model process.

    Args:
        celery: the celery object.
        service_name (str): name of the service associated to
            the active task.
        service_type (str): the type of service (here: 'deepmodel')
            which is not to be confused with the service name (here:
            'train' or 'test').
    """
    # Inspect active tasks.
    ins = celery.control.inspect()
    ins_act = ins.active()

    task_id = None
    for worker_name in ins_act.keys():
        # Iterate through tasks of current worker.
        try:
            tasks_meta = ins_act[worker_name]
            for meta in tasks_meta:
                task_id = meta['id']
                if (service_type in task_id) and (service_name in task_id):
                    break
                else:
                    task_id = None
        except IndexError:
            pass

    return task_id


def cancel_active_task_from_service(app, celery, service_name, service_type):
    """Cancels and active celery task from a specified service.

    Args:
        app: the application object.
        celery: the celery object.
        service_name (str): name of the service associated to
            the active task.
        service_type (str): the type of service (here: 'deepmodel')
            which is not to be confused with the service name (here:
            'train' or 'test').
    """
    while True:
        # Find the task associated to deep_model process.
        task_id = find_active_task_id(celery, service_name, service_type)

        # Kill task.
        if task_id:
            celery.control.revoke(task_id, terminate=True)
            app.logger.info(f"Task '{task_id}' canceled.")
        else:
            break


class SomeCLass:
    def __init__(self):
        pass


def get_class_instance_copy(original):
    """Returns a copy of an instantiated class object."""
    # Create an empty class.
    inst = SomeCLass()

    # Copy attributes.
    for i, j in original.__dict__.items():
        # Set copy dictionary to instance dictionary.
        inst.__dict__[i] = j

    # Copy methods.
    for attr in dir(original):
        if not attr.startswith('__'):
            method = getattr(original, attr)
            setattr(inst, attr, method)
    return inst


def get_folder_or_create(path, name=None):
    """Returns a folder path, and eventually creates
    the folder not existing."""
    if name is None:
        out_path = path
    else:
        out_path = os.path.join(path, name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def load_data(folder_name, show_info=True):
    """Loads numpy data from provided path."""
    # Get feature results directory.
    path = get_folder_or_create(f"data/processed", folder_name)

    # Get files of folder and build the dictionary.
    data_dict = {}
    _, _, files = os.walk(path).__next__()
    for name in files:
        # Keep name of file without format.
        tmp = name.split('.')[0]
        data_dict[tmp] = np.load(os.path.join(path, name))

    if show_info:
        print(f"'{folder_name}' data loaded.")

    return data_dict


def convert_boolean(x):
    return 1 if x is True else 0
