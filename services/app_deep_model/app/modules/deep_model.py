import os
import config
import tensorflow as tf

from utils.utils import load_data
from modules.models.deep_model.main import Solver
from modules.models.deep_model.generator import DataGenerator


class DeepModel:
    """Class of the learning module filtering objects. The module
    separates 'clear' from 'unclear' objects whose representation
    is e.g. occluded (low completeness) or overlapped (or bounded)
    by other satellite objects.

    Available methods allow to train, tests and apply a learning
    model.

    Attributes:
        data_generator (object): a module generating train batches.
        solver (object): training module containing the learning model
            and the iterative process for training and testing the model.
        show_info (bool): if True, show information related to the process
            itself.
    """
    def __init__(self, show_info=True):
        """Initialize a trainer with its respective learning model,
        and a generator of training data."""
        self.show_info = show_info
        self.data_generator = None
        self.solver = None

        # Delete eventual session from a learning model.
        tf.keras.backend.clear_session()

        # Run mode.
        tf.config.run_functions_eagerly(True)

    def define_solver_with_data_generator(
            self, model_name=None, epochs=1, batch_size=32, learning_rate=1e-4,
            input_size=128, metric_interval=1, valid_step=5.,
            inner_weight=1e-4, balance_train_samples_by_resampling=True,
            msg_handler=None):
        # Build data generator after collecting data.
        self.data_generator = DataGenerator(
            data=load_data('train'),
            validation_data=load_data('validation'),
            balance_train_samples_by_resampling=balance_train_samples_by_resampling,
            show_info=self.show_info)

        # Define trainer object.
        self.solver = Solver(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            in_shape=(input_size, input_size, 3),
            metric_interval=metric_interval,
            valid_step=valid_step,
            inner_weight=inner_weight,
            generator=self.data_generator,
            model_name=model_name,
            msg_handler=msg_handler,
            show_info=self.show_info)

    def train_model(self, augment, epoch_interval_saving=1, msg_handler=None):
        """Trains the available learning model from trainer
        object. Eventually restore previously saved variables
        of the model prior to training it.

        Args:
            augment (bool): if True, augment training data while
                generating them.
            epoch_interval_saving (int): epoch interval defining when
                to save variables of the model while training. If 0,
                no saving is performed.
            msg_handler: the message handler which enqueues and deliver
                text outside this function (via a thread event) to another
                function (or thread). This text will then be dequeued and
                displayed.
        """
        # Train the model.
        self.solver.train(
            augment=augment,
            epoch_interval_saving=epoch_interval_saving,
            msg_handler=msg_handler)

    def test_model(self, msg_handler=None):
        """Tests the available learning model from trainer
        object. Eventually restore previously saved variables
        of the model if it was not done when initiating the
        model.

        Args:
            msg_handler: the message handler which enqueues and deliver
                text outside this function (via a thread event) to another
                function (or thread). This text will then be dequeued and
                displayed.
        """
        # Test the model.
        data = load_data('test')
        self.solver.test(
            data=data["clear"],
            classes=[0] * len(data["clear"]),
            msg_handler=msg_handler)

    def load_model_variables(self, model_to_restore=None, weight_folder_from_model=None,
                             load_data_generator=True, msg_handler=None):
        """Loads variables of a saved learning model.

        Args:
            model_to_restore (str): name of the model to restore.
            weight_folder_from_model (str): name of the folder containing
                the name to restore.
            load_data_generator (bool): if True, load data indices
                used by generator to produce batches from a previously
                saved model. This is to avoid any data leak if the
                model in question is trained again.
            msg_handler: the message handler which enqueues and deliver
                text outside this function (via a thread event) to another
                function (or thread). This text will then be dequeued and
                displayed.
        """
        # Load model variables.
        self.solver.learning_model.load_variables(
            directory=os.path.join(config.WEIGHT_PATH, model_to_restore, weight_folder_from_model),
            msg_handler=msg_handler,
            show_info=self.show_info)

        # Load data generator indices.
        # INFO: Each time learning model from solver is changed
        # by a new loaded model, data indices related to that
        # new model must change as well. These indices are used
        # if a new model is trained based on a previously trained
        # one (to improve learning and avoid any data leak).
        if load_data_generator:
            self.data_generator.load_data(
                directory=os.path.join(config.WEIGHT_PATH, model_to_restore))  # renew generator
            self.solver.generator = self.data_generator  # replace old generator in trainer
