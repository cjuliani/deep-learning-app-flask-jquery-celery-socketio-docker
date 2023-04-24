import os
import time
import config
import tensorflow as tf

from datetime import datetime
from utils.utils import get_folder_or_create
from modules.models.deep_model.model import Model
from modules.models.metrics import write_loss_summaries, write_metric_summaries


class Solver:
    """Trainer module iterating the training and testing processes.
    This module has a learning model defined.

    Attributes:
        model_name (str): name of the model to save.
        epochs (int): the maximum number of training epochs.
        batch_size (int): the batch size.
        valid_step (int or float): the training step increment at
            which the learning model is tested on validation data.
        generator (object): the training data generator.
        learning_model (object): the learning model to optimize.
        summary_path (str): the path folder to save summary results.
        summary_train_path (str): the saving path for summaries of
            training losses and metrics.
        summary_validation_path (str): the saving path for summaries
            of validation losses and metrics.
        sum_train_writer (object or None): a summary file writer for
            training.
        sum_validation_writer (object or None): a summary file writer
            for validation.
        model_dir (str): the path to model folder where e.g. model
            weights can be saved.
        show_info (bool): if True, show information related to the process
            itself.
    """
    def __init__(self, epochs, batch_size, learning_rate, inner_weight,
                 in_shape, valid_step, metric_interval, generator, model_name=None,
                 msg_handler=None, show_info=True):
        self.model_name = model_name if model_name else 'undefined_model_name'

        if not model_name:
            msg = ("WARNING: Model name was not provided. Default name " +
                   "given to current model is 'undefined_model_name'.")
            if msg_handler: msg_handler(msg, True, break_line_before=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.valid_step = valid_step
        self.generator = generator
        self.summary_train_path = get_folder_or_create(
            path=os.path.join(config.SUMMARY_PATH, self.model_name),
            name='train')
        self.summary_validation_path = get_folder_or_create(
            path=os.path.join(config.SUMMARY_PATH, self.model_name),
            name='validation')
        self.model_dir = get_folder_or_create(
            path=config.WEIGHT_PATH,
            name=self.model_name)
        self.sum_train_writer = None
        self.sum_validation_writer = None
        self.show_info = show_info

        # Build dataset and learning model.
        self.generator.build_dataset(msg_handler)
        self.learning_model = Model(
            in_shape=in_shape,
            lr=learning_rate,
            metric_interval=metric_interval,
            num_classes=self.generator.num_classes,
            inner_weight=inner_weight)

    def train(self, augment, epoch_interval_saving, msg_handler=None):
        """Trains the learning model.

        Args:
            augment (bool): if True, apply augmentation to data when
                being generated.
            epoch_interval_saving (int): the epoch interval at which
                the model weights are saved. For example, if the
                interval is 5, these weights are saved at every 5 epochs
                processed.
            msg_handler: the message handler which enqueues and deliver
                text outside this function (via a thread event) to another
                function (or thread). This text will then be dequeued and
                displayed.
        """
        # Save train indices from generator to current
        # weight folder. They can  be restored later for testing.
        ind_path = os.path.join(self.model_dir, "train_indices")
        if not os.path.exists(ind_path):
            self.generator.save_data(directory=self.model_dir)

        # Delete previous summary event files from given folder.
        # Useful if training experiments require using same
        # summary output directories.
        try:
            for directory in [self.summary_train_path, self.summary_validation_path]:
                existing_summary_files = os.walk(directory).__next__()[-1]
                if existing_summary_files:
                    for file in existing_summary_files:
                        os.remove(os.path.join(directory, file))
        except (PermissionError, StopIteration):
            pass

        # Create summary writers.
        self.sum_train_writer = tf.summary.create_file_writer(self.summary_train_path)
        self.sum_validation_writer = tf.summary.create_file_writer(self.summary_validation_path)

        step = 0
        for epoch in range(self.epochs):
            # Define number of training steps.
            mod = len(self.generator.train_indices) % self.batch_size
            num_steps = len(self.generator.train_indices) // self.batch_size
            if mod > 0:
                num_steps += 1

            for step in range(num_steps):
                start = time.time()

                step = step + (epoch * num_steps)

                # Generate batch of training data.
                data_batch, class_batch, _ = self.generator.next_batch(
                    batch_size=self.batch_size,
                    augment=augment)

                model_args = (
                    data_batch,
                    tf.cast(class_batch, tf.float32))

                (loss_vector, metrics, metric_names) = self.learning_model.train_step(
                    model_args=model_args,
                    optimize=tf.constant(True))

                write_loss_summaries(
                    values=loss_vector,
                    names=['total', 'cls', 'inner'],
                    writer=self.sum_train_writer,
                    step=tf.cast(step, tf.int64))

                write_metric_summaries(
                    values=metrics,
                    names=metric_names,
                    writer=self.sum_train_writer,
                    step=tf.cast(step, tf.int64))

                # Measure training loop execution time.
                end = time.time()
                speed = round(end - start, 2)

                msg = (f'epoch {epoch}/{self.epochs}, step {step}/ {num_steps} ({speed} secs)' +
                      ' - total: {:.3f} - class: {:.3f} - inner: {:.3f} - '.format(*loss_vector) +
                      ' - '.join(['{}: {:.3f}'.format(nm, val) for nm, val in zip(metric_names, metrics)]))
                if msg_handler: msg_handler(msg, self.show_info)

                # Display results at given interval.
                if (step % self.valid_step) == 0:

                    # Get validation.
                    data_batch, class_batch = self.generator.next_validation()

                    model_args = (
                        data_batch,
                        tf.cast(class_batch, tf.float32))

                    (loss_vector, metrics, metric_names) = self.learning_model.train_step(
                        model_args=model_args,
                        optimize=tf.constant(False))

                    # Write losses and metrics.
                    write_loss_summaries(
                        values=loss_vector,
                        names=['total', 'cls', 'inner'],
                        writer=self.sum_validation_writer,
                        step=tf.cast(step, tf.int64))

                    write_metric_summaries(
                        values=metrics,
                        names=metric_names,
                        writer=self.sum_validation_writer,
                        step=tf.cast(step, tf.int64))

            # Do not save model variables if increment is 0 per epoch.
            if epoch_interval_saving == 0:
                continue

            if (epoch % epoch_interval_saving) == 0:
                # Define folder to solve weights.
                basename = f'weights_epoch-{epoch}_step-{step}'
                weight_folder_name = basename + datetime.now().strftime('_%Y%m%d-%H%M')
                weight_folder_path = get_folder_or_create(
                    path=self.model_dir,
                    name=weight_folder_name)

                # Save model weights and checkpoint.
                self.learning_model.save_variables(directory=weight_folder_path)

        if msg_handler: msg_handler('completed', False)

    def test(self, data, classes, msg_handler=None):
        """Tests the learning model.

        Args:
            data (tensor or array): input tensor.
            classes (list or tensor or array) the reference data to
                tests the model against.
            msg_handler: the message handler which enqueues and deliver
                text outside this function (via a thread event) to another
                function (or thread). This text will then be dequeued and
                displayed.
        """
        data_size = len(data)
        metric_names = None
        stacked_losses = tf.TensorArray(tf.float32, size=data_size)
        stacked_metrics = tf.TensorArray(tf.float32, size=data_size)
        for i in range(data_size):
            # Display current losses.
            dat = data[i]
            model_args = (
                tf.cast(tf.expand_dims(dat, axis=0), tf.float32),
                tf.cast(classes[i], tf.int32))

            (loss_vector, metrics, metric_names) = self.learning_model.train_step(
                model_args=model_args,
                optimize=tf.constant(False))

            # Display and enqueue message.
            msg = (f"{i + 1}/{data_size}: " +
                    'total: {:.3f} - cls: {:.3f} - inner: {:.3f}'.format(*loss_vector))
            if msg_handler: msg_handler(msg, self.show_info)

            # Get average vectors.
            stacked_losses = stacked_losses.write(i, tf.constant(loss_vector))
            stacked_metrics = stacked_metrics.write(i, metrics)

        # Calculate average metrics.
        avg_losses = tf.reduce_mean(stacked_losses.stack(), axis=0)
        avg_metrics = tf.reduce_mean(stacked_metrics.stack(), axis=0)

        # Display and enqueue message.
        msg = f'data number: {data_size}'
        if msg_handler: msg_handler(msg, self.show_info)

        names = ["total_loss", "classification_loss", "inner_loss"]
        for val, name in zip(avg_losses, names):
            # Display and enqueue message.
            msg = "{}: {:.3f}".format(name, val)
            if msg_handler: msg_handler(msg, self.show_info)

        try:
            # Graph mode.
            convert_name = lambda x: str(x.numpy().decode('ascii'))
            metric_names = [convert_name(name) for name in metric_names]
        except AttributeError:
            # Eagerly mode.
            convert_name = lambda x: str(x)
            metric_names = [convert_name(name) for name in metric_names]

        # Display current losses.
        for val, name in zip(avg_metrics, metric_names):
            msg = "{}: {:.3f}".format(name, val)
            if msg_handler: msg_handler(msg, self.show_info)

        if msg_handler: msg_handler('completed', False)
