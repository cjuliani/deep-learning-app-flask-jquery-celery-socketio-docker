import os
import random
import numpy as np
import pickle as pkl
import tensorflow as tf

from utils.utils import get_folder_or_create
from modules.models.augmentation import flip_and_rotate


class DataGenerator:
    """This module generates training and validation data
    with augmentation.

    Attributes:
        epochs_done (int): the number of epochs done while
            training.
        index_in_epoch (int): incremental epoch counter.
        train_indices (list): indices of train data.
        train_data (list): training arrays used as input to
            the learning model.
        train_classes (list): class IDs used as reference
            data for training.
        data (dict): data locally collected from a folder(s).
        validation_data (dict): validation data locally collected
            from a folder(s).
        valid_data (list): data used for validation.
        valid_classes (list): class IDs of validation data.
        valid_indices (list): indices of validation data.
        model_keys (list): keys used to retrieve a type of
            data among the provided ones ("A" and "B").
        num_classes (int): the number of classes used for filtering.
        valid_counter (int): a counter used to reset the validation
            dataset (shuffling) while training once the full dataset
            has been processed.
        model_data_indices (dict): the indices of data collected per
            class.
        balance_samples (bool): if True, balance the number of data
            used for training by resampling per class type ('clear'
            and 'not_clear').
        show_info (bool): if True, show information related to the process
            itself.
    """

    def __init__(self, data, validation_data, balance_train_samples_by_resampling=True,
                 show_info=True):
        self.epochs_done = 0
        self.index_in_epoch = 0
        self.train_indices = []
        self.train_data = []
        self.train_classes = []
        self.data = data
        self.valid_data = []
        self.valid_classes = []
        self.valid_indices = []
        self.validation_data = validation_data
        self.model_keys = list(data.keys())
        self.num_classes = len(self.model_keys)
        self.valid_counter = 0
        self.balance_samples = balance_train_samples_by_resampling
        self.model_data_indices = {key: {} for key in self.model_keys}
        self.show_info = show_info

    def build_dataset(self, msg_handler=None):

        if msg_handler: msg_handler('Dataset:', self.show_info, break_line_before=True)

        # Get maximum data length (for resampling).
        counts = [len(self.data[key]) for key in self.model_keys]
        max_num = np.max(counts)
        for i, key in enumerate(self.model_keys):
            # Get indices.
            train_num = counts[i]
            train_ind = list(range(train_num))

            if self.balance_samples:
                # Resample data for current class given maximum count
                # calculated among classes. This is to balance the
                # number of training data between classes.
                resample_count = max_num - train_num
                if resample_count > 0:
                    if self.show_info:
                        msg = (f"- '{key}' class has {train_num} train " +
                               f"samples (+ {resample_count} resampled)")
                        if msg_handler: msg_handler(msg, self.show_info, delay=0.01)
                    new_ids = np.random.choice(train_ind, resample_count).tolist()
                    train_ind = train_ind + new_ids
                else:
                    msg = f"- '{key}' class has {train_num} train samples (not resampled)"
                    if msg_handler: msg_handler(msg, self.show_info, delay=0.01)
            else:
                msg = f"- '{key}' class has ({train_num} train samples (not resampled)"
                if msg_handler: msg_handler(msg, self.show_info, delay=0.01)

            # Save indices.
            self.model_data_indices[key] = train_ind

        # Build up train data lists.
        self.setup_training_data(self.data)

        # Setup validation data.
        self.setup_validation_data()

    def setup_validation_data(self):
        """Defines validation data and classes.

        Individual data will be picked up by validation indices
        (shuffled randomly) during the training process.
        """
        self.valid_data = []
        self.valid_classes = []

        # Gather indices of all training classes.
        for j, key in enumerate(self.model_keys):
            self.valid_data.append([item for item in self.validation_data[key]])
            self.valid_classes.append([j] * len(self.validation_data[key]))

        self.valid_data = [item for sublist in self.valid_data for item in sublist]
        self.valid_classes = [item for sublist in self.valid_classes for item in sublist]

        # Shuffle indices.
        self.valid_indices = list(range(len(self.valid_data)))
        random.shuffle(self.valid_indices)

    def setup_training_data(self, data):
        """Defines training data and classes.

        Individual data will be picked up by training indices
        (shuffled randomly) during the training process.
        """
        self.train_data = []
        self.train_classes = []

        # Gather indices of all training classes.
        for j, key in enumerate(self.model_keys):
            indices = self.model_data_indices[key]
            self.train_data.append([data[key][i] for i in indices])
            self.train_classes.append([j] * len(indices))

        self.train_data = [item for sublist in self.train_data for item in sublist]
        self.train_classes = [item for sublist in self.train_classes for item in sublist]

        # Shuffle indices.
        self.train_indices = list(range(len(self.train_data)))
        random.shuffle(self.train_indices)

    def load_data(self, directory):
        """Loads the indices of training data from directory."""
        filename = os.path.join(directory, "train_indices_per_class")
        with open(filename, 'rb') as file:
            indices_dict = pkl.load(file)

        if self.show_info:
            print("Data generator loaded.")
        for key in self.model_keys:
            self.model_data_indices[key] = indices_dict[key]

        self.setup_training_data(self.data)

    def save_data(self, directory):
        """Saves the indices of training data to directory."""
        # Build directory if not existing.
        if not os.path.exists(directory):
            get_folder_or_create(directory)

        # Define data to save.
        data = self.model_data_indices
        name = "train_indices_per_class"

        # Save data.
        filename = os.path.join(directory, name)
        with open(filename, 'wb') as file:
            pkl.dump(data, file)

    def next_batch(self, batch_size, augment=True):
        """Returns a batch of data with respective class IDs
         and names.

         Args:
             batch_size (int): the batch size.
             augment (bool): if True, augment the training data.
         """
        # Define the start:end range of data to flush out for
        # training given the batch size.
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # Re-initialize and randomize training samples after every
        # epoch and continue flushing out batch data repeatedly.
        if self.index_in_epoch > len(self.train_indices) - 1:
            start = 0
            self.epochs_done += 1
            self.index_in_epoch = batch_size
            np.random.shuffle(self.train_indices)
        end = self.index_in_epoch

        data_batch, class_batch, class_name_batch = [], [], []
        for i in range(start, end):
            index = self.train_indices[i]
            data = self.train_data[index]
            class_ = self.train_classes[index]
            if augment:
                # Augment image
                selects1 = [random.choice(range(3))]
                data = flip_and_rotate(
                    array=data,
                    select=selects1,
                    angle=random.choice(np.arange(0, 360, 90)))

            data_batch.append(data)
            class_batch.append(class_)
            class_name_batch.append(self.model_keys[class_])

        # Stack data.
        data_batch = np.stack(data_batch)

        return data_batch, class_batch, class_name_batch

    def next_validation(self):
        """Returns a single validation data and respective class."""
        # Reset counter and randomize indices.
        if self.valid_counter > (len(self.valid_data) - 1):
            np.random.shuffle(self.valid_indices)
            self.valid_counter = 0
        # Get validation data index.
        index = self.valid_indices[self.valid_counter]
        self.valid_counter += 1
        # Return validation data at index.
        data = self.valid_data[index]
        class_ = self.valid_classes[index]
        return tf.expand_dims(data, axis=0), class_
