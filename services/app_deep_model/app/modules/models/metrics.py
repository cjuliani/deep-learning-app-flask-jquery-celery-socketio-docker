import tensorflow as tf


class Precision(tf.keras.metrics.Metric):
    """The metric module calculating the precision score.

    Attributes:
        score: the precision nscore accumulated.
        counter: the total number of scores accumulated.
            The average precision score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average precision score is set to 0 (reset)
        multiclass (bool): if True, calculate the precision
            score per class.
    """

    def __init__(self, name=None, dtype=None, interval=1., multiclass=False, **kwargs):
        super(Precision, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="precision", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.multiclass = multiclass

    @staticmethod
    def precision(y_pred, y_true, axis=None):
        """Returns the precision value."""
        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        return tp / (tp + fp + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new precision
        score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        if self.multiclass:
            # Get positive class in current state,
            # from (B, num_classes) matrix.
            pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)  # (pos_classes)
            precision = self.precision(y_pred, y_true, axis=0)
            precision = tf.reduce_mean(tf.gather(precision, pos_ind))
        else:
            precision = self.precision(y_pred, y_true)

        self.score.assign_add(precision)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class Recall(tf.keras.metrics.Metric):
    """The metric module calculating the recall score.

    Attributes:
        score: the recall score accumulated.
        counter: the total number of scores accumulated.
            The average recall score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average recall score is set to 0 (reset)
        multiclass (bool): if True, calculate the recall
            score per class.
    """

    def __init__(self, name=None, dtype=None, interval=1., multiclass=False, **kwargs):
        super(Recall, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="recall", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.multiclass = multiclass

    @staticmethod
    def recall(y_pred, y_true, axis=None):
        """Returns the recall value."""
        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=axis)
        return tp / (tp + fn + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new recall score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, 5), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        if self.multiclass:
            # Get positive class in current state,
            # from (B, num_classes) matrix.
            pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)  # (pos_classes)
            recall = self.recall(y_pred, y_true, axis=0)
            recall = tf.reduce_mean(tf.gather(recall, pos_ind))
        else:
            recall = self.recall(y_pred, y_true)

        self.score.assign_add(recall)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class Specificity(tf.keras.metrics.Metric):
    """The metric module calculating the specificity score.

    Attributes:
        score: the specificity score accumulated.
        counter: the total number of scores accumulated.
            The average specificity score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average specificity score is set to 0 (reset)
        multiclass (bool): if True, calculate the specificity
            score per class.
    """

    def __init__(self, name=None, dtype=None, interval=1., multiclass=False, **kwargs):
        super(Specificity, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="specificity", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.multiclass = multiclass

    @staticmethod
    def specificity(y_pred, y_true, axis=None):
        """Returns the specificity value."""
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=axis)
        return tn / (tn + fp + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new specificity
        score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        if self.multiclass:
            # Get positive class in current state,
            # from (B, num_classes) matrix.
            pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)  # (pos_classes)
            specificity = self.specificity(y_pred, y_true, axis=0)
            specificity = tf.reduce_mean(tf.gather(specificity, pos_ind))
        else:
            specificity = self.specificity(y_pred, y_true)

        self.score.assign_add(specificity)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


def write_loss_summaries(values, names, writer, step):
    """Write loss summaries to tensorboard.

    Args:
        values (tensor): loss values.
        names (list): loss names.
        writer: a summary file writer.
        step (int): the current training step.
    """
    with writer.as_default():
        for name, loss in zip(names, values):
            tf.summary.scalar(name, loss, step=step)


def write_metric_summaries(values, names, writer, step):
    """Write metrics to tensorboard.

    Args:
        values (tensor): metric values.
        names (list): metric names.
        writer: a summary file writer.
        step (int): the current training step.
    """
    with writer.as_default():
        for name, val in zip(names, values):
            try:
                tf.summary.scalar(str(name.numpy().decode('ascii')), val, step=step)
            except AttributeError:
                # If eagerly mode considered.
                tf.summary.scalar(str(name), val, step=step)
