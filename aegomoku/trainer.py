import tempfile
from typing import Dict

import tensorflow as tf
from keras import losses, optimizers, metrics
import datetime as dt
from timeit import default_timer

from aegomoku.tfrecords import to_tfrecords, load_dataset


class Trainer:

    def __init__(self,
                 model,
                 policy_loss=losses.CategoricalCrossentropy(),
                 value_loss=losses.MeanSquaredError(),
                 optimizer=optimizers.Adam(learning_rate=1e-3),
                 train_probs_metric=metrics.Mean('train_probs', dtype=tf.float32),
                 train_value_metric=metrics.Mean('train_value', dtype=tf.float32),
                 test_probs_metric=metrics.Mean('test_probs', dtype=tf.float32),
                 test_value_metric=metrics.Mean('test_value', dtype=tf.float32)):

        self.model = model
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.optimizer = optimizer
        self.train_probs_metric = train_probs_metric
        self.train_value_metric = train_value_metric
        self.test_probs_metric = test_probs_metric
        self.test_value_metric = test_value_metric

    def train(self, train_data_set, epochs_per_train=100, report_every=100, v_weight=10.):
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        start = default_timer()
        for epoch in range(epochs_per_train):
            for x_train, pi_train, v_train in train_data_set:
                self.train_step(x_train, pi_train, v_train, v_weight)
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_probs_metric.result(), step=epoch)

            if report_every is not None and epoch > 0 and epoch % report_every == 0:
                elapsed = default_timer() - start
                print(f'Epoch: {epoch}, Training: '
                      f'p: {self.train_probs_metric.result().numpy():.7}, '
                      f'v: {self.train_value_metric.result().numpy():.7} - '
                      f'elapsed: {elapsed:.5}s')
                start = default_timer()

        print(f'Epoch: {epochs_per_train}, Training: '
              f'{self.train_probs_metric.result().numpy(), self.train_value_metric.result().numpy()}')

        self.train_probs_metric.reset_states()
        self.train_value_metric.reset_states()

    @tf.function
    def train_step(self, inputs, pi_y, v_y, v_weight):
        with tf.GradientTape() as tape:
            probs, value = self.model(inputs, training=True)
            train_loss_p = self.policy_loss(pi_y, probs)
            train_loss_v = self.value_loss(v_y, value)
            total_loss = train_loss_p + v_weight * train_loss_v

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_probs_metric(train_loss_p)
        self.train_value_metric(train_loss_v)


TERMINAL_OPPORTUNITY = "TERMINAL_OPPORTUNITY"
TERMINAL_THREAT = "TERMINAL_THREAT"
ENDGAME = "ENDGAME"
ALL_GAMEPLAY = "ALL_GAMEPLAY"
ALL_COURSES = [TERMINAL_OPPORTUNITY, TERMINAL_THREAT, ENDGAME, ALL_GAMEPLAY]


def create_curriculum(pickles_dir, batch_size, *focus) -> Dict:
    """
    Recursively searches the pickles_dir for "*.pickles* files and creates a curriculum using the given focus
    :param pickles_dir: directory containing the original game play pickle data
    :param batch_size:
    :return: dictionary of coiurses
    """

    # referencing the name is necessary as in Jupyter, enums are not necessarily equal although they appear to be.
    # The reason is obviously that they are instantiated more than once in a multi-process environment
    if len(focus) == 0:
        focus = ALL_COURSES

    def is_opportunity(_s, _p, v):
        return v > .9999

    def is_threat(_s, _p, v):
        return v < -.95

    def is_endgame(_s, _p, v):
        return -.8 > v or v > .8

    courses = {
        TERMINAL_OPPORTUNITY: {'title': "Terminal Opportunities",
                               'filter': is_opportunity},
        TERMINAL_THREAT: {'title': "Terminal Threats",
                          'filter': is_threat},
        ENDGAME: {'title': "General Endgame",
                  'filter': is_endgame},
        ALL_GAMEPLAY: {'title': "All Gameplay",
                       'filter': None}
    }

    for course_type, course in courses.items():
        if course_type in focus:
            print(f"Preparing course: {course['title']}")
            tfrecords_dir = tempfile.mkdtemp()
            tfrecords_files = to_tfrecords(pickles_dir, target_dir=tfrecords_dir, condition=course['filter'])

            ds = load_dataset(tfrecords_files, batch_size=batch_size)
            ds1 = load_dataset(tfrecords_files, batch_size=1)
            count = 0
            for _ in ds1:
                count += 1
            course['num_examples'] = count
            course['dataset'] = ds
            course['data_dir'] = tfrecords_dir
            print(f"Prepared course: {course['title']}: {count} examples")
            print()

    return courses
