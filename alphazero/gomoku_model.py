import datetime as dt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from alphazero.interfaces import NeuralNet, TrainParams


class GomokuModel(tf.keras.Model):
    """
    A naive model just to start with something
    """

    def __init__(self, input_size: int, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size

        first, pot, agg, peel = self.create_model()
        self.first = first
        self.potentials = pot
        self.aggregate = agg
        self.peel = peel


    def call(self, sample):

        # add two more channels filled with zeros. They'll be carrying the 'influence' of the surrounding stones.
        # That allows for arbitrarily deep chaining within our architecture

        y = self.first(sample)
        for potential in self.potentials:
            y = potential(y)
        soft = self.peel(self.aggregate(y))

        value = tf.reduce_max(soft)
        pi = tf.nn.softmax(tf.keras.layers.Flatten()(soft))

        return pi, value


    def create_model(self):

        # Compute the current player's total potential, can be arbitrarily repeated
        # to create some forward-looking capabilities
        first = tf.keras.layers.Conv2D(
            filters=32, kernel_size=self.kernel_size,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size, self.input_size, 3))

        potentials = [
            tf.keras.layers.Conv2D(
                name=f'Potential_{i}',
                filters=32, kernel_size=self.kernel_size,
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.random_normal_initializer(),
                activation=tf.nn.relu,
                padding='same',
                input_shape=(self.input_size, self.input_size, 5))
            for i in range(5)
            ]

        aggregate = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size-1, self.input_size-1, 5))

        # 'peel' off the boundary and provide Q semantics with tanh
        peel = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
            bias_initializer=tf.constant_initializer(0.),
            activation=tf.nn.tanh,
            trainable=False)

        return first, potentials, aggregate, peel


class NeuralNetAdapter(NeuralNet):

    def __init__(self, input_size, *args):
        self.input_size = input_size
        self.policy_loss = tf.keras.losses.CategoricalCrossentropy()
        self.value_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.policy = GomokuModel(input_size=input_size, kernel_size=11)
        self.policy.build(input_shape=(None, input_size, input_size, 3))
        super().__init__(*args)

    def predict(self, state):
        return self.policy.call(state)


    def save_checkpoint(self, folder, filename):
        raise NotImplementedError


    def load_checkpoint(self, folder, filename):
        raise NotImplementedError

    # @tf.function
    def train(self, examples, params: TrainParams):
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        all_train_ds = self.create_dataset(examples)

        for epoch in tqdm(range(params.epochs_per_train), desc="   Training"):
            for x_train, pi_train, v_train in all_train_ds:
                self.train_step(x_train, pi_train, v_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

            # for x_test, y_test in test_dataset:
            #     test_step(model, x_test, y_test)
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', test_loss.result(), step=epoch)
            #     tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        print(f'Epochs: {params.epochs_per_train}, Loss: {self.train_loss.result()}')

        self.train_loss.reset_states()

    def train_step(self, x, pi_y, v_y):
        with tf.GradientTape() as tape:
            p, v = self.policy(x, training=True)  # noqa: training should be recognized?!
            loss1 = self.policy_loss(pi_y, p)
            loss2 = self.value_loss(v_y, v)
            total_loss = loss1 + loss2
        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.train_loss(total_loss)

    def test_step(self, x_test, y_test):
        raise NotImplementedError()
        # predictions = self.policy.call(x_test)
        # loss = self.policy_loss(y_test, predictions)

        # test_loss(loss)
        # test_accuracy(y_test, predictions)

    @staticmethod
    def create_dataset(examples, num_subset: int = None, batch_size=1024):
        subset = examples[num_subset] if num_subset is not None else examples
        x_train = np.asarray([t[0] for t in subset], dtype=float)
        pi_train = np.asarray([t[1] for t in subset])
        v_train = np.asarray([t[2] for t in subset])
        x_train_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
        pi_train_ds = tf.data.Dataset.from_tensor_slices(pi_train).batch(batch_size)
        v_train_ds = tf.data.Dataset.from_tensor_slices(v_train).batch(batch_size)
        all_train_ds = tf.data.Dataset.zip((x_train_ds, pi_train_ds, v_train_ds))
        return all_train_ds

    #
    #   Find a reasonable implementation for reasonable actions...;-)
    #
    def get_reasonable_actions(self, state):
        probs, _ = self.predict(state)
        max_prob = np.max(probs, axis=None)
        return probs[[probs > max_prob * 0.8]]
