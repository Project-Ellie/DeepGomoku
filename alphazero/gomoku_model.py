import datetime as dt
import numpy as np
import tensorflow as tf

from alphazero.interfaces import NeuralNet


class GomokuModel(tf.keras.Model, NeuralNet):
    """
    A naive model just to start with something
    """
    def __init__(self, input_size: int, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size

        # The different layers
        self.first = None
        self.potentials = None
        self.policy_aggregate = None
        self.flatten = None
        self.value_aggregate = None
        self.peel = None
        self.dense = None

        # Algorithmic components
        self.policy_loss = tf.keras.losses.CategoricalCrossentropy()
        self.value_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        self.create_model()


    def call(self, sample, training=False, debug=False):  # noqa: currently not using any layers who care about training

        # add two more channels filled with zeros. They'll be carrying the 'influence' of the surrounding stones.
        # That allows for arbitrarily deep chaining within our architecture

        sample = sample / self.input_size / self.input_size

        y = self.first(sample)
        for potential in self.potentials:
            y = potential(y)

        if debug:
            print(f"Potential: {tf.reduce_sum(y).numpy()}")

        value_head = self.peel(self.value_aggregate(y))
        if debug:
            print(f"Value Head: {tf.reduce_sum(value_head).numpy()}")
        value = self.flatten(value_head)
        value = self.dense(value)

        logits = self.peel(self.policy_aggregate(y))
        if debug:
            print(f"Policy Head: {tf.reduce_sum(logits).numpy()}")
        pi = tf.nn.softmax(self.flatten(logits))

        return pi, value


    def create_model(self):

        # Compute the current player's total potential, can be arbitrarily repeated
        # to create some forward-looking capabilities
        self.first = tf.keras.layers.Conv2D(
            name="initial",
            filters=32, kernel_size=self.kernel_size,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size, self.input_size, 3))

        self.potentials = [
            tf.keras.layers.Conv2D(
                name=f'potential_{i}',
                filters=32, kernel_size=self.kernel_size,
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.random_normal_initializer(),
                activation=tf.nn.relu,
                padding='same',
                input_shape=(self.input_size, self.input_size, 5))
            for i in range(5)
        ]

        self.policy_aggregate = tf.keras.layers.Conv2D(
            name="policy_aggregator",
            filters=1, kernel_size=1,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size-1, self.input_size-1, 5))

        self.value_aggregate = tf.keras.layers.Conv2D(
            name="value_aggregator",
            filters=1, kernel_size=1,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.tanh,
            padding='same',
            input_shape=(self.input_size-1, self.input_size-1, 5))

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)

        # 'peel' off the boundary
        self.peel = tf.keras.layers.Conv2D(
            name="border_off",
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False)


    def train(self, train_examples, test_examples=None, n_epochs=1):
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        all_train_ds = self.create_dataset(train_examples)
        all_test_ds = self.create_dataset(test_examples)

        # for epoch in tqdm(range(params.epochs_per_train), desc="   Training"):
        for epoch in range(n_epochs):
            for x_train, pi_train, v_train in all_train_ds:
                self.train_step(x_train, pi_train, v_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

            if epoch % 100 == 1:
                print(f'Epoch: {epoch}, Training: {self.train_loss.result()}, '
                      f'Test: {self.test_loss.result()}')

            for x_test, pi_test, v_test in all_test_ds:
                self.test_step(x_test, pi_test, v_test)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)

        print(f'Epochs: {n_epochs}, Loss: {self.train_loss.result()}, '
              f'Test: {self.test_loss.result()}')

        self.train_loss.reset_states()

    def train_step(self, x, pi_y, v_y):
        with tf.GradientTape() as tape:
            p, v = self.call(x, training=True)  # noqa: training should be recognized?!
            loss1 = self.policy_loss(pi_y, p)
            loss2 = self.value_loss(v_y, v)
            total_loss = 1 * loss1 + 1. * loss2
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.train_loss(total_loss)

    def test_step(self, x_test, pi_test, v_test):
        p, v = self.call(x_test, training=True)  # noqa: training should be recognized?!
        loss1 = self.policy_loss(pi_test, p)
        loss2 = self.value_loss(v_test, v)
        total_loss = loss1 + loss2
        self.test_loss(total_loss)

    # TODO: This mustn't be a model method!
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


    def predict(self, state):
        return self.call(state)


    def save_checkpoint(self, folder, filename):
        pass


    def load_checkpoint(self, folder, filename):
        pass


    def get_advisable_actions(self, state):
        """
        :param state: the board's math representation
        :return: a list of integer move representations with probabilities close enough to the maximum (see: cut_off)
        """
        probs, _ = self.call(state)
        max_prob = np.max(probs, axis=None)
        probs = probs.reshape(self.board_size * self.board_size)
        advisable = np.where(probs > max_prob * self.cut_off, probs, 0.)
        return [int(n) for n in advisable.nonzero()[0]]




#
#    OLD STUFF
#

# class NeuralNetAdapter(NeuralNet):
#
#     def __init__(self, input_size, *args):
#         """
#         :param input_size: size of the input signal: it's boardsize + 2, if you include the boundary!!
#         """
#         self.input_size = input_size
#         self.policy_loss = tf.keras.losses.CategoricalCrossentropy()
#         self.value_loss = tf.keras.losses.MeanSquaredError()
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#         self.policy = GomokuModel(input_size=input_size, kernel_size=11)
#         self.policy.build(input_shape=(None, input_size, input_size, 3))
#         super().__init__(*args)
#
#
#     def get_advisable_actions(self, state):
#         """
#         :param state: the board's math representation
#         :return: a list of integer move representations with probabilities close enough to the maximum (see: cut_off)
#         """
#         probs, _ = self.call(state)
#         max_prob = np.max(probs, axis=None)
#         probs = probs.reshape(self.board_size * self.board_size)
#         advisable = np.where(probs > max_prob * self.cut_off, probs, 0.)
#         return [int(n) for n in advisable.nonzero()[0]]
#
#
#     def predict(self, state, debug=False):
#         return self.policy.call(state, debug)
#
#
#     def save_checkpoint(self, folder, filename):
#         raise NotImplementedError
#
#
#     def load_checkpoint(self, folder, filename):
#         raise NotImplementedError
#
#     def train(self, train_examples, test_examples=None, n_epochs=1):
#         current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
#         train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#         train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#
#         all_train_ds = self.create_dataset(train_examples)
#         all_test_ds = self.create_dataset(test_examples)
#
#         # for epoch in tqdm(range(params.epochs_per_train), desc="   Training"):
#         for epoch in range(n_epochs):
#             for x_train, pi_train, v_train in all_train_ds:
#                 self.train_step(x_train, pi_train, v_train)
#             with train_summary_writer.as_default():
#                 tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
#
#             if epoch % 100 == 1:
#                 print(f'Epoch: {epoch}, Loss: {self.train_loss.result()}')
#
#             for x_test, y_test in test_dataset:
#                 test_step(model, x_test, y_test)
#              with train_summary_writer.as_default():
#                  tf.summary.scalar('loss', test_loss.result(), step=epoch)
#                  tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
#
#         print(f'Epochs: {n_epochs}, Loss: {self.train_loss.result()}')
#
#         self.train_loss.reset_states()
#
#     def train_step(self, x, pi_y, v_y):
#         with tf.GradientTape() as tape:
#             p, v = self.policy(x, training=True)  # noqa: training should be recognized?!
#             loss1 = self.policy_loss(pi_y, p)
#             loss2 = self.value_loss(v_y, v)
#             total_loss = 1 * loss1 + 1. * loss2
#         grads = tape.gradient(total_loss, self.policy.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
#
#         self.train_loss(total_loss)
#
#     def test_step(self, x_test, y_test):
#         raise NotImplementedError()
#         predictions = self.policy.call(x_test)
#         loss = self.policy_loss(y_test, predictions)
#
#         test_loss(loss)
#         test_accuracy(y_test, predictions)
#
#     @staticmethod
#     def create_dataset(examples, num_subset: int = None, batch_size=1024):
#         subset = examples[num_subset] if num_subset is not None else examples
#         x_train = np.asarray([t[0] for t in subset], dtype=float)
#         pi_train = np.asarray([t[1] for t in subset])
#         v_train = np.asarray([t[2] for t in subset])
#         x_train_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
#         pi_train_ds = tf.data.Dataset.from_tensor_slices(pi_train).batch(batch_size)
#         v_train_ds = tf.data.Dataset.from_tensor_slices(v_train).batch(batch_size)
#         all_train_ds = tf.data.Dataset.zip((x_train_ds, pi_train_ds, v_train_ds))
#         return all_train_ds
#
#     #
#     #   Find a reasonable implementation for reasonable actions...;-)
#     #
#     def get_reasonable_actions(self, state):
#         probs, _ = self.predict(state)
#         max_prob = np.max(probs, axis=None)
#         return probs[[probs > max_prob * 0.8]]
