import datetime as dt

import keras.losses
import numpy as np
import tensorflow as tf


from aegomoku.interfaces import NeuralNet


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, name, **kwargs):
        super(ResnetIdentityBlock, self).__init__(name=name, **kwargs)
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return x


class GomokuModel(tf.keras.Model):
    """
    A naive model just to start with something
    """

    def __init__(self, input_size: int, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = 6  # kernel_size

        first, pot, agg_p, agg_v, flatten, dense, peel = self.create_model()

        self.first = first
        self.potentials = pot
        self.policy_aggregate = agg_p
        self.flatten = flatten
        self.value_aggregate = agg_v
        self.peel = peel
        self.dense = dense


    def call(self, sample, debug=False):

        # add two more channels filled with zeros. They'll be carrying the 'influence' of the surrounding stones.
        # That allows for arbitrarily deep chaining within our architecture

        sample = sample / self.input_size / self.input_size

        y = self.first(sample)
        for p1, p2 in self.potentials:
            y_in = y
            y = p1(y)
            y = p2(y+y_in)

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
        first = tf.keras.layers.Conv2D(
            name="initial",
            filters=32, kernel_size=self.kernel_size,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.elu,
            padding='same',
            input_shape=(self.input_size, self.input_size, 3))

        potentials = [(
            tf.keras.layers.Conv2D(
                name=f'potential_{i}_1',
                filters=32, kernel_size=self.kernel_size,
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.random_normal_initializer(),
                activation=tf.nn.elu,
                padding='same',
                input_shape=(self.input_size, self.input_size, 5)),

            tf.keras.layers.Conv2D(
                name=f'potential_{i}_2',
                filters=32, kernel_size=self.kernel_size,
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.random_normal_initializer(),
                activation=tf.nn.relu,
                padding='same',
                input_shape=(self.input_size, self.input_size, 5))

        )
            for i in range(7)
            ]

        policy_aggregate = tf.keras.layers.Conv2D(
            name="policy_aggregator",
            filters=1, kernel_size=1,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.relu,
            padding='same',
            input_shape=(self.input_size-2, self.input_size-2, 5))

        value_aggregate = tf.keras.layers.Conv2D(
            name="value_aggregator",
            filters=1, kernel_size=1,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer(),
            activation=tf.nn.tanh,
            padding='same',
            input_shape=(self.input_size-2, self.input_size-2, 5))

        flatten = tf.keras.layers.Flatten()
        dense = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)

        # 'peel' off the boundary
        peel = tf.keras.layers.Conv2D(
            name="border_off",
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
            bias_initializer=tf.constant_initializer(0.),
            trainable=False,
            padding='valid')

        return first, potentials, policy_aggregate, value_aggregate, flatten, dense, peel


def special_loss(y_true, y_pre):
    import keras.backend as k
    y_true = tf.cast(y_true, tf.float32)
    diff = y_true - y_pre
    return k.sum(y_true * diff * diff, axis=1)


class NeuralNetAdapter(NeuralNet):

    def __init__(self, input_size, *args):
        """
        :param input_size: size of the input signal: it's boardsize + 2, if you include the boundary!!
        """
        self.input_size = input_size
        self.board_size = input_size - 2
        self.policy_loss = tf.keras.losses.CategoricalCrossentropy()
        # self.policy_loss = tf.keras.losses.MeanSquaredError()
        #self.policy_loss = special_loss

        self.value_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.policy = GomokuModel(input_size=input_size, kernel_size=11)
        self.policy.build(input_shape=(None, input_size, input_size, 3))
        self.cut_off = 0.8  # TODO: provide from params
        super().__init__(*args)


    def get_advisable_actions(self, state):
        """
        :param state: the board's math representation
        :return: a list of integer move representations with probabilities close enough to the maximum (see: cut_off)
        """
        probs, _ = self.policy.call(state)
        max_prob = np.max(probs, axis=None)
        probs = probs.reshape(self.board_size * self.board_size)
        advisable = np.where(probs > max_prob * self.cut_off, probs, 0.)

        # ####################################################################################
        # TODO: remember randomly adding seemingly random moves to overcome potential bias!!!
        # ####################################################################################

        return [int(n) for n in advisable.nonzero()[0]]


    def evaluate(self, state, debug=False):
        return self.policy.call(state, debug)


    def save_checkpoint(self, folder, filename):
        raise NotImplementedError


    def load_checkpoint(self, folder, filename):
        raise NotImplementedError

    def train(self, train_examples, test_examples=None, epochs_per_train=1, report_every=100):
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        all_train_ds = self.create_dataset(train_examples)
        if test_examples is not None:
            all_test_ds = self.create_dataset(test_examples)

        # for epoch in tqdm(range(params.epochs_per_train), desc="   Training"):
        for epoch in range(epochs_per_train):
            for x_train, pi_train, v_train in all_train_ds:
                self.train_step(x_train, pi_train, v_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

            if epoch % report_every == 1:
                print(f'Epoch: {epoch}, Training: {self.train_loss.result()}, '
                      f'Test: {self.test_loss.result()}')

            if test_examples is not None:
                for x_test, pi_test, v_test in all_test_ds:
                    self.test_step(x_test, pi_test, v_test)
                with train_summary_writer.as_default():
                    tf.summary.scalar('test_loss', self.test_loss.result(), step=epoch)

        print(f'Epochs: {epochs_per_train}, Loss: {self.train_loss.result()}, ')
              #f'Test: {self.test_loss.result()}')

        self.train_loss.reset_states()

    def train_step(self, x, pi_y, v_y):
        with tf.GradientTape() as tape:
            p, v = self.policy(x, training=True)  # noqa: training should be recognized?!
            loss1 = self.policy_loss(pi_y, p)
            loss2 = self.value_loss(v_y, v)
            total_loss = 1 * loss1 + 1. * loss2
        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.train_loss(total_loss)

    def test_step(self, x_test, pi_test, v_test):
        p, v = self.call(x_test, training=False)  # noqa: training should be recognized?!
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
        all_train_ds = all_train_ds.shuffle(buffer_size=batch_size)
        return all_train_ds

    #
    #   Find a reasonable implementation for reasonable actions...;-)
    #
    def get_reasonable_actions(self, state):
        probs, _ = self.evaluate(state)
        max_prob = np.max(probs, axis=None)
        return probs[[probs > max_prob * 0.8]]
