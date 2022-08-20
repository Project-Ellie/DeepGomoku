import tensorflow as tf
import keras
from keras import layers
import keras.initializers.initializers_v2 as initializers
from aegomoku.policies.primary_detector import PrimaryDetector


class GomokuResnet(keras.Model):

    def __init__(self, board_size, num_sensor_filters: int, num_blocks: int, *args, **kwargs):
        self.board_size = board_size
        self.input_size = board_size + 2
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3), name="inputs")

        policy_aggregate = layers.Conv2D(
            name="policy_aggregator",
            filters=1, kernel_size=1,
            kernel_initializer=initializers.TruncatedNormal(seed=1, stddev=0.08),
            bias_initializer=tf.constant_initializer(0.),
            activation=tf.nn.relu,
            padding='same')

        peel = layers.Conv2D(
            name="border_off",
            filters=1, kernel_size=(3, 3),
            kernel_initializer=tf.constant_initializer([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
            bias_initializer=tf.constant_initializer(0.),
            padding='valid',
            trainable=False)

        detector = PrimaryDetector(self.board_size, activation=tf.keras.activations.tanh, name='heuristics')

        x = xp(1, num_sensor_filters, 11)(inputs)
        c = ct(1, 4, 5)(x)

        features = detector.call(inputs)
        for i in range(2, num_blocks+1):
            i1 = layers.concatenate([features, c], axis=-1)
            x1 = xp(i, 32, 5)(i1)
            c1 = ct(i, 4, 3)(x1)
            c = layers.Add(name=f"skip_{i}")([c1, c])

        x = policy_aggregate(c)
        y = peel(x)
        flatten = layers.Flatten()(y)
        softmax = layers.Softmax()(flatten)

        super().__init__(inputs=inputs, outputs=softmax, *args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)


def xp(seqno, filters, kernel):
    return layers.Conv2D(
        name=f"expand_{seqno}_{kernel}x{kernel}",
        filters=filters, kernel_size=kernel,
        kernel_initializer=initializers.TruncatedNormal(seed=1, stddev=0.08),
        bias_initializer=tf.constant_initializer(0.),
        activation=tf.nn.softplus,
        padding='same')


def ct(seqno, filters, kernel):
    return layers.Conv2D(
        name=f"contract_{seqno}_{kernel}x{kernel}",
        filters=filters, kernel_size=kernel,
        kernel_initializer=initializers.TruncatedNormal(seed=1, stddev=0.08),
        bias_initializer=tf.constant_initializer(0.),
        activation=tf.nn.softplus,
        padding='same')
