import tensorflow as tf
from keras import losses, optimizers, metrics
import datetime as dt
from timeit import default_timer


class Trainer:

    def __init__(self,
                 model,
                 policy_loss=losses.CategoricalCrossentropy(),
                 value_loss=losses.MeanSquaredError(),
                 optimizer=optimizers.Adam(learning_rate=1e-3),
                 train_probs_metric=metrics.Mean('train_loss', dtype=tf.float32),
                 train_value_metric=metrics.Mean('train_loss', dtype=tf.float32),
                 test_probs_metric=metrics.Mean('train_loss', dtype=tf.float32),
                 test_value_metric=metrics.Mean('train_loss', dtype=tf.float32)):

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
                      f'p: {self.train_probs_metric.result().numpy():.5}, '
                      f'v: {self.train_value_metric.result().numpy():.5} - '
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
