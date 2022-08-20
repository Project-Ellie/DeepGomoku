import tensorflow as tf
from keras import losses, optimizers, metrics
import datetime as dt


class Trainer:

    def __init__(self,
                 policy_loss=losses.CategoricalCrossentropy(),
                 value_loss=losses.MeanSquaredError(),
                 optimizer=optimizers.Adam(learning_rate=1e-3),
                 train_metric=metrics.Mean('train_loss', dtype=tf.float32),
                 test_metric=None):

        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.optimizer = optimizer
        self.train_metric = train_metric
        self.test_metric = test_metric


    def train(self, model, train_data_set, epochs_per_train=100, report_every=100):
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in range(epochs_per_train):
            for x_train, pi_train, _ in train_data_set:
                self.train_step(model, x_train, pi_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_metric.result(), step=epoch)

            if epoch % report_every == 1:
                print(f'Epoch: {epoch}, Training: {self.train_metric.result()}')

        print(f'Epochs: {epochs_per_train}, Loss: {self.train_metric.result()}')

        self.train_metric.reset_states()

    @tf.function
    def train_step(self, model, inputs, pi_y):
        with tf.GradientTape() as tape:
            probs = model(inputs, training=True)
            total_loss = self.policy_loss(pi_y, probs)

        grads = tape.gradient(total_loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.train_metric(total_loss)
