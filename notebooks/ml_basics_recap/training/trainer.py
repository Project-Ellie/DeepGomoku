import tensorflow as tf


class Trainer:

    def __init__(self, the_model, train_data, test_data, loss_object=None):

        self.model = the_model
        self.train_ds = train_data
        self.test_ds = test_data

        self.loss_object = loss_object if loss_object is not None else tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')

    @tf.function
    def train_step(self, input_batch, labels_batch):
        with tf.GradientTape() as tape:
            predictions = self.model(input_batch, training=True)
            loss = self.loss_object(labels_batch, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels_batch, predictions)

    @tf.function
    def test_step(self, test_batch, labels_batch):
        predictions = self.model(test_batch, training=False)
        t_loss = self.loss_object(labels_batch, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels_batch, predictions)


    def train(self, epochs):
        for epoch in range(epochs):
            for stats in [self.train_loss, self.train_accuracy, self.test_accuracy, self.test_loss]:
                stats.reset_states()

            for train_batch, train_labels in self.train_ds:
                self.train_step(train_batch, train_labels)

            for test_batch, test_labels in self.test_ds:
                self.test_step(test_batch, test_labels)

            print(
                f'Epoch {epoch+1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result()},     '
                f'Test Loss: {self.test_loss.result()}, '
                f'Test Accuracy: {self.test_accuracy.result()}'
            )
