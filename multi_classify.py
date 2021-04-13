from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class MultiClassifier:

    def __init__(self, num_words, epochs):
        self.num_words = num_words
        self.epochs = epochs
        self.model = None
        self.eval = False if epochs == 20 else True

    def load_data(self):
        return reuters.load_data(num_words=self.num_words)

    def get_text(self, data):
        word_id_index = reuters.get_word_index()
        id_word_index = dict([(id, value) for (value, id) in word_id_index.items()])
        return ' '.join([id_word_index.get(i - 3, '?') for i in data])

    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i,sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    def to_one_hot(self, labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i,label in enumerate(labels):
            results[i, label] = 1
        return results

    def plt_loss(self, history):
        plt.clf()
        loss = history.histroy['loss']
        val_loss = history.histroy['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plt_accuracy(self, history):
        plt.clf()
        acc = history.histroy['accuracy']
        val_acc = history.histroy['val_accuracy']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self):
        results = self.model.evaluate(self.x_test, self.one_hot_test_labels)
        print('evaluate test data:')
        print(results)


    def train(self):
        (train_data, train_labels), (test_data, test_labels) = self.load_data()
        print(len(train_data))
        print(len(test_data))
        print(train_data[0])
        print(train_labels[0])
        print(self.get_text(train_data[0]))

        self.x_train = x_train = self.vectorize_sequences(train_data)
        self.x_test = x_test = self.vectorize_sequences(test_data)

        self.one_hot_train_labels = one_hot_train_labels = self.to_one_hot(train_labels)
        self.one_hot_test_labels = one_hot_test_labels = self.to_one_hot(test_labels)

        model = self.model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]

        y_val = one_hot_train_labels[:1000]
        partial_y_train = one_hot_train_labels[1000:]

        history = model.fit(partial_x_train, partial_y_train, epochs=self.epochs, batch_size=512, validation_data=(x_val, y_val))



        if self.eval:
            self.evaluate()
            print(self.model.predict(x_test))
        else:
            self.plt_loss(history)
            self.plt_accuracy(history)


classifier = MultiClassifier(num_words=10000, epochs=9)
classifier.train()