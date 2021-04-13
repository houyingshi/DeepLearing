import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt

imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])


def get_max_word_num(data):
    """获取所有评论中字数最多的数量"""
    return max([max(sequence) for sequence in data])


def get_text(comment_num):
    """将数字形式的评论转化为文本"""
    # word_index = tf.keras.datasets.imdb.get_word_index()
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    text = ' '.join([reverse_word_index.get(i - 3, '?') for i in comment_num])
    return text


def vectorize_sequences(sequences, diamension = 10000):
    results = np.zeros((len(sequences), diamension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results


max_word_num = get_max_word_num(train_data)
print(max_word_num)
comment = get_text(train_data[0])
print(comment)

x_train = vectorize_sequences(train_data)
print(x_train[0])
print(len(x_train[0]))
x_test = vectorize_sequences(test_data)
print(x_test[0])
print(len(x_test[0]))

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs= 4, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(results)

result = model.predict(x_test)
print(result)