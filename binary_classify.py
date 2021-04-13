import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

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
