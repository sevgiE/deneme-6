import tensorflow as tf
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


image = tf.io.read_file("handwriting.jpg")
image = tf.image.decode_jpeg(image, channels=1)
image = tf.image.resize(image, (28, 28))
image = tf.keras.utils.normalize(image, axis=1)


predictions = model.predict(image)

class_index = tf.argmax(predictions[0])


class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
predicted_label = class_labels[class_index]

print("El yazısı sınıflandırılmış:", predicted_label)
