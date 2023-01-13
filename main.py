import tensorflow as tf
from tensorflow import keras

# Görüntüleri yükle ve etiketleri al
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Görüntüleri normalize et
x_train = x_train / 255.0
x_test = x_test / 255.0

# Modeli oluştur
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=5)

# Modeli test et
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# El yazısı görüntüsünü yükle
image = tf.io.read_file("handwriting.jpg")
image = tf.image.decode_jpeg(image, channels=1)
image = tf.image.resize(image, (28, 28))
image = tf.keras.utils.normalize(image, axis=1)

# Görüntüyü model ile sınıflandır
predictions = model.predict(image)

# En yüksek olasılık ile sınıflandırılmış sınıfın indeksini al
class_index = tf.argmax(predictions[0])

# Sınıf indeksini sınıf etiketine dönüştür
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
predicted_label = class_labels[class_index]

print("El yazısı sınıflandırılmış:", predicted_label)
