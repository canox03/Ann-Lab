import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

import numpy as np

index = 1
test_image = test_images[index]

test_image = np.reshape(test_image, (1, 28, 28, 1 )) # batch(cuz trained on 32 batch SGD), x, y, rbg(can ignore)

prediction = model.predict(test_image)

predicted_label = np.argmax(prediction)

print("Predicted label:", predicted_label)

test_labels[1]

import numpy as np
import time



indices = [0, 1, 2, 3, 4]

for index in indices:
    test_image = test_images[index]

    test_image_reshaped = np.reshape(test_image, (1, 28, 28, 1))

    prediction = model.predict(test_image_reshaped)

    predicted_label = np.argmax(prediction)

    plt.imshow(test_image, cmap='gray')
    plt.axis('off')
    plt.show()

    print("Predicted label:", predicted_label)


