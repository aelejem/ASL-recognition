from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Rescaling # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2 as cv

dir_name = "your/dir"

image_size = (200, 200)
batch_size = 256
epoch = 10

train_ds, val_ds = image_dataset_from_directory(dir_name, seed=1, validation_split=0.1, subset="both", image_size=image_size, batch_size=batch_size)
train_ds.shuffle(buffer_size=10000)
train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential()
model.add(Input(shape=(200, 200, 3)))
model.add(Rescaling(1./255))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(28, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_ds, epochs=10)

model.save("your/dir/your_model_name.keras")
