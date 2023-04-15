import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical


# Set the path to the directory containing the image data
path = '/Users/tylerblack/Desktop/Image Repository/images'

# Set the list of classes
classes = ['Bus', 'Car', 'Hydrant', 'Other', 'Palm', 'Stair', 'Traffic Light', 'Bicycle', 'Bridge', 'CrossWalk', 'Motorcycle', 'Chimney']
num_classes = len(classes)

# Load the image data
x_train = []
y_train = []
x_test = []
y_test = []
# Set the target size for resizing
target_size = (32, 32)

for i, c in enumerate(classes):
    class_path = os.path.join(path, c)
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f != '.DS_Store']  # Exclude non-file and .DS_Store files
    num_train = int(len(images) * 0.8)
    num_test = len(images) - num_train
    for j, image in enumerate(images):
        img_path = os.path.join(class_path, image)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(target_size)  # Resize the image
        img = np.array(img)
        
        if j < num_train:
            x_train.append(img)
            y_train.append(i)
        else:
            x_test.append(img)
            y_test.append(i)


# Convert the data to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot the training and validation accuracy over time
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
