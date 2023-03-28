#!/usr/bin/env python
# coding: utf-8

# In[62]:


import urllib.request
import zipfile
import io
import pandas as pd
#conda install -c conda-forge keras-preprocessing #Installs keras preproccesing in annaconda

# URL of the zip file on GitHub
url = 'https://github.com/Valderum/AI_Group_Project/archive/refs/heads/main.zip'

# Download the zip file
with urllib.request.urlopen(url) as response:
    zip_file = io.BytesIO(response.read())

# Extract the CSV file from the zip file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    csv_file = zip_ref.open('AI_Group_Project-main/Moto_Train.csv')

# Read the CSV file into a Pandas dataframe
df = pd.read_csv(csv_file)

# Display the dataframe
df.head()


# In[ ]:





# In[63]:


import urllib.request
import zipfile
import io
import os
import pandas as pd
# URL of the zip file on GitHub
url = 'https://github.com/Valderum/AI_Group_Project/raw/main/MotorcycleFolder.zip'

# Download the zip file
with urllib.request.urlopen(url) as response:
    zip_file = io.BytesIO(response.read())

# Extract the images from the zip file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall()  # extract all files from the zip file
    for i in range(1, 82):
        img_file = 'MotorcycleFolder/Motorcycle-{}.png'.format(i)
        if os.path.exists(img_file):
            # process the image here
            # for example, you can use a library like PIL to read the image
            # or you can save the image to a local directory
            pass
        else:
            print(f"Error: {img_file} not found.")

# Extract the CSV file from the zip file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    csv_file = zip_ref.open('Moto_Train.csv')

# Read the CSV file into a Pandas dataframe
df = pd.read_csv(csv_file)

# Display the dataframe
df.head()
df['filename']="Motorcycle-"+df["ID"].astype(str)+".png"


# In[64]:


TRAIN_PCT = 0.9
TRAIN_CUT = int(len(df) * TRAIN_PCT)

df_train = df[0:TRAIN_CUT]
df_validate = df[TRAIN_CUT:]

print(f"Training size: {len(df_train)}")
print(f"Validate size: {len(df_validate)}")


# In[65]:


training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip=True,
  vertical_flip=True,
  fill_mode='nearest')

train_generator = training_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='MotorcycleFolder/',
        x_col="filename",
        y_col="motorcycle_present",
        target_size=(120, 120),
        batch_size=32,
        class_mode='input')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

val_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_validate,
        directory='MotorcycleFolder/',
        x_col="filename",
        y_col="motorcycle_present",
        target_size=(120, 120),
        class_mode='input')


# In[2]:


from tensorflow.keras.callbacks import EarlyStopping
import time
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 256x256
    # with 3 bytes color.
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
        input_shape=(120, 120, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.summary()

epoch_steps = 250 # Number of batches to process for each epoch
validation_steps = len(df_validate) // 32

model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=5, verbose=1, mode='auto',
        restore_best_weights=True)

start_time = time.time()

history = model.fit(
    train_generator,  
    verbose=1, 
    validation_data=val_generator, 
    callbacks=[monitor], 
    epochs=25,
    steps_per_epoch=epoch_steps,
    validation_steps=validation_steps
)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))


# In[4]:


from tensorflow.keras.callbacks import EarlyStopping
import time

import tensorflow as tf
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 120x120
    # with 3 bytes color.
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
        input_shape=(120, 120, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.summary()

epoch_steps = 250 # Number of batches to process for each epoch
validation_steps = len(df_validate) // 32

model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=5, verbose=1, mode='auto',
        restore_best_weights=True)

start_time = time.time()

history = model.fit(
    train_generator,  
    verbose=1, 
    validation_data=val_generator, 
    callbacks=[monitor], 
    epochs=25,
    steps_per_epoch=epoch_steps,
    validation_steps=validation_steps
)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




