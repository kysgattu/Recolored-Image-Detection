#!/usr/bin/env python
# coding: utf-8

# # DETECTION OF RECOLORED IMAGES USING  DEEP DISCRIMINATIVE MODEL

# ## Training The Model

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("dataset"))


# In[2]:


FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 # RGB color


# > - Use the dataset created by RecImDet.ipnyb for training the model

# In[3]:


filenames = os.listdir("dataset/trainingset")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'pic':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df['category']=df['category'].astype(str)


# In[7]:


df.dtypes


# In[8]:


sample = random.choice(filenames)
image = load_img("dataset/trainingset/"+sample)
plt.imshow(image)


# In[9]:


df['category'].value_counts().plot.bar()


# ![RecDeNet.jpg](attachment:RecDeNet.jpg)
# - **Input Layer**: It represent input image data. It will reshape image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.
# - **Convolution Layer**: This layer will extract features from image.
# - **Pooling Layer**: This layerreduce the spatial volume of input image after convolution.
# - **Fully Connected Layer**: It connect the network from a layer to another layer
# - **Output Layer**: It is the predicted values layer.

# In[10]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# **Callbacks**

# In[11]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# > **Early Stop**
#    - To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased

# In[12]:


earlystop = EarlyStopping(monitor='val_loss',patience=10)


# > **Learning Rate Reduction**
# - We will reduce the learning rate when then accuracy not increase for 2 steps

# In[13]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.05, 
                                            min_lr=0.00001)


# In[14]:


callbacks = [earlystop,learning_rate_reduction]


# In[15]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[16]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=4


# In[17]:


print(train_df.shape)
print(validate_df.shape)


# In[18]:


train_df['category'].value_counts().plot.bar()


# In[19]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


# In[20]:


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "dataset/trainingset", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)


# In[21]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "dataset/trainingset", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)


# **Fit the model**

# In[22]:


epochs=3 if FAST_RUN else 20
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    
)


# **Save the model**

# In[23]:


model.save_weights("RecImgDecNet.h5")


# **Visualize Training**

# In[24]:


fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
ax1.plot(history.history['accuracy'], color='y', label="Training accuracy")
ax1.plot(history.history['val_accuracy'], color='g',label="Validation accuracy")


#ax1.set_xticks(np.arange(1, epochs, 1))
#ax1.set_yticks(np.arange(0, 1, 0.1))
legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# ## Testing The Model

# **Preparing Test Data**

# In[25]:


test_filenames = os.listdir("dataset/testingset")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# In[26]:


print(nb_samples)


# **Creating Test Generator**

# In[27]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "dataset/testingset", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# **Predict**
#  

# - For categoral classication the prediction will come with probability of each category.

# In[28]:


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


# In[29]:


print(predict)


# In[30]:


threshold = 0.5
test_df['probability'] = predict
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)


# In[31]:


test_df['category'].value_counts().plot.bar()


# **Visualize the Test Results**

# In[32]:


sample_test = test_df #.head(5)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    probability = row['probability']
    img = load_img("dataset/testingset/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(11, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' '(' + "{}".format(round(probability, 2)) + ')')
plt.figtext(1,1,'Category(0):RECOLORED,Category(1):ORIGINAL',fontsize='large')    
plt.tight_layout()
plt.show()


# **Submission of Test Results to a CSV**

# In[33]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)


# In[34]:


print('\n \n Category(0):RECOLORED,Category(1):ORIGINAL \n \n')
print(test_df)


# In[ ]:





# In[ ]:




