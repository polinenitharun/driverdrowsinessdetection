import tensorflow as tf
import numpy as np
from PIL import Image,ImageOps 
import os
import pandas as pd
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn import model_selection
import seaborn as sns
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

data=[]
label=[]
def dataset(image,label_value):
  image=ImageOps.grayscale(image)
  image=image.resize((52,52))
  image=np.array(image)
  data.append(image)
  label.append(label_value)


for i in os.listdir('C:/Users/14176/Desktop/ankit/Driver_Drowsiness_Detection-main/imagedataset/closed_eye'):
  image = Image.open('C:/Users/14176/Desktop/ankit/Driver_Drowsiness_Detection-main/imagedataset/closed_eye/'+i)
  dataset(image,0)
closed_datasize=len(data)
print("total number of closed images =",closed_datasize)
for i in os.listdir('C:/Users/14176/Desktop/ankit/Driver_Drowsiness_Detection-main/imagedataset/open_eye'):
  image = Image.open('C:/Users/14176/Desktop/ankit/Driver_Drowsiness_Detection-main/imagedataset/open_eye/'+i)
  dataset(image,1)
open_datasize=len(data)-closed_datasize
print("total number of open images =",open_datasize)
print("total number of images =",len(data))
print("total number of labels =",len(label))

len(data),len(label)

df=pd.DataFrame(label,columns=['target'])
Close = df[df['target']==0]['target'].count()
Open = df[df['target']==1]['target'].count()
Open_percent = Open/(Open+Close)
Close_percent = Close/(Close+Open)

var = df['target']
count = var.value_counts()
plt.bar(count.index, count)
plt.xticks(count.index, count.index.values)
plt.ylabel("Counts")
plt.title('target')
plt.show()

plt.figure(figsize=(20,20))
for i in range(6):
    plt.subplot(9,9,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data[i],cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(label[i])
plt.show()

data=np.array(data)
label=to_categorical(label)
print(data[0])
print(data.shape)
print(label.shape)

data=data/255.0

train_images,test_images,train_labels,test_labels=model_selection.train_test_split(data,label,test_size=0.33,random_state=42)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_images=train_images.reshape(list(train_images.shape) + [1])
test_images=test_images.reshape(list(test_images.shape)+[1])
print(train_images.shape)
print(test_images.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(52,52,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(train_images, train_labels, epochs=2,batch_size=128,validation_data = (test_images,test_labels))

model.evaluate(test_images,test_labels,batch_size=64)

plt.plot(history.history['accuracy'], color='black',)
plt.plot(history.history['val_accuracy'], color='red',)
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Confusion Matrix

# Predict the values from the validation dataset
Y_pred = model.predict(test_images)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(test_labels,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
confusion_mtx = pd.DataFrame(confusion_mtx , index = [i for i in range(2)] , columns = [i for i in range(2)])

# plot the confusion matrix
sns.heatmap(confusion_mtx, annot=True, linewidths=1,cmap="Wistia",linecolor="gray")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

model.save('models/train.h5', overwrite=True)

