import numpy as np 
import matplotlib.pyplot as plt
import os
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

acc = history1.history['categorical_accuracy']
val_acc = history1.history['val_categorical_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)

# Line Chart
# Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training categorical accurarcy')
plt.plot(epochs, val_acc, 'g', label='Validation categorical accurarcy')
plt.title('Training and Validation categorical accurarcy')
plt.legend()

plt.figure()

# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
import random

test_dir = '/content/Testing'
test_paths = []
for label in os.listdir(test_dir):
    for file in os.listdir(test_dir+'/'+label):
        test_paths.append(test_dir+'/'+label+'/'+file)
random.shuffle(test_paths)
# show an example of the list
print(test_paths[0])

from tensorflow.keras.preprocessing.image import load_img

def open_images(paths):
    '''
    Opens a batch of images, given the image path(s) as a list
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(64,64), color_mode='grayscale')
        image = np.array(image)/255.0
        images.append(image)
    return np.array(images)

def predict(i):
      images = open_images([test_paths[i]])
      predicted = model1.predict(images)[0]
      predicted = np.argmax(predicted)
      predicted = categories[predicted]
      label = test_paths[i].split('/')[-2]
      plt.imshow(images[0], cmap='gray')
      print('Predicted:', predicted)
      print('Actual:', label)

i = random.randint(0,len(test_paths))
predict(i)
