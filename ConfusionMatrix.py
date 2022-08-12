import random
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import classification_report
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns

test_dir = './Testing'
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
test_paths = []
for label in os.listdir(test_dir):
    for file in os.listdir(test_dir+'/'+label):
        test_paths.append(test_dir+'/'+label+'/'+file)
random.shuffle(test_paths)
# show an example of the list
print(test_paths[0])

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

pred = []
actual = []
for i in test_paths:
    images = open_images([i])
    predicted = model1.predict(images)[0]
    predicted = np.argmax(predicted)
    predicted = categories[predicted]
    pred.append(predicted)
    label = i.split('/')[-2]
    actual.append(label)

#Classification Report
print(classification_report(actual, pred, target_names=categories))

data = {'y_Actual':    actual,
        'y_Predicted': pred}

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix1 = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

#Confusion Matrix Heatmap
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix1, fmt="d", annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()