import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image

data_dir = ('/content/Training')
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
for i in categories:
    path = os.path.join(data_dir, i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img)) 

#plt.imshow(img_array)
print(img_array.shape)

#display samples from each class: Glioma, Meningioma, No_Tumor, Pituitary_Tumor
plt.figure(figsize=(20, 16))

images_path = ['/glioma/Tr-glTr_0000.jpg', '/meningioma/Tr-meTr_0000.jpg', '/notumor/Tr-noTr_0000.jpg', '/pituitary/Tr-piTr_0000.jpg']

for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    img = cv2.imread(data_dir + images_path[i])
    img = cv2.resize(img, (250, 250))
    plt.imshow(img)
    plt.title(categories[i], fontdict={'fontsize': 15})