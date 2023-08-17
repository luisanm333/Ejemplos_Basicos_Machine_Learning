# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:31:51 2023

Base de datos Fashion-MNIST

Es un conjunto de datos de las imágenes de los artículos de la tienda Zalando

La base de datos contiene 70 000 imágenes (28 x 28 píxeles) en escala de grises en 10 categorías.
 
Se utilizan 60 000 imágenes para entrenar la red y 10 000 imágenes para prueba

Las clases son las siguientes:
0 T-shirt/top    3 Dress      6 Shirt       9 Ankle boot
1 Trouser        4 Coat       7 Sneaker
2 Pullover       5 Sanda      8 Bag

@author: luis_
"""

import os
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 

from tensorflow.keras.utils import to_categorical #para realizar transformaciones
from keras import layers

from sklearn.metrics import confusion_matrix
import itertools

#os.system('cls')

#Creando una lista que contiene los nombres de las clases
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',  'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',  'Ankle boot']


# Look at confusion matrix 
# Note: this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusión',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(range(10), class_names, rotation = 45)
    # plt.yticks(tick_marks, classes)
    plt.yticks(range(10), class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Observación')
    plt.xlabel('Predicción')
    
# Para graficar una imagen y su clase 
# Modificadas de https://www.tensorflow.org/tutorials/keras/classification

def plot_image(i, predictions_array, true_label, img):
    """
    Esta función imprime una imagen y su clase predecida con el modelo de la red neuronal artificial
    Si la clase predecida es correcta, la imprime de color azul
    Si la clase predecida es incorrecta, la imprime de color rojo
    
    Args: 
        img: es el conjunto de imagenes de la base de datos
        true_label: es el conjunto de clases de la base de datos
        i: es el índice de la imagen contenida en el conjunto de imagenes de la base de datos
        predictions_array: es un vector de la predicción de la clase 
                           que se hizo con el modelo de la imagen [i]
        
        plot_image(i, predictions[i], test_lbl, test_img)
        
    """
    #predictions_array, true_label, img = predictions_array, true_label[i], img[i]  }
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label[i]:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array), 
                                         class_names[true_label[i]]),
                                         color=color)
    
    
def plot_value_array(i, predictions_array, true_label):
    
    """
    Esta función imprime una grafica de barras de las predicciones de clase de una imagen
    
    Si la clase predecida es correcta, la imprime de color azul
    Si la clase predecida es incorrecta, la imprime de color rojo
    
    Args: 
        true_label: es el conjunto de clases de la base de datos
        i: es el índice de la imagen contenida en el conjunto de imagenes de la base de datos
        predictions_array: es un vector de la predicción de la clase 
                           que se hizo con el modelo de la imagen [i]
    
    plot_value_array(i, predictions[i],  test_lbl)
    """
    predictions_array, true_label = predictions_array, true_label[i] 
    plt.xticks(range(10), class_names, rotation = 90)   
    plt.yticks([])   
    thisplot = plt.bar(range(10), predictions_array, color="#c8bfff")   
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)   
    thisplot[predicted_label].set_color('red')   
    thisplot[true_label].set_color('blue')

"""
Importando la base de datos
"""
fashion_mnist = keras.datasets.fashion_mnist 

""" 
Asignando los datos de entrenamiento y los datos de prueba
"""
# train_img contiene los valores de 0 a 255 de cada pixel de las imagenes, 
# train_lbl contiene la clase de la imagen (0, 1, ..., 9)
# test_img y test_lbl contienen valores similares a los antes mencionados, pero para imagenes de prueba

(train_img, train_lbl), (test_img, test_lbl)  = fashion_mnist.load_data()

# Viendo una de las imagenes de la base de datos
# cmap es "color map" o mapa de colores, se elige gris para ver 
# el fondo blanco y los números en negro
Ex = 1
fig, f1 = plt.subplots() #plt.figure()
f1.imshow(train_img[Ex], cmap = plt.cm.binary)
f1.set_title(r'Ejemplo de prenda')
print("Ejemplo de clase:")
print(train_lbl[Ex])
plt.xticks([])  # para que no aparezca el eje x en la imagen
plt.yticks([])  # para que no aparezca el eje y en la imagen

# Graficando un histograma de los números de entrenamiento
fig, f2 = plt.subplots()
f2.hist(train_lbl)
f2.set_xlabel('clase')
f2.set_ylabel('Apariciones')
f2.set_title(r'Histograma datos de entrenamiento') 
plt.xticks(range(10), class_names, rotation=45)

# Escalando los vectores de entero de 8 bits en el rango [0,  255] a flotantes de 32 bits en el rango [0, 1]
train_img = train_img.astype('float32') 
test_img = test_img.astype('float32') 
train_img /= 255
test_img /= 255

# visualizando 15 imagenes
fig, f3 = plt.subplots() #plt.figure(figsize=(10,10))
for i in range(15):     
    plt.subplot(3,5,i+1)     
    plt.xticks([])  # para que no aparezca el eje x en la imagen
    plt.yticks([])  # para que no aparezca el eje y en la imagen   
    # plt.grid(False)     
    plt.imshow(train_img[i], cmap=plt.cm.binary)    
    plt.xlabel(class_names[train_lbl[i]]) 
    plt.show() 
fig.suptitle(r'Visualizando varias prendas')


"""
Definición del modelo
"""

model = keras.Sequential() 
model.add(layers.Flatten(input_shape=(28, 28))) 
#Flatten() transforma los tensores x de 2 dimensiones (28 x 28) a un vector de una dimensión (784)
model.add(layers.Dense(10, activation='sigmoid')) 
model.add(layers.Dense(10, activation='softmax'))

# Viendo la estructura del modelo 
model.summary()

"""
Configurando el método de aprendizaje
"""
# el método compile() configura el proceso de aprendizaje
# loss:  función de coste que se utilizará para evaluar el grado de 
# error entre las salidas calculadas y las salidas  deseadas de los datos de entrenamiento.
# sparse_categorical_crossentropy: funciona con clases de un solo dígito (0, 1, ..., 9)
# adam = Adam optimizer
# metrics: métrica que utilizará para monitprear el proceso de aprendizaje 
# (y prueba) de la red neuronal

model.compile(loss="sparse_categorical_crossentropy",  optimizer = "adam",  metrics = ['accuracy']) 

"""
Entrenando el modelo
"""
model.fit(train_img, train_lbl, epochs = 5)

"""
Evaluando el modelo
"""
test_loss, test_acc = model.evaluate(test_img, test_lbl)


"""
Predicción de datos con el modelo ya entrenado
"""
predictions = model.predict(test_img) #haciendo la predicción de todos los datos

i = 12 # eligiendo uno de ellos para saber su predicción
fig, f4 = plt.subplots()  # plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_lbl, test_img)

plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_lbl)
plt.show()
fig.suptitle(r'Predicción de una prenda con el modelo')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 3
num_cols = 5
num_images = num_rows*num_cols

fig, f5 = plt.subplots() # plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_lbl, test_img)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_lbl)
plt.tight_layout()
plt.show()
fig.suptitle(r'Predicción de varias prendas con el modelo')


"""
Desplegando la Matriz de confusión
"""
test_lbl = to_categorical(test_lbl, num_classes = 10) 

# Predict the values from the validation dataset
Y_pred = model.predict(test_img)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(test_lbl, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
fig, f4 = plt.subplots()
plot_confusion_matrix(confusion_mtx, classes = range(10))