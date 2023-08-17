# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:00:54 2023

Se realiza la clasificación de imágenes de dígitos escritos a mano.

La base de datos contiene 60,000 elementos de entrenamiento y 10,000 para prueba.

Las imagenes están normalizadas a 20 × 20 píxeles centradas en imagenes de 28 x 28 pixeles
con valores entre [0, 255]

De igual forma, a cada imagen le corresponde una etiqueta que indica qué dígito representa 
(entre el 0 y el 9), es decir, a qué clase corresponde

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
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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


"""
Importando la base de datos
"""
mnist = tf.keras.datasets.mnist 

""" 
Asignando los datos de entrenamiento y los datos de prueba
"""
# x_train contiene los valores de 0 a 255 de cada pixel de las imagenes, 
# y_train contiene la clase de la imagen (0, 1, ..., 9)
# x_test y y_test contienen valores similares a los antes mencionados, pero para imagenes de prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data() 


# Viendo una de las imagenes de la base de datos
# cmap es "color map" o mapa de colores, se elige gris para ver 
# el fondo blanco y los números en negro
Ex = 50
fig, f1 = plt.subplots()
f1.imshow(x_train[Ex], cmap = plt.cm.binary)
f1.set_title(r'Ejemplo de digito')
print("Ejemplo de clase:")
print(y_train[Ex])

# Graficando un histograma de los números de entrenamiento
fig, f2 = plt.subplots()

f2.hist(y_train)
f2.set_xlabel('Digito')
f2.set_ylabel('Apariciones')
f2.set_title(r'Histograma datos de entrenamiento')       


# Escalando los vectores de entero de 8 bits en el rango [0,  255] a flotantes de 32 bits en el rango [0, 1]
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255

# Transformando los tensores x de 2 dimensiones (28 x 28) a un vector de una dimensión (784)
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784)


# Modificando las clases o etiquetas
# En lugar de tener un digito del 0 al 9, cada categoría se representa con
# un vector de 10 dimensiones, con elementos 0, excepto en la categría correspondiente
# Ex. La categoría 2 pasa a ser el vector 0 0 1 0 0 0 0 0 0 0 

y_train = to_categorical(y_train, num_classes = 10) 
y_test = to_categorical(y_test, num_classes = 10) 
print(y_train[Ex])

"""
Definición del modelo
"""
model = keras.Sequential() #Sequential permite la creación de una red neuronal básica
model.add(layers.Dense(10, activation='sigmoid',input_shape=(784,))) # solo tenemos que establecer la forma de los tensores para la primera capa
model.add(layers.Dense(10, activation='softmax')) 

# Viendo la estructura del modelo 
model.summary()


"""
Configurando el método de aprendizaje
"""
# el método compile() configura el proceso de aprendizaje
# loss:  función de coste que se utilizará para evaluar el grado de 
# error entre las salidas calculadas y las salidas  deseadas de los datos de entrenamiento.
# categorical_crossentropy: Computes the crossentropy loss between the labels and prediction
# sgd = stocastic gradient descent
# metrics: métrica que utilizará para monitprear el proceso de aprendizaje 
# (y prueba) de la red neuronal

model.compile(loss="categorical_crossentropy",  optimizer = "sgd",  metrics = ['accuracy']) 

"""
Entrenando el modelo
"""
model.fit(x_train, y_train, epochs = 5)

"""
Evaluando el modelo
"""
test_loss, test_acc = model.evaluate(x_test, y_test)

"""
Desplegando la Matriz de confusión
"""
# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
fig, f4 = plt.subplots()
plot_confusion_matrix(confusion_mtx, classes = range(10))

"""
Predicción de datos con el modelo ya entrenado
"""
predictions = model.predict(x_test) #haciendo la predicción de todos los datos

Ex2 = 1250 # eligiendo uno de ellos para saber su predicción
pred = np.argmax(predictions[Ex2])  
suma = np.sum(predictions[Ex2]) #comprobando que la suma de las componentes de la predicción sea 1
fig, f4 = plt.subplots()
f4.imshow(x_test[Ex2].reshape(28,28), cmap=plt.cm.binary) #visualizando el dato de prueba
f4.set_title(r'Digito que se está prediciendo')

