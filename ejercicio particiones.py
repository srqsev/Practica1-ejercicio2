from perceptrón_simple import Perceptron

import csv
import numpy as np
import random

#!Funcion para leer los archivos, se crean los dataset de cada archivo
#!Despues se van a utilizar para crear las particiones

def makeDataSet(f):
    with open(f,'r') as firstFile:
    #*Convierte el reader en una lista con todos los datos
        return list(csv.reader(firstFile))

dataSet_1 = []
dataSet_2 = []
dataSet_3 = []
dataSet_4 = []

#*Se crean los dataset
dataSet_1 = makeDataSet('spheres1d10.csv')
dataSet_2 = makeDataSet('spheres2d10.csv')
dataSet_3 = makeDataSet('spheres2d50.csv')
dataSet_4 = makeDataSet('spheres2d70.csv')

myPerceptron = Perceptron()

#*Inicia con el primer dataset
myPerceptron.startPerceptron(dataSet_1, 5, "Grafico  Particiones")
myPerceptron.startPerceptron(dataSet_2, 10, "Grafico con 10% de perturbación")
myPerceptron.startPerceptron(dataSet_3, 10, "Grafico con 50% de perturbación")
myPerceptron.startPerceptron(dataSet_4, 10, "Grafico con 70% de perturbación")