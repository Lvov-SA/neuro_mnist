from NeuralNetwork import NeuralNetwork
import matplotlib
import matplotlib.pyplot
import numpy


# dataFile = open("data/mnist_train_100.csv")
# dataList = dataFile.readlines()
# dataFile.close()

# allValues = dataList[0].split(',')
# imageArray = numpy.asfarray(allValues[1:]).reshape((28,28))

# matplotlib.pyplot.imshow(imageArray,cmap='Greys',interpolation='None')
# matplotlib.pyplot.show()

# slisedInput = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01

# onodes=10
# targets = numpy.zeros(onodes) + 0.01
# targets[int(allValues[0])] = 0.99

def getKey(array, max):
    k = 0
    for a in array:
        if a == max:
            return k
        k+=1

inputNodes = 784
hiddenNodes = 100
outputNodes = 10

learningRate = 0.3
net = NeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)

dataFile = open("data/mnist_train_100.csv")
trainingDataList = dataFile.readlines()
dataFile.close()

for record in trainingDataList:
    allValues = record.split(',')
    slisedInput = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(outputNodes) + 0.01
    targets[int(allValues[0])] = 0.99
    net.train(slisedInput, targets)

dataFile = open("data/mnist_test_10.csv")
testDataSet = dataFile.readlines()
dataFile.close()

for record in testDataSet:
    allValues = record.split(',')
    print("Маркер:")
    print(allValues[0])
    imageArray = numpy.asfarray(allValues[1:]).reshape((28,28))

    matplotlib.pyplot.imshow(imageArray,cmap='Greys',interpolation='None')
    matplotlib.pyplot.show()
    slisedInput = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    print("Результат работы сети:")
    res = net.query(slisedInput)
    print(getKey(res,max(res)))