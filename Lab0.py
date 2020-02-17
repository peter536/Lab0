import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
        #pass   #TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
        #pass   #TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 40, minibatches = True, mbs = 100):
        ####### implement minibatching ###########
        # For a given number of epochs
        for i in range(epochs):

            # do a forward pass
            layer1, layer2 = self.__forward(xVals)
            
            # calculate layer 2 error and delta
            layer2_error = yVals - layer2
            layer2_delta = layer2_error * self.__sigmoidDerivative(layer2)

            # calculate layer 1 error and delta, given layer 2 delta
            layer1_error = np.dot(layer2_delta, self.W2.T)
            layer1_delta = layer1_error * self.__sigmoidDerivative(layer1)

            # change the weights!
            self.W2 += np.dot(layer1.T, layer2_delta) * self.lr
            self.W1 += np.dot(xVals.T, layer1_delta) * self.lr
            
            if (i % 20 == 0):
                print("Epoch %d out of 30,000 done" % i)
        return
        pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0) ## DONE
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    xTrain = xTrain.reshape(xTrain.shape[0], IMAGE_SIZE)
    xTest = xTest.reshape(xTest.shape[0], IMAGE_SIZE) 
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        #print("Not yet implemented.")                   #TODO: Write code to build and train your custom neural net.
        ann = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, 512)
        ann.train(xTrain, yTrain)
        return ann
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        #print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        ann = tf.keras.models.Sequential([tf.keras.layers.Dense(512, activation=tf.nn.sigmoid), tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid)])
        ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        ann.fit(xTrain, yTrain, epochs=5, batch_size=128)
        return (ann)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custom neural net.
        #data_Flat = data.reshape(data.shape[0], (data.shape[1] * data.shape[2]))
        preds = model.predict(data)
        return preds
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = to_categorical(model.predict(data, batch_size=128), NUM_CLASSES)
        #print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
