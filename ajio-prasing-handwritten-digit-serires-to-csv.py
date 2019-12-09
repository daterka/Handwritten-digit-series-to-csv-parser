import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.losses import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import timeit
import time
import csv

populationSize = 10
metric = 'loss'
modelArchFilePatch = "saved-models/cnn-model-arch.json"
modelWeightFilePatch = "saved-models/cnn-model-weights.h5"

class TimeCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.timeOfTraining = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epochStartTime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epochDuration = (time.time() - self.epochStartTime)
        self.timeOfTraining += self.epochDuration
        self.times.append(self.epochDuration)

class ConvolutionalNeuralNetwork:
    batch_size = 128
    epochs = 8
    num_classes = 10

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data sets downloaded")

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    print("Data formatted")


    def __init__(self, id):
        self.id = id
    
    def defineModelArchitecture(self):
        self.model = Sequential() 
        self.model.add(Conv2D(1, kernel_size=(3,3), input_shape=self.input_shape)) 
        self.model.add(MaxPooling2D(pool_size=(2,2))) 
        self.model.add(Flatten()) 
        self.model.add(Dense(72,activation = 'relu'))
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(10,activation="softmax"))

    def modelSummary(self):
        self.model.summary()
    
    def buildModel(self):
        self.model.compile(
            optimizer="adam", 
            loss=mse, 
            metrics=['accuracy', 'mae'])

    def trainModel(self):
        # self.fp = 'weights.{epoch:02d}-{val_accu:.2f}.hdf5'
        self.fp = 'weights_best.hdf5'

        checkPointCallback = keras.callbacks.callbacks.ModelCheckpoint(
            filepath = self.fp, 
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=True, 
            mode='auto', 
            period=1)

        self.timeCallback = TimeCallback()

        callbacksList = [checkPointCallback, self.timeCallback]

        self.history = self.model.fit(
            self.x_train, 
            self.y_train_cat, 
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            verbose=0, 
            validation_split=0.1,
            callbacks=callbacksList)

        self.model.load_weights(self.fp)

        self.timeOfTraining = self.timeCallback.timeOfTraining

    def evaluateModel(self):
        self.metrics = self.model.evaluate(
            self.x_test, 
            self.y_test_cat, 
            verbose=1)

    def results(self):
        print('id : ', self.id)
        print('Test loss:', self.metrics[0])
        print('Test accuracy:', self.metrics[1], '%')
        print('Test mae:', self.metrics[2])
        print('Training duration : ', self.timeOfTraining, 's')


    
    def saveModel(self):
        self.model_json = self.model.to_json()
        with open(modelArchFilePatch, "w") as json_file:
            json_file.write(self.model_json)

        self.model.save_weights(modelWeightFilePatch)
        print("CNN model saved")

    def loadModel(self):
        self.model = model_from_json(modelArchFilePatch)
        self.model.load_weights(modelWeightFilePatch)

    def modelAccuracyPlot(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

    def modelLossPlot(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def modelMSEPlot(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['mae'])
        plt.title('model mae')
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

    def printDigit(self):
        pass

    def predict(self, first = -1, last = -1):
        if(last != -1):
            self.prediction = self.model.predict_classes(self.x_test[first:last])
            print(self.prediction)
            print(self.y_test[first:last])
        elif(first != 0):
            self.prediction = self.model.predict_classes(self.x_test[:first])
            print(self.prediction)
            print(self.y_test[:first])
        else:
            self.prediction = self.model.predict_classes(self.x_test)
            print(self.prediction)
            print(self.y_test)

    def predictionsToCSV(self):
        rows = zip(self.y_test, self.prediction)
        with open(
            'prediction.csv', 
            mode='w', 
            newline='') as prediction_csv:
            csv_writer = csv.writer(prediction_csv)
            csv_writer.writerow(['Labels', 'Predictions'])
            csv_writer.writerows(rows)





def pickBest(population, metric):
    if(metric == 'loss'):
        population.sort(reverse = False, key=lambda e: e.metrics[0])
    elif(metric == 'accuracy'):
        population.sort(reverse = True, key=lambda e: e.metrics[1])
    elif(metric == 'mae'):
        population.sort(reverse = False, key=lambda e: e.metrics[2])

    return population[0]


def main():
    population = []
    for i in range(populationSize):
        cnn = ConvolutionalNeuralNetwork(i)
        cnn.defineModelArchitecture()
        cnn.buildModel()
        cnn.trainModel()
        cnn.evaluateModel()
        cnn.results()
        population.append(cnn)

    best = pickBest(population, metric)
    best.saveModel()

    print('\nBest model:')
    best.results()
    best.predict()
    best.predictionsToCSV()



    

    
if __name__ == "__main__":
    main()
        



