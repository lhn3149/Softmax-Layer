import numpy as np
import random
import matplotlib.pyplot as plt
from io import StringIO
import math
# import scipy.special

class SoftmaxClassifier:
    
    def __init__ (self, epoch, learnRate, batchSize, regStrength, momentum):
        self.epoch = epoch
        self.learnRate = learnRate
        self.batchSize = batchSize
        self.regStrength =regStrength
        self.weight = None
        self.momentum = momentum 

    
    def train (self, x, y, xTest, yTest):
        """"
           x: data, y: label
        """

        data_dimension = x.shape[1]
        label = np.unique(y)
        numLabel = len(label)
        y_enc = self.oneHotEncoding(y,numLabel)
        yEncTest = self.oneHotEncoding(yTest,numLabel)
        
        self.weight = 0.001*np.random.rand(data_dimension, numLabel)
        self.velocity = np.zeros (self.weight.shape)
        Loss_record = []
        Acc_record =[]

        Loss_record_test = []
        Acc_record_test =[]


        for _ in range(self.epoch): #loop over epochs (all data set)
            # for training, update weight each time
            
            trainLoss = self.train_minibatch(x,y_enc)
            accuracy = self.meanPerClass(x,y)
            Acc_record.append(accuracy)
            Loss_record.append(trainLoss)


            ### this is for testing, weight is not updated

            testLoss, notuse = self.computeLoss(xTest, yEncTest)
            accuracy = self.meanPerClass(xTest,yTest)
            Acc_record_test.append(accuracy)
            Loss_record_test.append(testLoss)


        return Loss_record, Acc_record, Loss_record_test, Acc_record_test


    def oneHotEncoding(self, y,numLabel): 
        y_unique = np.unique(y)
        identity = np.eye(numLabel)
        yEnc = np.zeros([y.shape[0], numLabel])
        for i in range(y.shape[0]):
            idx = np.where(y_unique == y[i])
            yEnc[i] = identity[idx]
        return yEnc  

        
    def train_minibatch(self,x,y):
        """
            compute gradient of mini batches
            Update weight repeatedly each batch
        """
        [x,y] = shuffle_data(x,y)

        losses = []
        
        for i in range(0,len(x), self.batchSize):
            xBatch = x[i:i+self.batchSize]
            yBatch = y[i:i+self.batchSize]
            loss, grad = self.computeLoss(xBatch,yBatch)
            self.velocity = self.momentum*self.velocity + self.learnRate*grad
            self.weight = self.weight + self.velocity
            losses.append(loss)
        return np.mean(losses)
        
    def softmax(self,input):
        e_x = np.exp(input - np.max(input))
        e_x_sum = np.sum(e_x, axis = 1)
        output  = e_x/e_x_sum[:,None]
        return output

    def computeLoss(self,x,yEnc):
        """
            compute loss and gradent for mini_batch 
        """
        numOfSample = len(x)
        y_predict = np.dot(x, self.weight) # output  for specific class 
        prob = self.softmax(y_predict) # probability of each prediction using softmax 
        loss = np.sum(-np.log10(prob)*yEnc )/numOfSample + 1/2 * self.regStrength * np.sum(self.weight*self.weight)
        gradient = 1/numOfSample *np.dot(x.T,(yEnc-prob)) + (self.regStrength*self.weight)
        return loss, gradient

    def meanPerClass (self,xtopredict,groundtruth):
        """
            calculate mean per class: c
            compare actual y and predicted y
        """
        ypred = self.predict(xtopredict)
        ypred = ypred.reshape(-1,1)

        return np.mean(np.equal(groundtruth,ypred))
    
    def predict(self,x):
        # only need x and finalized weight -> output predicted y
        ypred = x.dot(self.weight)
        return np.argmax(ypred,axis = 1)



class TrainingModel:
    
    def __init__ (self, epoch, learnRate, batchSize, regStrength, momentum, model):
        self.epoch = epoch
        self.learnRate = learnRate
        self.batchSize = batchSize
        self.regStrength =regStrength
        self.weight = None
        self.momentum = momentum
        self.model = model

    
    def train (self, x, y, xTest, yTest):
        """"
           x: data, y: label
        """

        data_dimension = x.shape[1]
        label = np.unique(y)
        numLabel = len(label)
    
        self.weight = 0.001*np.random.rand(data_dimension, 1)
        self.velocity = np.zeros (self.weight.shape)

        Loss_record = []
        Acc_record =[]

        Loss_record_test = []
        Acc_record_test =[]


        for _ in range(self.epoch): #loop over epochs (all data set)
            # for training, update weight each time
            
            trainLoss = self.train_minibatch(x,y)
            accuracy = self.meanPerClass(x,y)
            Acc_record.append(accuracy)
            Loss_record.append(trainLoss)


            ### this is for testing, weight is not updated

            #testLoss, notuse = self.computeLoss(xTest, yTest, yEncTest)
            #accuracy = self.meanPerClass(xTest,yTest)
            #Acc_record_test.append(accuracy)
            #Loss_record_test.append(testLoss)


        return Loss_record, Acc_record, Loss_record_test, Acc_record_test

    def train_minibatch(self,x,y):
        """
            compute gradient of mini batches
            Update weight repeatedly each batch
        """

        losses = []
        
        for i in range(0,len(x), self.batchSize):
            xBatch = x[i:i+self.batchSize]
            yBatch = y[i:i+self.batchSize]
            loss, grad = self.computeLoss(xBatch,yBatch)
            self.velocity = self.momentum*self.velocity + self.learnRate*grad
            self.weight = self.weight - self.velocity
            losses.append(loss)
        return np.mean(losses)


    def computeLoss(self,x,y):
        """
            compute loss and gradent for mini_batch 
        """
        loss = []
        gradient = []
        if self.model == "L1decay":
            [loss, gradient] = self.L1decay(x, y)
        if self.model == "L2decay":
            [loss, gradient] = self.L2decay(x, y)
        if self.model == "Poisson":
            [loss, gradient] = self.Poisson(x,y)
        return loss, gradient

    def L1decay(self, x, y): # y is groundtruth
        numOfSample = len(x)
        y_predict = np.dot(x, self.weight) # output  for specific class 
        #linear regression is just y = wT.x
        y = y.reshape(-1,1)
        loss = (np.sum(np.square(y_predict-y)) + self.regStrength*np.sum(self.weight))/numOfSample 
        gradient = (2*np.dot(x.T, (y_predict-y)) + self.regStrength)/numOfSample
        return loss, gradient 

    def L2decay(self, x, y):
        numOfSample = len(x)
        y_predict = np.dot(x, self.weight) # output  for specific class 
        #linear regression is just y = wT.x

        loss = (np.sum(np.square(y_predict-y)) + self.regStrength*np.sum(self.weight*self.weight))/numOfSample
        xtrans = x.T
        y = y.reshape(-1,1)
        yterm = y_predict-y

        first_term = 2*np.dot(x.T, (y_predict-y))/numOfSample
        second_term = 2*self.regStrength*sum(self.weight)/numOfSample

        gradient = first_term + second_term
        return loss, gradient

    def Poisson(self, x, y): 
        """
            not yet finish because log of factorial
        """
        numOfSample = x.shape[0]
        y_predict = np.dot(x, self.weight) # output  for specific class 
        pois_pred = np.exp(y_predict)
        y = y.reshape(-1,1)
        ytimey = y*y_predict
        #logfac = np.log10(scipy.special.factorial(y))
        loss = np.sum((pois_pred + ytimey) )/numOfSample
        gradient = np.dot(x.T, pois_pred) - np.dot(x.T,y) 

        return loss, gradient

    def meanPerClass (self,xtopredict,groundtruth):
        """
            calculate mean per class: c
            compare actual y and predicted y
        """
        ypred = self.predict(xtopredict)
        ypred = ypred.reshape(-1,1)

        return np.mean(np.equal(groundtruth,ypred))
    
    def predict(self,x):
        # only need x and finalized weight -> output predicted y
        ypred = x.dot(self.weight)
        return np.argmax(ypred,axis = 1)


def shuffle_data(x,y):
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    x_new  = x[idx]
    y_new = y[idx]  
    return x_new, y_new
    

def load_data(train_file):
    data_file = 
    data = np.genfromtxt(train_file, delimiter=",")

    trainYears = data [0:463713,0]
    trainFeat = data [0:463713, 1:]
    trainFeatAvg = np.mean(trainFeat)
    trainFeatStd = np.std(trainFeat)
    trainFeatNorm = (trainFeat - trainFeatAvg)/trainFeatStd
    [trainYears, trainFeatNorm] = shuffle_data (trainYears, trainFeatNorm)

    testYears = data [463714:,0]
    testFeat = data [463714:,1:]
    testFeatAvg = np.mean(testFeat)
    testFeatStd = np.std(testFeat)
    testFeatNorm = (testFeat - testFeatAvg)/testFeatStd
    [testYears, testFeatNorm] = shuffle_data (testYears, testFeatNorm)
    
    return trainYears, trainFeatNorm, testYears, testFeatNorm 


def Add_Biases(trainFeat, testFeat, bias):

    addbiases = np.zeros(trainFeat.shape[0]).reshape(-1,1) + bias
    trainFeat = np.append(addbiases,trainFeat,axis=1)
    addbiases_test = np.zeros(testFeat.shape[0]).reshape(-1,1) + bias
    testFeat = np.append(addbiases_test,testFeat,axis=1)
    return trainFeat, testFeat


def musicMSE (pred,gt):
    yearPred = np.round(pred)
    numYears = yearPred.shape[0]
    val = np.sum(np.square(yearPred - gt))/numYears
    return val

def plotGraph(trainlossRecord, testLossRecord, trainAccuracy, testAccuracy):
    plt.plot(trainlossRecord, label="Train loss")
    plt.legend(loc='best')
    plt.title("Loss")
    plt.xlabel("Epochs")

    plt.figure()
    plt.plot(trainAccuracy, label="Train Accuracy")
    plt.legend(loc='best')
    plt.title("Mean per class Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Mean per class Accuracy")
    plt.show()

def Training (trainFeat_bias, trainYears, testFeat_bias, testYears, hyperparam, model):
    """
        model: model for training (L1, L2, Poisson)
    """
    epochs = hyperparam[0]
    learningRate = hyperparam[1]
    batchSize = hyperparam[2]
    regStrength = hyperparam[3]
    momentum = hyperparam[4]
    sftmx = TrainingModel (epoch=epochs, learnRate=learningRate, batchSize=batchSize,
                       regStrength=regStrength, momentum=momentum, model = model)
    trainLosses, trainAcc, testLosses, testAcc = sftmx.train(trainFeat_bias, trainYears, testFeat_bias, testYears)
    plotGraph(trainLosses,testLosses,trainAcc,testAcc)


    
if __name__ == '__main__':
    # normalized
    [trainYears, trainFeat, testYears, testFeat] = load_data("YearPredictionMSD.txt")
    trainYears = trainYears - np.min(trainYears)
    testYears = testYears - np.min (testYears)
    [trainFeat_bias, testFeat_bias] = Add_Biases(trainFeat, testFeat, 1)

    epochs = 20
    learningRate = 0.0001
    batchSize = 200
    regStrength = 0.000001
    momentum = 0.0005
    hyperparam = [epochs,learningRate,batchSize,regStrength,momentum]
    # model can be #Poisson #L1decay #L2decay
    trainFeat_trial = trainFeat_bias[0:50]
    trainYears_trial = trainYears[0:50]
    testFeat_trial = testFeat_bias[0:50]
    testYears_trial = testYears[0:50]


    #sftmx = SoftmaxClassifier(epoch=epochs, learnRate=learningRate, batchSize=batchSize,
    #                   regStrength=regStrength, momentum=momentum)
    #trainLosses, trainAcc, testLosses, testAcc = sftmx.train(trainFeat_trial, trainYears_trial, testFeat_trial, testYears_trial)
    #plotGraph(trainLosses, testLosses, trainAcc, testAcc)
    #Training(trainFeat_trial, trainYears_trial, testFeat_trial, testYears_trial, hyperparam, "L2decay")
    #Training(trainFeat_trial, trainYears_trial, testFeat_trial, testYears_trial, hyperparam, "L1decay")    
    Training(trainFeat_trial, trainYears_trial, testFeat_trial, testYears_trial, hyperparam, "Poisson")

    
    



    
