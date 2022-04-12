
import itertools
import os
import pickle
import random, time
import matplotlib.pyplot as plt
import numpy as np
import wandb

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


        for i in range(self.epoch): #loop over epochs (all data set)
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

    def test (self, x, y):
        """"
           x: data, y: label
           return testLoss, testAccuracy
        """


    def oneHotEncoding(self, y, numLabel): 
        y = y.reshape(-1)
        return np.eye(numLabel)[y]
        
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

        return np.mean(np.equal(groundtruth,ypred))
    
    def predict(self,x):
        # only need x and finalized weight -> output predicted y
        ypred = x.dot(self.weight)
        return np.argmax(ypred,axis = 1)

    
def shuffle_data(x,y):
    idx = list(range(len(x)))
    random.shuffle(idx)
    x_new  = x[idx]
    y_new = y[idx]  
    return x_new, y_new

def loadData(listOfTrainFilePath, testFilePath):
    """
    Load training and testing data from different file and separate the train/test
    input sample and labels
    :param listOfTrainFilePath: List of training batch file path
    :param testFilePath:  List of testing file path
    :return: train/test input sample and labels
    """
    xTrain = []
    yTrain = []

    for i in range(len(listOfTrainFilePath)):
        data, labels = loadBatch(listOfTrainFilePath[i])
        xTrain = data if i == 0 else np.concatenate([xTrain, data], axis=0)
        yTrain = labels if i == 0 else np.concatenate([yTrain, labels], axis=0)

    xTest, yTest = loadBatch(testFilePath)
    yTest = np.array(yTest)
    yTrain = np.array(yTrain)
    return xTrain, yTrain, xTest, yTest

def loadBatch(filePath):
    """
    Read a single file and return its data and labels
    :param filePath: A file path
    :return: data and labels
    """
    dataDict = unpickle(filePath)
    return dataDict[b'data'], dataDict[b'labels']

def unpickle(file):
    """
    Read the data from file and return the dictionary
    :param file: A file path
    :return: dictionary containing information of data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def normalizeData(xTrain, xTest):
    """
    Normalize a data to -1 to 1 range. Subtract the mean from sample and divide it
    with 255.
    :param xTrain: Training sample
    :param xTest: Testing sample
    :return: normalize training and testing sample
    """
    meanImage = np.mean(xTrain, axis=0)
    xTrainD = xTrain - meanImage
    xTestD = xTest - meanImage
    xTrainD = np.divide(xTrainD, 255.)
    xTestD = np.divide(xTestD, 255.)
    return xTrainD, xTestD


    """
    Display number of different image from every class in single plot.
    :param data: Training data
    :param labels: Training classes
    :param meta: A list of classes name
    :param row: Number of row in a plot
    :param col: Number of column in a plot
    :param scale: how big the image should be
    :return: None
    """
    idx = count = 0
    imgWidth = imgHeight = 32
    keys = [lb for lb in range(len(meta))]
    labelCount = dict.fromkeys(keys, 0)

    figWidth = imgWidth / 80 * row * scale
    figHeight = imgHeight / 80 * col * scale
    fig, axes = plt.subplots(row, col, figsize=(figHeight, figWidth))

    while count < row * col:
        if labelCount[labels[idx]] >= row:
            idx += 1
            continue

        r = labelCount[labels[idx]]
        c = labels[idx]
        axes[r][c].imshow(data[idx])
        axes[r][c].set_title('{}: {}'.format(labels[idx],
                                             meta[labels[idx]].decode("utf-8")))
        axes[r][c].axis('off')
        plt.tight_layout()
        labelCount[labels[idx]] += 1
        count += 1
        idx += 1
    # plt.savefig(filepath)
    plt.show()

def getConfusionMatrix(actualLabel, predictedLabel, numOfClass):
    """
    Calculate a confusion matrix from actual label and predicted label
    :param actualLabel: Actual or target label
    :param predictedLabel: Predicted label
    :param numOfClass: Number of labels in dataset
    :return: confusion matrix of numOfclass x numOfclass
    """
    confMtrx =[]
    for _ in range(numOfClass):
        confMtrx.append([])
        for _ in range(numOfClass):
            confMtrx[-1].append(0)

    for sampleNum in range(actualLabel.shape[0]):
        confMtrx[int(actualLabel[sampleNum])][int(predictedLabel[sampleNum])] += 1
    confMtrx = np.array(confMtrx)
    return confMtrx

def plotGraph(trainlossRecord, testLossRecord, trainAccuracy, testAccuracy):
    plt.plot(trainlossRecord, label="Train loss")
    plt.plot(testLossRecord, label="Test loss")
    plt.legend(loc='best')
    plt.title("Softmax Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.figure()
    plt.plot(trainAccuracy, label="Train Accuracy")
    plt.plot(testAccuracy, label="Test Accuracy")
    plt.legend(loc='best')
    plt.title("Mean per class Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Mean per class Accuracy")
    plt.show()

def plotConfusionMatrix(model, xTest, actualLabel, classes):
    """
    
    """
    predY = model.predict(xTest)
    predY = predY.reshape((-1, 1))
    conMatrx = []

    for _ in range(10):
        conMatrx.append([])
        for _ in range(10):
            conMatrx[-1].append(0)

    for sampleNum in range(actualLabel.shape[0]):
        conMatrx[int(actualLabel[sampleNum])][int(predY[sampleNum])] += 1
    conMatrx = np.array('Confusion matrix')

    plt.imshow(conMatrx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = conMatrx.max() / 2.
    for i, j in itertools.product(range(conMatrx.shape[0]), range(conMatrx.shape[1])):
        plt.text(j, i, format(conMatrx[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conMatrx[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    """
    Main method
    """
    start = time.time()
    path = "C:\\Users\\nguye\\OneDrive\\Documents\\RIT-Computer Vision\\HW1\\data\\cifar-10-batches-py"
    # complete path of training file
    TRAIN_FILENAMES = [os.path.join(path, 'data_batch_' + str(i)) for i in range(1, 2)]
    TEST_FILENAME = os.path.join(path, 'test_batch') # complete path of testing file
    META_FILENAME = os.path.join(path, 'batches.meta') # complete path of meta file

    meta = unpickle(META_FILENAME)
    meta = meta[b'label_names']

    #wandb.init()
    #config = wandb.config
    #config.learning_rate = 0.002


    xTrain, yTrain, xTest, yTest = loadData(TRAIN_FILENAMES, TEST_FILENAME)
    xTrain, xTest = normalizeData(xTrain, xTest)
    yTrain = yTrain.reshape((-1, 1))
    yTest = yTest.reshape((-1, 1))


    epochs = 200
    learningRate = 0.07
    batchSize = 500
    regStrength = 0.0002
    momentum = 0.008

    sftmx = SoftmaxClassifier(epoch=epochs, learnRate=learningRate, batchSize=batchSize,
                       regStrength=regStrength, momentum=momentum)
    trainLosses, trainAcc, testLosses, testAcc = sftmx.train(xTrain, yTrain, xTest, yTest)

    classes = np.array([0,1,2,3,4,5,6,7,8,9])

    plotGraph(trainLosses, testLosses, trainAcc, testAcc)
    plotConfusionMatrix(sftmx, xTest, yTest, "0123456789",
                        normalize=True, title='Normalized confusion matrix')
    plt.show()

    #for i in range (10):
    #    wandb.log({"loss": trainLosses})

    end = time.time()