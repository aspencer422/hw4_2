#-------------------------------------------------------------------------
# AUTHOR: Anthony Spencer
# FILENAME: perpceptron.py
# SPECIFICATION: implement perpceptron
# FOR: CS 4210- Assignment #4
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

count=0
perp_accuracy=0
mlp_accuracy=0
mlp_lr = ''
mlp_shuff = ''
perp_lr = ''
perp_shuff = ''

for w in n: #iterates over n

    for b in r: #iterates over r

        for a in range(2): #iterates over the algorithms

            #Create a Neural Network classifier
            if a==0:
               clf = Perceptron(eta0=w, shuffle=b, max_iter=1000) #eta0 = learning rate, shuffle = shuffle the training data
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,), shuffle =b, max_iter=1000) #learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer, shuffle = shuffle the training data

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            acc=0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
               class_predicted = clf.predict([x_testSample])
               if class_predicted == y_testSample:
                  acc = acc + 1
            temp_accuracy = acc/len(X_test)


            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #handle iter 1
            if count == 0:
               perp_accuracy = temp_accuracy
               perp_lr = str(w)
               perp_shuff = str(b)
            #handle inter 2
            elif count == 1:
               mlp_accuracy = temp_accuracy
               mlp_lr = str(w)
               mlp_shuff = str(b)
            #handle inter > 2
            else:
               if perp_accuracy < temp_accuracy and a == 0:
                  perp_accuracy = temp_accuracy
                  perp_lr = str(w)
                  perp_shuff = str(b)
               elif mlp_accuracy < temp_accuracy and a==1:
                  mlp_accuracy = temp_accuracy
                  mlp_lr = str(w)
                  mlp_shuff = str(b)
            
            
            print("Highest Perceptron accuracy so far: %.2f, Parameters: learning rate=%s, shuffle=%s" % (perp_accuracy,perp_lr,perp_shuff))
            print("Highest MLP accuracy so far: %.2f, Parameters: learning rate=%s, shuffle=%s" % (mlp_accuracy,mlp_lr,mlp_shuff))
            print()
            count= count+1



            


            

            











