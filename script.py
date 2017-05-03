import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle
import time
import csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from multiprocessing import Process

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def SVM_DEF(C, model):
	gamma=None
	C=None
	start=time.time()
	clf = SVC(kernel=model)
	clf.fit(X_train, Y_train)
	accTrain=clf.score(X_train,Y_train)
	accValidation=clf.score(X_validation,Y_validation)
	accTest=clf.score(X_test,Y_test)
	end=time.time()
	out="\n"+model+","+str(C)+","+str(gamma)+","+str(accTrain)+","+str(accValidation)+","+str(accTest)+","+str((end-start)/60)+",minutes"
	svmOutput.write(out)
	svmOutput.flush()

def SVM_RBF(C, model):
	gamma=None
	start=time.time()
	clf = SVC(C=C, kernel=model)
	clf.fit(X_train, Y_train)
	accTrain=clf.score(X_train,Y_train)
	accValidation=clf.score(X_validation,Y_validation)
	accTest=clf.score(X_test,Y_test)
	end=time.time()
	out="\n"+model+","+str(C)+","+str(gamma)+","+str(accTrain)+","+str(accValidation)+","+str(accTest)+","+str((end-start)/60)+",minutes"
	svmOutput.write(out)
	svmOutput.flush()

def SVM_RBF_GAMMA(model, gamma):
	C=None
	start=time.time()
	clf = SVC(kernel=model, gamma=gamma)
	clf.fit(X_train, Y_train)
	accTrain=clf.score(X_train,Y_train)
	accValidation=clf.score(X_validation,Y_validation)
	accTest=clf.score(X_test,Y_test)
	end=time.time()
	out="\n"+model+","+str(C)+","+str(gamma)+","+str(accTrain)+","+str(accValidation)+","+str(accTest)+","+str((end-start)/60)+",minutes"
	svmOutput.write(out)
	svmOutput.flush()
	f = open('svmOut.csv', 'r')
	accrbf = np.zeros(shape=(11, 4))
	acclin = np.zeros(shape=(1, 3))
	accgam = np.zeros(shape=(1, 3))
	accdef = np.zeros(shape=(1, 3))
	reader = csv.DictReader(f, delimiter=',')

	i=0
	for line in reader:
		if(line['C']!='None'):
			accrbf[i,0] = float(line['C'])
			accrbf[i,1] = float(line['TrainingAccuracy'])
			accrbf[i,2] = float(line['ValidationAccuracy'])
			accrbf[i,3] = float(line['TestingAccuracy'])
			line['Time'] = float(line['Time'])
			i+=1
		elif(line['Kernel']=='linear'):
			acclin[0,0]=float(line['TrainingAccuracy'])
			acclin[0,1]=float(line['ValidationAccuracy'])
			acclin[0,2]=float(line['TestingAccuracy'])
		elif(line['gamma']=='1.0'):
			accgam[0,0]=float(line['TrainingAccuracy'])
			accgam[0,1]=float(line['ValidationAccuracy'])
			accgam[0,2]=float(line['TestingAccuracy'])
		elif(line['Kernel']=='rbf' and line['C']=='None'):
			accdef[0,0]=float(line['TrainingAccuracy'])
			accdef[0,1]=float(line['ValidationAccuracy'])
			accdef[0,2]=float(line['TestingAccuracy'])

	accrbf.sort(0)
	print('\nSupport Vector Machines')
	print('=======================')
	print('\nLinear Kernel')
	print('\tTraining set Accuracy:' + str(acclin[0,0]*100) + '%')
	print('\tValidation set Accuracy:' + str(acclin[0,1]*100) + '%')
	print('\tTesting set Accuracy:' + str(acclin[0,2]*100) + '%')
	print('\n')
	print('\nRadial Basis Function Kernel(Default)')
	print('\tTraining set Accuracy:' + str(accdef[0,0]*100) + '%')
	print('\tValidation set Accuracy:' + str(accdef[0,1]*100) + '%')
	print('\tTesting set Accuracy:' + str(accdef[0,2]*100) + '%')
	print('\n')
	print('\nRadial Basis Function Kernel(Gamma=1.0)')
	print('\tTraining set Accuracy:' + str(accgam[0,0]*100) + '%')
	print('\tValidation set Accuracy:' + str(accgam[0,1]*100) + '%')
	print('\tTesting set Accuracy:' + str(accgam[0,2]*100) + '%')

	fig = plt.figure(figsize=[12,6])
	plt.subplot(1, 1, 1)
	plt.plot(accrbf[:,0],accrbf[:,1], label='RBF Training')
	plt.plot(accrbf[:,0],accrbf[:,2], label='RBF Validation')
	plt.plot(accrbf[:,0],accrbf[:,3], label='RBF Testing')
	plt.title('Accuracies for SVM')
	plt.legend()
	plt.show()
	plt.savefig('SVM_PLOT.png')

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    
    ##################
    train_data, labeli = args
    w = initialWeights
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    train_data = np.append(np.ones((np.size(train_data,0),1)),train_data,1)
    theta=1/(1+np.exp(-np.dot(w.T,train_data.T)))
    oneMinusTheta=1/(1+np.exp(np.dot(w.T,train_data.T)))
    error = (-1/n_data)*(np.dot(labeli.T,np.log(theta.T))+np.dot(1-labeli.T,np.log(oneMinusTheta.T)))
    error_grad = (1/n_data)*np.dot((theta-labeli.T),train_data)
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad.flatten()


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    label = np.zeros((data.shape[0], 1))
    data = np.append(np.ones((np.size(data,0),1)),data,1)
    prediction=sigmoid(np.dot(data,W))
    label=prediction.argmax(1).astype(int)
    label=np.array(label).reshape(label.size, 1)
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    
    ##################
    # YOUR CODE HERE #
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = labeli.shape[1]
    
    w = params.reshape(n_feature + 1, n_class)
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    train_data = np.append(np.ones((np.size(train_data,0),1)),train_data,1)
    theta = np.multiply(np.reciprocal(np.exp(np.dot(train_data,w)).sum(1)).reshape(n_data,1),np.exp(np.dot(train_data,w)))
    error = (-1)*np.multiply(np.log(theta),labeli).sum()
    error_grad = np.dot((theta - labeli).T,train_data).T
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad.flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    data = np.append(np.ones((np.size(data,0),1)),data,1)
    prediction = np.multiply(np.reciprocal(np.exp(np.dot(data,W)).sum(1)).reshape(data.shape[0],1),np.exp(np.dot(data,W)))
    label=prediction.argmax(1).astype(int)
    label=np.array(label).reshape(label.size, 1)
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
svmOutput = open("svmOutM.csv", "a+")

out="Kernel,C,gamma,TrainingAccuracy,ValidationAccuracy,TestingAccuracy,Time,TimeUnits"
svmOutput.write(out)
svmOutput.flush()

X_train,Y_train,X_validation,Y_validation,X_test,Y_test=preprocess()

# SVM starts here

#Linear Default
model='linear'
Process(target=SVM_DEF, args=(1.0, model)).start()
#Linear Ends

#RBF starts
model='rbf'
#RBF Default
Process(target=SVM_DEF, args=(1.0, model)).start()
#RBF Default Ends

#RBF gamma=1.0
tg=Process(target=SVM_RBF_GAMMA, args=(model, 1.0)).start()
#RBF gamma Ends

#RBF-C=1.0~100.0
Process(target=SVM_RBF, args=(1.0, model)).start()
for i in range(10):
	Process(target=SVM_RBF, args=(float((i+1)*10), model)).start()
#RBF-C=1.0~100.0 Ends


##################

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
	
f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

