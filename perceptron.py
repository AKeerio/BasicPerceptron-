import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# make a guess based on current inputs and weights
def predict(inputs, weights):
    threshold=0.0
    activation=0.0
    for input, weight in zip(inputs,weights):
        activation+=input*weight
    if(activation>=0):
        return 1.0
    else:
        return 0.0

def train(inputs, targets, weights, lrate, iterations):
    # For each iteration
    for k in range(iterations):
        for i in range(len(inputs)):
            prediction=predict(inputs[i],weights)
            error = targets[i] - prediction
            for j in range(len(weights)):
                weights[j] = weights[j] + (lrate * error * inputs[i][j])
    return weights

def plot(newweights, X, Y,feature1, feature2):
    for i in range(len(X)):
        prediction=newweights[0]+X[i][1]*newweights[1]+X[i][2]*newweights[2]+X[i][3]*newweights[3]+X[i][4]*newweights[3]

        # Check if they have same sign i.e. plus or minus
        if(abs(prediction + Y[i]) == abs(prediction) + abs(Y[i])):
            print("Correct", prediction, Y[i])
            if (Y[i] < 0):
                plt.scatter(feature1[i], feature2[i], color="brown", marker="^", label='setosa')
            else:
                plt.scatter(feature1[i], feature2[i], color="blue", marker="^", label='sersocolor')
        else:
            print("In-correct", prediction, Y[i])
            plt.scatter(feature1[i], feature2[i], color="red", marker="x", label='misclassified')


    versicolor = mpatches.Patch(color='blue', label='Versicolor')
    setosa = mpatches.Patch(color='brown', label='Setosa')
    misclassified=mpatches.Patch(color='red', label='Misclassified')

    plt.legend(handles=[versicolor, setosa, misclassified])
    plt.xlabel('Feature 2')
    plt.ylabel('Feature 1')
    plt.show()

def plotBefore(X, Y, feature1, feature2):
    # Plot graph

    for i in range(len(X)):
        if (Y[i] < 0):
            plt.scatter(feature1[i], feature2[i], color="blue", marker="x")
        else:
            plt.scatter(feature1[i], feature2[i], color="red", marker="x")

    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.show()

# Draw line
def classify():
    return 0

def main():
    # Download data
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # Seperate names from sepal and petals widths and legnths
    data.tail()

    # Put all flower names in a seperate array
    Y = data.iloc[0:len(data), 4].values

    # Replace names with numbers (Iris-satosa=-1)
    Y = np.where(Y == 'Iris-setosa', -1, 1)

    # Take all input values and put them in an array
    X = data.iloc[0:len(data), [0, 1, 2, 3, 4]].values

    # Add bias input 1 in the inputs array
    for i in range(len(X)):
        X[i] = [1, X[i][0], X[i][1], X[i][2], X[i][3]]

    # Randomly define weights
    weights=[]
    for i in range(4):
        weights.append(2 * np.random.random() -1)

    # Define learning rate
    lrate=0.001

    # Define number of iterations for training
    iterations=10

    newweights=train(X,Y,weights,lrate,iterations)

    # X[i][1]=sepal lengh and  X[i][2]=sepal width
    # X[i][3]=petal lengh and X[i][2]=petal width
    feature1 = []
    feature2 = []
    for i in range(len(X)):
        feature1.append(X[i][1])
        feature2.append(X[i][4])
    # Plot graph before any machine learning
#    plotBefore(X, Y,feature1,feature2)
    plot(newweights, X, Y,feature1,feature2)

main()

