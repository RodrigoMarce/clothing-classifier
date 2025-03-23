import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    n = batchSize
    numSamples = trainingImages.shape[0]
    numClasses = trainingLabels.shape[1]
    Epochs = 10
    #Already have bias added in _main_ so no need to add
    Wtilde = np.random.randn(785, numClasses) * 1e-5

    indices = np.random.permutation(numSamples)
    trainingImages, trainingLabels = trainingImages[indices], trainingLabels[indices]  # Shuffle the data
    
    for epoch in range(Epochs):
        batchLoss = []
        for i in range(0, numSamples, n):
            #Grab MiniBatch
            X = trainingImages[i:i+n]
            Y = trainingLabels[i:i+n]
            Score = X @ Wtilde
            #Calculate Gradient
            yPred =  np.exp(Score) / np.sum(np.exp(Score), axis=1, keepdims=True) 
            diff = yPred - Y
            gradient = (X.T @ diff) / n
            #Apply L2 Regularization:
            gradient[:-1, :] += (alpha / numSamples) * Wtilde[:-1, :] # Don't regularize the bias term
            #Update Weights
            Wtilde -= epsilon * gradient
            entropyLoss = -np.sum(Y * np.log(yPred)) / n
            batchLoss.append(entropyLoss)
    
    #Last 20 losses
    if len(batchLoss) >= 20:
        last20 = batchLoss[-20:]
        
    #Calculate Percent Accurate with Testing Data
    testScore = testingImages @ Wtilde
    PercentAccurate = np.mean(np.argmax(testScore, axis=1) == np.argmax(testingLabels, axis=1))
    print("Last twenty losses:")
    print([float(round(num, 5)) for num in last20])
    print("Percent Accurate: ", PercentAccurate)
    return Wtilde
                    
def oneHotEncode(labels, numClasses):
    oneHotLabels = np.zeros((len(labels), numClasses))
    oneHotLabels[np.arange(labels.shape[0]), labels] = 1
    return oneHotLabels

def visualizeVectors(Wtilde, numClasses):
    for i in range(numClasses):
        plt.imshow(Wtilde[:-1, i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"Class {i}")
        plt.savefig(f"vector_{i}.png")
        plt.close()
    
if __name__ == "__main__":  
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")
    numClasses = trainingLabels.max() + 1
    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = np.hstack((trainingImages, np.ones((trainingImages.shape[0], 1))))
    testingImages = np.hstack((testingImages, np.ones((testingImages.shape[0], 1))))
    # Change from 0-9 labels to "one-hot" binary vector labels.
    trainingLabels = oneHotEncode(trainingLabels, numClasses)
    testingLabels = oneHotEncode(testingLabels, numClasses)
    # Train the model this also gets last 20 losses and percent correct on testing data
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)
    # Visualize the vectors
    visualizeVectors(Wtilde, numClasses)
    
