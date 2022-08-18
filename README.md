# Implementing-Back-Propogation-from-Scratch
implementing the backpropagation algorithm and train your first multi-layer perceptron to distinguish between 4 classes. We implement everything  without using existing libraries like Tensorflow, Keras, or PyTorch.

We are provided with two files “train data.csv“ and “train labels.csv“ The dataset contains 24754 samples, each with 784 features divided into 4 classes (0,1,2,3). You should divide this into training, and validation sets (a validation set is used to make sure your network did not overfit). We will then provide your model which will be tested with an unseen test set. Use one input layer, one hidden layer, and one output layer in your implementation. The labels are one-hot encoded. For example, class 0 has a label of [1, 0, 0, 0] and class 2 has a label of [0,0,1,0]. Make sure you use the appropriate activation function in the output layer. One single function that allows us to use the network to predict the test set. This function outputs the labels one-hot encoded in a numpy array.
