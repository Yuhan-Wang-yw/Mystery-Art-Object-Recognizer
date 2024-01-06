# Mystery-Art-Object-Recognizer
A CNN(convolutional neural network) model to classify images of artworks of ancient cities and to recognize the style of mystery artworks. Final project for MTH 353 Seminar in Deep Learning &amp; ARH 212 Ancient Cities and Sanctuaries.

### Brief Abstract:

Convolutional Neural Network (CNN) is a subtype of Neural Networks that is mainly used for applications in image recognition. For our final project, we plan to train a CNN that recognizes artworks from ancient Near Eastern, Egyptian, Minoan, Mycenaean, Greek, Etruscan, Roman that were introduced in the ARH 212 course. The training data of this CNN would be pictures of artworks from those cultures, and we will split the pictures into small fractions so that we can both have more data for training and focus on details of the pictures. The label for the training data will be the culture origin of the artworks in the photo. With this trained neural network, we will choose a photo of a mystery object from one of the categories and ask the neural network to predict what culture or time period the mystery object is from. This will be a classification problem.

### Mathematics in the project:

Neural network embodies many important math concepts and functions as its foundation. Linear algebra, multivariate calculus and basic notions of statistics will all be used in the training of a CNN. Linear algebra is used for example, when each neuron calculates its input with its weights, it performs a dot product of two matrices or vectors. Multivariate calculus is used in the evaluation and training of the CNN, for example, when performing backpropagation, we can apply gradient descent to the function to find a local minimum that minimizes the loss of a CNN. Basic statistics, like probability, is the very basis of the CNN because the CNN outputs a probability of how likely one artwork is from one culture.

