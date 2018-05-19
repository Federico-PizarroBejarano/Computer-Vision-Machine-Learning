# Computer-Vision-Machine-Learning
Abstract 

This investigation determines the extent that characters can be identified in images using the logistic regression 
and single-layer neural network algorithms. Optical character recognition (OCR) is a computer vision, supervised 
learning problem. The dependent variables were the optimal value of the regularization parameter lambda, the 
accuracy on the training, cross validation, and test sets, and the time needed to train each classifier. 
A dataset of 74,000 images composed of fonts, handwritten characters, and real images of letters and numbers was used. 
For the purposes of this investigation only a subset of the font dataset was used. Each image was resized to be 
20x20 pixels and then converted to a 1x400 vector of pixel values. 

The logistic regression algorithm attempts to fit parameters to the 400 pixel values to form a hypothesis function. 
To optimize the parameters, the algorithm defines a cost function and then performs gradient descent on the parameters. 
The tunable parameters were additional features added in an attempt to create more complex, representative functions. 

A single-layer neural network passes the input data to a hidden layer where the data is partially processed. The 
partially processed data is then passed to the output layer where the final predictions are made. The tunable 
parameter was the number of hidden units in the hidden layer. 

The logistic regression algorithm achieved an accuracy of 85.14% with no added features and a lambda 
value of 1. The neural network achieved a significantly higher accuracy of 90.19% using 200 hidden units and no 
regularization.

Logistic regression had a time complexity of O(n) while the neural network had a significantly better time complexity
of O(√h). This paper investigates the properties of both algorithms as well as establishes the inability of both 
algorithms to identify characters to sufficiently high accuracies.


For all the data, both raw and processed, as well as all other documents here is the link. The files are fairly large and did not fit on GitHub

https://drive.google.com/drive/folders/1JOXEk2pxX9CoB0AVmuOTvcEy2aBJ4Dcz?usp=sharing
