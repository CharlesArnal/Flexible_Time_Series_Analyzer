# Flexible_Time_Series_Analyzer
A somewhat flexible time series analyzer, created in the context of a small data project

The input are meant to be batches of time series in the shape of [batch_size, sequence_length, dim] arrays
The outputs should be of the shape [batch_size,sequence_length] (though only a few lines of codes would be needed to allow it to produce outputs of the shape [batch_size,sequence_length,dim_2] )
In details, the code currently expects the data to be in the shape of two csv files (one for training, one for testing) with three columns : "xprice","yprice" and "returns".
The goal is to predict returns[t] using {xprice[s],yprice[s]}{s<=t}, i.e. the returns at time t using all previously available information regarding xprice and yprice
This was a constraint of the project - it is easy to modify to allow more general input.

The code wraps a Keras convolutional recurrent neural network in a scikit learn wrapper.
This allows us to conveniently use the scikit learn crossvalidation random search tool for parameters selection.
Parameters include the width and number of layers used and the use of convolutional NN with much larger kernels to get easy access to longer term information

One can simply specify the parameters grid in the modelEstimate function, as well as the number of training epochs
One can also change the number of parameters configuration tested by the randomized search
(ideally, I would have let these be arguments of modelEstimate, but was again prevented from doing so by the instructions of the dataproject)

The two main functions are modelEstimate and modelForecast
modelEstimate takes as input the path to a csv file containing the training data, and outputs the selected trained model
modelForecast takes as input a trained model and a path to a csv file containing the test data, and outputs predictions

![illustration_model](https://user-images.githubusercontent.com/71833961/145279905-28059054-8e29-421d-bcbb-0c233e965231.png)
