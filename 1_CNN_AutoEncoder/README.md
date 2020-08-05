# Auto Encoder Model ( Fully Connected & CNN Implementation ) 
Master Class Project with followind requirements 

[Reach via Colab](https://colab.research.google.com/drive/1t-EHlS9G04PF7jcnPd_l47aoRJ8pcaBR#scrollTo=LRpZsKC6aQY6)

In this project, auto-encoder is implemented using MNIST dataset. 

## 1 - Auto Encoder Model With Fully Connected Network

***Layers in model are like below :***

*Input ( X ) with units 784 ( input size )*

*Encoder Layer with units 128*

*Embedding Layer with units 64*

*Decoder Layer with units 128*

*Output ( X_HAT ) with units 784 ( input size )*

**Units In Hidden :** I choose 128 and 64 for units in layers for easy calculation and
prevent information loss comparing to the number of units of 32 or 16.

**Activation function :** I choose activation function Sigmoid in last step because we need
a data between zero and one to reach a greyscale image at the end.

I choose loss function as mean square error as in stated project requirements.

## 2 - Auto Encoder Model With Convolutional Neural Network

I choose activation function Relu in layers except the last one.

I choose activation function Sigmoid in last step because of the same reason in FC.

I choose (1,2,2,1) as kernel size, it would be good our image data points sized as 28*28, not so large and so small.

***Layers in model are like below :***

*Input ( X )*

*First Convolutional Layer ( Encoder )*

*First Pooling Layer*

*Second Convolutional Layer ( Embedding)*

*Second Pooling Layer*

*First Fully Connected Layer ( Decoder 1)*

*Second Fully Connected Layer ( Decoder 2)*

*Output Layer*

## 3 - Training Results 
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/training_results.png) 


## Original and Decoded Images 

## For Test Suit 4
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/test_suit_4_encoded_decoded.png) 


## For Test Suit 10
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/test_suit_10_encoded_decoded.png) 


## T-Sne Results For Original And Embedded Data 

## For Test Suit 4 
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/test_suit_4_fcn_cnn_comparison.png) 


## For Test Suit 10 
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/test_suit_10_fcn_cnn_comparison.png) 


## Random Picks 

## For Test Suit 4
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/test_suit_4_random_picks.png)


## For Test Suit 10 
![](https://github.com/mateskarayol/deep-learning-1-auto-encoder/blob/master/results/test_suit_10_random_picks.png)


## Comments About The Models
While training different hyperparameters and optimizers, I realized that convolutional model produces better results than the fully connected one.

I try to point out for one of the good results in training set ( test suit 10 ) and one of the less good ones ( test suit 4 ).

Plotting original and test images as image, we could see that the convolutional network results better.

T-sne results shows us our embedded data on fully connected network and convolutional network are classified better than the original test data and it implies that
our model works fine and eliminate the data points storing the important data.

***References***

*1 - https://gist.github.com/tomokishii/7ddde510edb1c4273438ba0663b26fc6#filemnist_ae1-py*

*2 - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py*

*3 - https://towardsdatascience.com/deep-autoencoders-using-tensorflow-c68f075fd1a3*

*4 - https://github.com/Seratna/TensorFlow-Convolutional-AutoEncoder*

*5 - https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_with_legend.html*

*6 - https://github.com/GauravSaini728/t-SNE-on-MNIST-and-Visualization/blob/master/t-SNE%20MNIST.ipynb*

*7 - https://www.easy-tensorflow.com/tf-tutorials/neural-networks/two-layer-neuralnetwork?view=article&id=124:two-layer-neural-network*

*8 - https://github.com/easy-tensorflow/easy-tensorflow/blob/master/1_TensorFlow_Basics/Tutorials/2_Tensor_Types.ipynb*

