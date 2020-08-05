# LSTM - Long short-term memory network
LSTM model for classification purpose

[Reach Via Colab](https://colab.research.google.com/drive/1H0eLy7MpxWcOTk5IhpZ3G99-l9hxj_zW#scrollTo=6i8-NBIWKzcB)

### 1 – Building Model

I added Input Layer, LSTM Layer and Output Layer to model. 

Firstly, I reshape the training data from (55000, 784) to (55000, 28, 28). 

***In Input layer, I use;***

Input class in Keras to create it

28 as time step

28 as number of inputs for each time step

***In LSTM layer, I use;***

LSTM class in Keras to create it

128 as hidden unit

(28 X 28) as input shape as in input layer

*Because I try to build a classification model, I let the model to use default activation functions which are also used for classification.*

Tanh as activation function as in default ( tanh predicts a probability in range of -1 and +1 ) 

Sigmoid as recurrent activation function as in default ( sigmoid predicts a propbability in range 0 and 1) 

***In Output layer I use;***

Dense class in Keras to create it

10 as number of classes because we have 10 different labels

Softmax activation function (  So I try to build a classification model for mnist data, I use a function that gives a probability distributions and softmax would be helpful for multiclass classification)

I use Model class in Keras to create model object with input layer and output layer. 

#### Model Summary

|Layer (type)                |Output Shape            |Param       |
|----------------------------|------------------------|------------|
|Input_Layer (InputLayer)    | (None, 28, 28)         |   0        | 
|LSTM_Layer (LSTM)           | (None, 128)            |   80384    | 
|Output_Layer (Dense)        | (None, 10)             |   1290     | 

### 2 – Compiling and Running Model

I use categorical cross entropy as loss function.

Keras cross entropy have 2 alternative functions as binary and categorical, so I have 10 different labels in data categorical would be more appropriate. 

I use Adam optimizer in compiling. 

In previous project, I have used different optimizers and hyper parameter combinations, so I have a good understanding of capability of optimizers and Adam would be produce better results.

I take 128 as batch size and 5 as epoch.

It means data is trained with set of 128 data points for each training and whole data set will be trained 5 times.

I take True as shuffle.

It means that data will be shuffled before taking batches, in this way batch data would represent the whole data set more accurately, because of the randomness. 

### 3 – Testing Model

Before testing, I reshape the test data from (10000, 784) to (10000, 28, 28) and then I use model.evaluate to get the results. 


### 4 – Applying T-SNE, KMeans and Plotting 

I get the intermediate model ending with LSTM layer with layer name and use test data to get the output from LSTM layer. 

I convert the output comes from LSTM as 2D with T-SNE.

I calculate the centroids ( which is the center of the cluster ) for each class data based on the results converted in T-SNE with the supported code blocks in project description. 

I categorize the data and create a color map to use in plotting.

I use pyplot to plot data in 2D with the data set converted in T-SNE and color map. 

### 5 – Results
#### Training Results

    Epoch 1/5
    55000/55000 [==============================] - 12s 211us/step - loss: 0.5907 - accuracy: 0.8040
    Epoch 2/5
    55000/55000 [==============================] - 10s 179us/step - loss: 0.1675 - accuracy: 0.9486
    Epoch 3/5
    55000/55000 [==============================] - 10s 173us/step - loss: 0.1122 - accuracy: 0.9664
    Epoch 4/5
    55000/55000 [==============================] - 10s 175us/step - loss: 0.0856 - accuracy: 0.9733
    Epoch 5/5
    55000/55000 [==============================] - 10s 175us/step - loss: 0.0683 - accuracy: 0.9788

    Test Results
    Test loss        :  0.06877894503846764
    Test accuracy    :  0.9793999791145325

#### Plotting

![](https://github.com/mateskarayol/DeepLearning_2_LSTM/blob/master/results/result1.png)

### 6 – References

https://keras.io/api/losses/

https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction

https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy

https://keras.io/examples/mnist_hierarchical_rnn/

https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
