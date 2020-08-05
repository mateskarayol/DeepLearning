# Variational Auto Encoders
Deep Learning - AutoEncoder modeled with LSTM and VAE

[Reach via Colab](https://colab.research.google.com/drive/153EL0gg-7Cm7z6i1dh3P39RuUUXvdfwG#scrollTo=3lxk-PhGg6Ib)

### 1 - Encoder With LSTM

***Layers in model are like below :***

I prepare LSTM Encoder Model containing : 

	1 Input Layer with shape 28 * 28

    1 LSTM Layer 
	
      with 256 hidden unit
		
      with 28*28 shape
		
      with activation function tanh
		
      with recurrent activation function sigmoid
	
    2 layers for mean and variance which are taking LSTM Layer
	
    1 Lambda generating Sample Distribution with latent dimension 16 
	
Eventually Encoder Model takes Input Layer and generates mean, variance and Sampling layer as output.


#### Encoder Model Summary

![](https://github.com/mateskarayol/DeepLearning_3_VAE/blob/master/result/encoder_model_summary.png)

#### Sampling 
In Sampling, re-parameterization trick is used make gradient descent like random sampling. We use sampling in Lambda Layer, and so we prevent producing random samples. It helps to calculate partial derivatives based on mean and standard deviation. 

### 2 - Fully Connected VAE Decoder

***Layers in model are like below :*** 

    I prepare Fully Connected Decoder model containing : 
	    1 Input Layer taking the latent layer from encoder as input with shape (16,)
	    1 Hidden Layer 
	      	with hidden units 256 as in LSTM layer in Encoder model
	      	with activation function relu
    	1 Output Layer
	      	with shape 28*28 
	      	with activation function sigmoid

Eventually Decoder Model takes latent input layer as input and generates output.

#### Decoder Model Summary 

![](https://github.com/mateskarayol/DeepLearning_3_VAE/blob/master/result/decoder_model_summary.png)

#### 3 - Variational Auto Encoder

I use the sampling layer from outputs of the Encoder Model as input for Decoder model. 

With the Input Layer of the Encoder Model and Output Layer of the Decoder Model, I prepared VAE model with Adam Optimizer and Loss Function combining Binary Cross Entropy and KL Divergence. 

#### VAE Model Summary

![](https://github.com/mateskarayol/DeepLearning_3_VAE/blob/master/result/vae_model_summary.png)

#### Hyperparameters

*Batch Size = 256 ( Size of each batch )*

*Epoch  = 30  ( Size of epoch )*

*Encoder (LSTM)Hidden Units  = 256  ( Hidden units in LSTM layer )*

*Latent Dimension = 16  ( Dimension in the latent space which is the input 	of decoder layer )*

*Decoder Hidden Units = 256  ( Dimension in hidden layer of the decoder model )*

*Validation Split = 0.3  ( Ratio that splits data from training set for validation through epocs )*


#### Loss Function

In VAE, we convert the input data into two vector namely Mean And  Variance in hidden space. We try to make distribution in hidden state like normal distibution. 

In AutoEncoders, we use MSE loss function, but in VAEs, additionally, use a regularization parameter, KL Divergence. KL Divergence prevents overfitting and support model to save important feautures. 

In loss function below  calculates reconstruction loss based on binary cross entrapy and KL divergence. It returns the mean of them.  ( References - 3)


```
def loss(true, pred):

reconstruction_loss = binary_crossentropy(keras.flatten(true), keras.flatten(pred)) * n_inputs * time_steps
kl_loss = 1 + variance_v - keras.square(mean_v) - keras.exp(variance_v)
kl_loss = keras.sum(kl_loss, axis=-1)
kl_loss *= -0.5
return keras.mean(reconstruction_loss + kl_loss)
```

#### Activation function

I choose activation function Sigmoid in last step because we need a data between zero and one to reach a greyscale image at the end.

I choose loss function as mean square error as in stated project requirements.


### 4 - Results

#### Training Set 1

![](https://github.com/mateskarayol/DeepLearning_3_VAE/blob/master/result/result1.png)

#### Training Set 2

![](https://github.com/mateskarayol/DeepLearning_3_VAE/blob/master/result/result2.png)

#### Training Set 3

![](https://github.com/mateskarayol/DeepLearning_3_VAE/blob/master/result/result3.png)


***References***

*1 - https://keras.io/examples/variational_autoencoder/*

*2 - https://colab.research.google.com/drive/1X3L28H6HHBAWZ-42l-zRhMgeyJoGndDa?usp=sharing#scrollTo=IawMfcajJarL*

*3 -https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/*

*4 - https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/*

*5 - https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73*

*6 - https://keras.io/examples/variational_autoencoder_deconv/*

*7- https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73*
