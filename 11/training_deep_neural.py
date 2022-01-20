
#--------------------------------------------------------------
# VANISHING/EXPLODING GRADIANTS
# backpropagation algorithm works by going from the output layer
# to the input layer, propagating the error gradient along the 
# way. Once the algorithm has computed the gradient of the cost
# function with regard to each parameter in the network, it uses
# these gradients to update each parameter with a Gradient Descent
# step

# By defautl, Keras uses Glorot initialization with a uniform 
# distribution. you can change this to He initialization by
# setting

Keras.layers.Dense(10, activation="relu", Kernel_initializer"he_normal")

# if you want He initialization with a uniform distribution but
# based on fan_avg rather than fan_in, you can use the 
# VarianceScaling initializer like this

he_avg_init = keras.initializers.VarianceScaling(scale=2,
mode='fan_avg', distribution='uniform')

keras.layers.Dense(10, activation='sigmoid', kernel_initializer=
he_avg_init)

# so which activation function should you use for the hidden layers
# of your deep neural networks? Although your mileage will vary,
# in general SELU > ELU > LEAKY RELU(and its variants)>  RELU >
# tanh > logistic.
# ReLU is the most used activation function(by far), many libraries
# and hardware acelerators provide ReLU-specific optimizations;
# therefor, if speed is your priority, Rellu might still be the 
# best choice

# To use Leaky ReLU activation function, create a LeakyReLU layer
# and add it to your model just after the layer you want to apply
# it to 

model = Keras.models.Sequential([
    [...],
    Keras.layers.Dense(10, kernel_initializer="he_normal"),
    Keras.layers.LeakyReLU(alpha=0.2),
    [...]
])

# PReLU : no official implementation
#---------------------------------------------------------------
# Implementing Batch Normalization with Keras

model = keras.model.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu",
    kernel_initializer="he_normal"),
    keras.layers.BatchNormaliztion(),
    keras.layers.Dense(100, activation="elu"),
    kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

# for deeper networks, batch normalization can make a huge 
# impact
# The authors of the BN paper argued in favor of adding the BN
# layers before the activation functions, rather than after.
# To add the BN layers before the activation functions, you must
# remove the activation function from the hidden layers and add 
# them as separate layers after the BN layers. Moreover, since a
# Batch Mormalization layer includes one offset parameter per input,
# you can remove the bias term from the previous layer
# see below:

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer="he_normal", 
    use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100, kernel_initializer="he_normal",
use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax")
])

#--------------------------------------------------------------
# GRADIANT CLIPPING
# This is a popular techniqur to mitagate the exploding gradients
# by cliping the gradients during backpropagation so that they
# never exceed somethreshold
# This technique is mostly used in recurrent neural networks as 
# Batch Normalization is tricky to use in RNNs. for other types
# of network BN is usually sufficient


#----------------------------------------------------------------
# REUSING PRETRAINED LAYERS( transfer learning )
# it is generally not a good idea to train a very large DNN from
# scratch: instead, you should always try to find an existing
# neural network that accomplishes a similar task to the one you
# are trying to tackle. (see chapter 14 to see how to find them)

# This technique speeds up training and also require significantly
# less training data

# NOTE------------------------------------
# If the input pictures of your new task don’t have the same size as the ones used in
# the original task, you will usually have to add a preprocessing step to resize them to
# the size expected by the original model. More generally, transfer learning will work
# best when the inputs have similar low-level features


#----------------------------------------------------------------
# UNSUPERVISED PRETRAINING
# Suppose you want to tackle a complex task for which you don't
# have much labeled training data, but unfortunately you cannot find
# a model trained on a similar task. Don't lose hope! First, you 
# should try to gather more labeled training data, but if you can't,
# you may still be able to perform "unsupervised pretraining".
# Indeed, it is often cheap to gather unlabeled training examples, but
# expensive to label them. if you can gather plenty of unlabeled
# training data, you can try to use it to train an unsupervised model,
# such as an autoencoder or a generative adversarial network. Then
# you can reuse the lower layers of the autoencoder or the lower
# layers of the GAN's discriminator, add the output layer for your 
# task on top and fine-tune the final network using supervised
# learning(i.e with the labeled training examples)

# Unsupervised pretraining( today typically using autoencoders or 
# GANs rather than RBMs) is still a good option when you have a 
# complex task to solve, no similar model you can reuse, and little
# labeled training data but plenty of unlabeled training data.

#--------------------------------------------------------------
# PRETRAINING ON AN AUXILIARY TASK
# if you do not have much labeled training data, one last option
# is to train  afirst neural network on an auxiliary task for 
# which you can easily obtain or generate labeled training data, 
# then reuse the lower layers of that network for your actual
# task. The first neural network's lower layers will learn feature
# detectors that will likely be resusable by the second neural
# network
# for example, if you want to build a system to recognize faces, 
# you may only have a few pictures of each individual- clearly not
# enought to train a good classifier. Gathering hundreds of pictures
# of each person would not be pratical. You could, however, gather
# a lot of pictures of random people on the web and train a first
# neural network whether or not two different pictures feature the
# same person. such a network would learn good feature detectors
# for faces, so reusing its lower layers would allow you to train
# a good face classifier that uses little training data.


# SELF-SUPERVISED Learning is when you automatically generate the
# labels from the data itself, then you train a model on the 
# resulting "labeled" dataset using supervised learning techniques.
# since this approach requires no human labeling whatsoever, it is 
# best classified as a form of unsupervised learning


#---------------------------------------------------------------
# FOUR WAYS TO SPEED UP TRAINING( and reach a better solution )
# 1. Applying a good initialization strategy for the connection
#    weights.
# 2. Using a good activation function.
# 3. Using Batch Normalization.
# 4. Reusing parts of a pretrained network(possibly built on an
#    auxillary task or using unsupervised learning)
# 5. Another huge speed boost comes from using a faster optimizer
#    

#----------------------------------------------------------------
# FAST OPTIMIZERS
# momentum optimizer:

optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

# the onedrawback of momentum optimization is that it adds yet
# another hyperparameter to tune. However, the momentum value of 
# 0.9 usually works well in practice and almost always goes faster
# than regular Gradient Descent

# Nesterov Accelerated Gradient:
# to use NAG see the implementation below:

optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)


# AdaGrad optimizer:
# AdaGrad frequently performs well for simple quadratic problems,
# but it often stops too early when training neural networks. The
# learning rate gets scaled down so much that the algorithm ends
# up stopping entirely before reaching the global optimum. you 
# should not use it to train deep neural networks(it may be 
# efficient for simpler tasks such as Linear Regression, though)


# NOTE----------------------------------------------------
# Adaptive optimization methods(including RMSProp, Adam, and Nadam 
# optimization) are often great, converging fast to a good solution.
# However, a 2017 paper by Ashia C. wilson et al. showed that they 
# can lead to solutions that generalize poorly on some datasets. So 
# when you are disappointed by your model's performance, try using 
# plain Nesterov Accelerated Gradient instead: your dataset may
# just be allergic to adaptive gradients. Also check out the 
# latest research, because it's moving fast

#All the optimization algorithms just presented produce dense models,
#meaning that most parameters will be nonzero. If you need a blazingly
#fast model at runtime, or if you need it to take up less memory, you
#may prefer to end up with a sparse model instead.
#One easy way to achieve this is to train the model as usual, then get rid
#of the tiny weights (set them to zero). Note that this will typically not
#lead to a very sparse model, and it may degrade the model’s
#performance.
#A better option is to apply strong ℓ  regularization during training (we
#will see how later in this chapter), as it pushes the optimizer to zero
#out as many weights as it can (as discussed in “Lasso Regression” in
#Chapter 4).
#If these techniques remain insufficient, check out the TensorFlow
#Model Optimization Toolkit (TF-MOT), which provides a pruning API
#capable of iteratively removing connections during training based on
#their magnitude.

#--------------------------------------------------------------
# OPTIMIZER COMPARISON
#----------------------------------------------------------------
# the best optimizers are listed in descending order from the best
# to the worst
# 1. AdaMax
# 2. Nadam
# 3. Adam
# 4. RMSprop
# 5. SGD(momentum=...., nesterov=True)
# 6. SGD(momentum=....)

#----------------------------------------------------------------
# Learning Rate Scheduling
# Finding a good learning rate is very important. If you set it much too high,
# training may diverge (as we discussed in “Gradient Descent”). If you set it
# too low, training will eventually converge to the optimum, but it will take
# a very long time. If you set it slightly too high, it will make progress very
# quickly at first, but it will end up dancing around the optimum, never
# really settling down. If you have a limited computing budget, you may
# have to interrupt training before it has converged properly, yielding a
# suboptimal solution
# But you can do better than a constant learning rate: if you start with a large
# learning rate and then reduce it once training stops making fast progress,
# you can reach a good solution faster than with the optimal constant
# learning rate. There are many different strategies to reduce the learning
# rate during training. It can also be beneficial to start with a low learning
# rate, increase it, then drop it again. These strategies are called learning
# schedules 
