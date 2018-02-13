---
--- Created by shuieryin.
--- DateTime: 18/12/2017 7:20 PM
---

require "../../../lib/util"

require "optim"
require "util"
require "ex4_func"
local matio = require 'matio'
local Plot = require 'itorch.Plot'

-- Machine Learning Online Class - Exercise 4 Neural Network Learning
--  Instructions
--  ------------

--  This file contains code that helps you get started on the
--  linear exercise. You will need to complete the following functions
--  in this exericse:

require 'sigmoidGradient'
require 'randInitializeWeights'
require 'nnCostFunction'
require 'displayData'
require 'checkNNGradients'
require 'predict'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- Setup the parameters you will use for this exercise
local input_layer_size = 400  -- 20x20 Input Images of Digits
local hidden_layer_size = 25   -- 25 hidden units
local num_labels = 10          -- 10 labels, from 1 to 10
-- (note that we have mapped "0" to label 10)

-- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  You will be working with a dataset that contains handwritten digits.

local X, y = loadData()

-- In this part of the exercise, we load some pre-initialized
-- neural network parameters.

pause()

-- ================ Part 2: Loading Parameters ================
-- In this part of the exercise, we load some pre-initialized
-- neural network parameters.

print('Loading Saved Neural Network Parameters ...')

-- Load the weights into variables Theta1 and Theta2
-- load all arrays from file
local tensors = matio.load('ex4weights.mat', { 'Theta1', 'Theta2' })
local Theta1 = tensors['Theta1']
local Theta2 = tensors['Theta2']

-- In this part of the exercise, we load some pre-initialized
-- neural network parameters.

-- Unroll parameters
local nn_params = torch.reshape(Theta1, Theta1:numel(), 1):cat(torch.reshape(Theta2, Theta2:numel(), 1), 1)

-- ================ Part 3: Compute Cost (Feedforward) ================
--  To the neural network, you should first start by implementing the
--  feedforward part of the neural network that returns the cost only. You
--  should complete the code in nnCostFunction.m to return cost. After
--  implementing the feedforward to compute the cost, you can verify that
--  your implementation is correct by verifying that you get the same cost
--  as us for the fixed debugging parameters.

--  We suggest implementing the feedforward cost *without* regularization
--  first so that it will be easier for you to debug. Later, in part 4, you
--  will get to implement the regularized cost.

print('\nFeedforward Using Neural Network ...')

-- Weight regularization parameter (we set this to 0 here).
local lambda = 0

local J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

print('Cost at parameters (loaded from ex4weights):')
print(J)
print("This value should be about 0.287629")
pause()

-- =============== Part 4: Implement Regularization ===============
--  Once your cost function implementation is correct, you should now
--  continue to implement the regularization with the cost.

print('\nChecking Cost Function (w/ Regularization) ... ')

-- Weight regularization parameter (we set this to 1 here).
lambda = 1

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

print('Cost at parameters (loaded from ex4weights):')
print(J)
print("This value should be about 0.383770")
pause()

-- ================ Part 5: Sigmoid Gradient  ================
--  Before you start implementing the neural network, you will first
--  implement the gradient for the sigmoid function. You should complete the
--  code in the sigmoidGradient.m file.

print('\nEvaluating sigmoid gradient...')

local g = sigmoidGradient(torch.Tensor({ -1, -0.5, 0, 0.5, 1 }))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
print(g)
pause()

-- ================ Part 6: Initializing Pameters ================
--  In this part of the exercise, you will be starting to implment a two
--  layer neural network that classifies digits. You will start by
--  implementing a function to initialize the weights of the neural network
--  (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...')

local initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
local initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

-- Unroll parameters
local initial_nn_params = torch.reshape(initial_Theta1, initial_Theta1:numel(), 1):cat(torch.reshape(initial_Theta2, initial_Theta2:numel(), 1), 1)

-- =============== Part 7: Implement Backpropagation ===============
--  Once your cost matches up with ours, you should proceed to implement the
--  backpropagation algorithm for the neural network. You should add to the
--  code you've written in nnCostFunction.m to return the partial
--  derivatives of the parameters.

print('Checking Backpropagation... ')

--  Check gradients by running checkNNGradients
checkNNGradients()
pause()

-- =============== Part 8: Implement Regularization ===============
--  Once your backpropagation implementation is correct, you should now
--  continue to implement the regularization with the cost and gradient.

print('\nChecking Backpropagation (w/ Regularization) ...')

--  Check gradients by running checkNNGradients
lambda = 3
checkNNGradients(lambda)

-- Also output the costFunction debugging values
local debug_J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

print('\nCost at (fixed) debugging parameters (w/ lambda = %f): ', lambda )
print('(For lambda = 3, this value should be about 0.576051):')
print(debug_J)
pause()

-- =================== Part 8: Training NN ===================
--  You have now implemented all the code necessary to train a neural
--  network. To train your neural network, we will now use "fmincg", which
--  is a function which works similarly to "fminunc". Recall that these
--  advanced optimizers are able to train our cost functions efficiently as
--  long as we provide them with the gradient computations.

print('\nTraining Neural Network...')

--  You should also try different values of lambda
lambda = 1
-- Now, costFunction is a function that takes in only one argument(the
-- neural network parameters)
local costHistory = {}
local iterCount = 1
local last_str = ''
local trained_nn_params = optim.cg( function(p)
    local curCost, curParams = nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
    io.write(('\b \b'):rep(#last_str))
    local str = 'Iteration    ' .. iterCount .. ' | Cost: ' .. curCost
    io.write(str)
    io.flush()
    last_str = str
    costHistory[iterCount] = curCost
    iterCount = iterCount + 1
    return curCost, curParams
end, initial_nn_params, { maxIter = 50 })

-- plot costHistory
local plot = Plot():line(torch.range(1, #costHistory):totable(), costHistory, 'blue', 'Cost convergence'):legend(true):title('Line Plot Demo'):draw()
plot:title('Cost history')
plot:xaxis('Iter times'):yaxis('Cost')
plot:legend(true):draw()
itorchHtml(plot, 'costHistory.html')

-- Obtain Theta1 and Theta2 back from nn_params
local endPos = hidden_layer_size * (input_layer_size + 1)
Theta1 = torch.reshape(trained_nn_params[{ { 1, endPos } }], hidden_layer_size, (input_layer_size + 1))
Theta2 = torch.reshape(trained_nn_params[{ { 1 + endPos, trained_nn_params:numel() } }], num_labels, (hidden_layer_size + 1))
print('')
pause()

-- ================= Part 9: Visualize Weights =================
--  You can now "visualize" what the neural network is learning by
--  displaying the hidden units to see what features they are capturing in
--  the data.

print('Visualizing Neural Network... ')
local displayData = displayData(Theta1[{ {}, { 2, Theta1:size(2) } }])
image.save("Theta1.png", image.toDisplayTensor {
    input = displayData
})
os.execute(openCmd .. ' "' .. paths.cwd() .. '/Theta1.png"')
pause()

-- ================= Part 10: Implement Predict =================
--  After training the neural network, we would like to use it to predict
--  the labels. You will now implement the "predict" function to use the
--  neural network to predict the labels of the training set. This lets
--  you compute the training set accuracy.

local pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: ', pred:eq(y:long()):sum() / y:size(1) * 100)
