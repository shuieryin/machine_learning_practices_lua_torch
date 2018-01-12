---
--- Created by shuieryin.
--- DateTime: 10/12/2017 9:09 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "optim"
require "ex1_func"
local Plot = require 'itorch.Plot'

-- Machine Learning Online Class
--  Exercise 1: Linear regression with multiple variables
--
--  Instructions
--  ------------
--
--  This file contains code that helps you get started on the
--  linear regression exercise.
--
--  You will need to complete the following functions in this
--  exericse:

require 'warmUpExercise'
require 'plotData'
require 'gradientDescent'
require 'computeCost'
require 'gradientDescentMulti'
require 'computeCostMulti'
require 'featureNormalize'
require 'normalEqn.lua'

--  For this part of the exercise, you will need to change some
--  parts of the code below for various experiments (e.g., changing
--  learning rates).

-- ================ Part 1: Feature Normalization ================
print('Loading data ...\n')
local X, y, theta, X_ori = loadData("ex1data2.txt")
-- Print out some data points
print('First 10 examples from the dataset:')
print(X[{ { 1, 10 } }]:cat(y[{ { 1, 10 } }], 2):t())
pause()

-- Scale features and set them to zero mean
print('Normalizing Features ...')

local X_norm, mu, sigma = featureNormalize(X_ori)
X = torch.ones(X:size(1), 1):cat(X_norm, 2)

-- ================ Part 2: Gradient Descent ================

-- Instructions: We have provided you with the following starter
--               code that runs gradient descent with a particular
--               learning rate (alpha).
--
--               Your task is to first make sure that your functions -
--               computeCost and gradientDescent already work with
--               this starter code and support multiple variables.
--
--               After that, try running gradient descent with
--               different values of alpha and see which one gives
--               you the best result.
--
--               Finally, you should complete the code at the end
--               to predict the price of a 1650 sq-ft, 3 br house.
--
-- Hint: By using the 'hold on' command, you can plot multiple
--       graphs on the same figure.
--
-- Hint: At prediction, make sure you do the same feature normalization.
print('Running gradient descent ...')

-- Choose some alpha value
local alpha = 0.01
local num_iters = 400
local finalTheta, costHistory, numIterMtx = gradientDescentMulti(X, y, theta, alpha, num_iters)

local plot = Plot():line(numIterMtx[{ 1, {} }], costHistory[{ 1, {} }], 'blue', 'Gradient descent')
plot:title('Convergence of gradient descent with an appropriate learning rate')
plot:xaxis('Number of iterations'):yaxis('Cost J')
plot:legend(true)
plot:draw()
itorchHtml(plot, 'convergence.html')

-- Display gradient descent's result
print('Theta computed from gradient descent:')
print(finalTheta)

-- Estimate the price of a 1650 sq-ft, 3 br house
-- ====================== YOUR CODE HERE ======================
local sq_norm = (1650 - mu[1][1]) / sigma[1][1]
local br_norm = (3 - mu[1][2]) / sigma[1][2]
local house = torch.Tensor({ { 1, sq_norm, br_norm } })
local price = house * finalTheta -- You should change this
-- ============================================================
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):", price:sum())

-- ================ Part 3: Normal Equations ================

-- Instructions: The following code computes the closed form
--               solution for linear regression using the normal
--               equations. You should complete the code in
--               normalEqn.lua
--
--               After doing so, you should complete this code
--               to predict the price of a 1650 sq-ft, 3 br house.
print('Solving with normal equations...')
-- Calculate the parameters from the normal equation
local anotherX = torch.ones(X:size(1), 1):cat(X_ori, 2)
local finalNeTheta = normalEqn(anotherX, y)

-- Display normal equation's result
print('Theta computed from the normal equations: ')
print(finalNeTheta)

-- Estimate the price of a 1650 sq-ft, 3 br house
local nePrice = 0
-- ====================== YOUR CODE HERE ======================

-- ============================================================
print("Predicted price of a 1650 sq-ft, 3 br house (using normal equations):", nePrice)