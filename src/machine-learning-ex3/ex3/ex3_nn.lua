---
--- Created by shuieryin.
--- DateTime: 18/12/2017 7:20 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "torch"
require "ex3_func"
local matio = require 'matio'

-- Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks
--  Instructions
--  ------------

--  This file contains code that helps you get started on the
--  linear exercise. You will need to complete the following functions
--  in this exericse:

require 'lrCostFunction' --(logistic regression cost function)
require 'oneVsAll'
require 'predictOneVsAll'
require 'predict'
require 'displayData'

--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.

-- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  You will be working with a dataset that contains handwritten digits.

local X, y, m = loadData()
pause()

-- ================ Part 2: Loading Pameters ================
-- In this part of the exercise, we load some pre-initialized
-- neural network parameters.

print('Loading Saved Neural Network Parameters ...')

-- Load the weights into variables Theta1 and Theta2
-- load all arrays from file
local tensors = matio.load('ex3weights.mat', { 'Theta1', 'Theta2' })
local Theta1 = tensors['Theta1']
local Theta2 = tensors['Theta2']

-- ================= Part 3: Implement Predict =================
--  After training the neural network, we would like to use it to predict
--  the labels. You will now implement the "predict" function to use the
--  neural network to predict the labels of the training set. This lets
--  you compute the training set accuracy.

local pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: ', pred:eq(y):sum() / length(pred) * 100)
pause()

--  To give you an idea of the network's output, you can also run
--  through the examples one at the a time to see what it is predicting.

--  Randomly permute examples
local rp = torch.randperm(m)

for i = 1, m do
    -- Display
    print('Displaying Example Image')
    local curX = X[rp[i]]
    local curY = y[rp[i]][1]

    curX = torch.reshape(curX, 1, curX:numel())
    displayData(curX)

    pred = predict(Theta1, Theta2, curX)
    print('Neural Network Prediction: ' .. pred[1][1] .. ' (digit ' .. curY .. ')')

    -- Pause with quit option
    print('Paused - press enter to continue, q to exit:')
    local s = io.read()
    if s == 'q' then
        break
    end
end