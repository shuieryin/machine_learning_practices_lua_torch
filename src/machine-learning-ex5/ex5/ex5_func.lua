---
--- Created by shuieryin.
--- DateTime: 24/12/2017 9:30 PM
---

require "nn"
require "image"
require "optim"
local matio = require 'matio'
local Plot = require 'itorch.Plot'

function loadData()
    -- load all arrays from file
    local tensors = matio.load('ex5data1.mat', { 'X', 'y', 'Xval', 'yval', 'Xtest', 'ytest' })
    local X = tensors['X']
    local y = tensors['y']
    local Xval = tensors['Xval']
    local yval = tensors['yval']
    local Xtest = tensors['Xtest']
    local ytest = tensors['ytest']

    -- Plot training data
    local plot = Plot():circle(X[{ {}, 1 }], y[{ {}, 1 }], 'red', 'Training data')
    plot:title('Data')
    plot:xaxis('Change in water level (x)'):yaxis('Water flowing out of the dam (y)')
    plot:legend(true):draw()
    itorchHtml(plot, 'data.html')
    return X, y, Xval, yval, Xtest, ytest, plot
end