---
--- Created by shuieryin.
--- DateTime: 17/12/2017 9:34 PM
---

require "nn"
require "image"
local matio = require 'matio'

function loadData()
    --  We start the exercise by first loading and visualizing the dataset.
    --  You will be working with a dataset that contains handwritten digits.

    -- Load Training Data
    print('Loading and Visualizing Data ...')

    -- load all arrays from file
    local tensors = matio.load('ex4data1.mat', { 'X', 'y' })
    local X = tensors['X']
    local y = tensors['y']
    local m = X:size(1)

    -- Randomly select 100 data points to display
    math.randomseed(os.time())
    local sel = X[{ math.random(m), {} }]
    for i = 2, 100 do
        sel = sel:cat(X[{ math.random(m), {} }], 2)
    end
    sel = sel:t()
    local displayData = displayData(sel)
    image.save("data.png", image.toDisplayTensor {
        input = displayData
    })
    os.execute(openCmd .. ' "' .. paths.cwd() .. '/data.png"')
    return X, y
end

local function sigmoid(z)
    --SIGMOID Compute sigmoid functoon
    --   J = SIGMOID(z) computes the sigmoid of z.
    return torch.pow(1 + (-z):exp(), -1)
end

function nnTorch(net, y_labels, X, y, criterion, learningRate, errFunc)
    local m = X:size(1)
    local err = 0
    for i = 1, m do
        local y_label = y_labels[y[i][1]]
        local x = X[i]
        local pred = net:forward(x)
        err = err + errFunc(pred, y_label)
        net:zeroGradParameters()
        local critGrad = criterion:backward(pred, y_label)
        net:backward(x, critGrad)
        net:updateParameters(learningRate)
    end
    return err / m
end