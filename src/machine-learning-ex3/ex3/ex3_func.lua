---
--- Created by shuieryin.
--- DateTime: 17/12/2017 9:34 PM
---

require "optim"
require "image"
local matio = require 'matio'

function loadData()
    --  We start the exercise by first loading and visualizing the dataset.
    --  You will be working with a dataset that contains handwritten digits.

    -- Load Training Data
    print('Loading and Visualizing Data ...')

    -- load all arrays from file
    local tensors = matio.load('ex3data1.mat', { 'X', 'y' })
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
    return X, y, m
end