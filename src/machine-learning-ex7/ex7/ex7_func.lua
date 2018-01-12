---
--- Created by shuieryin.
--- DateTime: 03/01/2018 2:46 PM
---

require "image"
require "optim"
require "util"
require "gnuplot"
local matio = require 'matio'
local Plot = require 'itorch.Plot'

function loadData(filename, color, title)
    -- load all arrays from file
    local tensors = matio.load(filename, { 'X' })
    local X = tensors['X']

    -- Plot training data
    local plot = Plot():circle(X[{ {}, 1 }], X[{ {}, 2 }], color, 'Data')
    plot:title(title)
    plot:xaxis('X1'):yaxis('X2')
    plot:legend(true):draw()
    itorchHtml(plot, splitFilename(filename) .. '.html')
    return X, plot
end