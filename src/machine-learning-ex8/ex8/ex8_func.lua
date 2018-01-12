---
--- Created by shuieryin.
--- DateTime: 08/01/2018 1:52 PM
---

require "image"
require "optim"
require "util"
require "gnuplot"
local matio = require 'matio'
local Plot = require 'itorch.Plot'

function loadData(filename, title)
    -- load all arrays from file
    local tensors = matio.load(filename, { 'X', 'Xval', 'yval' })
    local X = tensors['X']
    local Xval = tensors['Xval']
    local yval = tensors['yval']

    -- Plot training data
    local plot
    if title then
        plot = Plot():circle(X[{ {}, 1 }], X[{ {}, 2 }], 'blue', 'Data')
        plot:title(title)
        plot:xaxis('Latency (ms)'):yaxis('Throughput (mb/s)')
        plot:legend(false):draw()
        itorchHtml(plot, splitFilename(filename) .. '.html')
    end
    return X, Xval, yval, plot
end

function loadData2(filename)
    -- load all arrays from file
    local tensors = matio.load(filename, { 'Y', 'R' })
    local Y = tensors['Y']
    local R = tensors['R']

    return Y, R
end

function loadData3(filename)
    -- load all arrays from file
    local tensors = matio.load(filename, { 'X', 'Theta' })
    local X = tensors['X']
    local Theta = tensors['Theta']

    return X, Theta
end