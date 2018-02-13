---
--- Created by shuieryin.
--- DateTime: 27/12/2017 2:50 PM
---

require "../../../lib/util"

require "nn"
require "image"
require "optim"
require "util"
require "gnuplot"
local matio = require 'matio'
local Plot = require 'itorch.Plot'

function loadData(filename)
    -- load all arrays from file
    local tensors = matio.load(filename, { 'X', 'y' })
    local X = tensors['X']
    local y = tensors['y']

    -- Plot training data
    local plot = Plot():circle(X[{ {}, 1 }][y:eq(1)], X[{ {}, 2 }][y:eq(1)], 'black', 'Positive')
    plot:circle(X[{ {}, 1 }][y:eq(0)], X[{ {}, 2 }][y:eq(0)], 'magenta', 'Negative')
    plot:title('Data')
    plot:xaxis('X1'):yaxis('X2')
    plot:legend(true):draw()
    itorchHtml(plot, splitFilename(filename) .. '.html')
    return X, y, plot
end

function loadData2(filename)
    -- load all arrays from file
    local tensors = matio.load(filename, { 'X', 'y', 'Xval', 'yval' })
    local X = tensors['X']
    local y = tensors['y']
    local Xval = tensors['Xval']
    local yval = tensors['yval']

    -- Plot training data
    local plot = Plot():circle(X[{ {}, 1 }][y:eq(1)], X[{ {}, 2 }][y:eq(1)], 'black', 'Positive')
    plot:circle(X[{ {}, 1 }][y:eq(0)], X[{ {}, 2 }][y:eq(0)], 'magenta', 'Negative')
    plot:title('Data')
    plot:xaxis('X1'):yaxis('X2')
    plot:legend(true):draw()
    itorchHtml(plot, splitFilename(filename) .. '.html')
    return X, y, Xval, yval, plot
end

function gaussian(x1, x2, sigma)
    local sim = math.exp(-math.pow(x1 - x2, 2) / (2 * math.pow(sigma, 2)))
    return sim
end

function readFile(filename)
    local lines = lines_from(filename)
    return table.concat(lines, "\n")
end