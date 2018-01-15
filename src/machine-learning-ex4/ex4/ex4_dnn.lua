---
--- Created by shuieryin.
--- DateTime: 18/12/2017 7:20 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "optim"
require "util"
require "ex4_func"

-- Setup the parameters you will use for this exercise
local input_layer_size = 400  -- 20x20 Input Images of Digits
local hidden_layer_size = 25   -- 25 hidden units
local num_labels = 10          -- 10 labels, from 1 to 10
-- (note that we have mapped "0" to label 10)

local X, y = loadData()
local m = X:size(1)
local trainingSetBoundary = math.floor(m * 0.98)
local cvSetBoundary = trainingSetBoundary + math.floor((m - trainingSetBoundary) / 2)
local randomOrder = torch.randperm(m)
local trainOrder = randomOrder[{ { 1, trainingSetBoundary } }]
local cvOrder = randomOrder[{ { trainingSetBoundary + 1, cvSetBoundary } }]
local testOrder = randomOrder[{ { cvSetBoundary + 1, m } }]

local trainM = trainOrder:size(1)
local Xtrain = torch.Tensor(trainM, X:size(2))
local Ytrain = torch.Tensor(trainM, y:size(2))
for i = 1, trainM do
    Xtrain[i] = X[trainOrder[i]]
    Ytrain[i] = y[trainOrder[i]]
end

local cvM = cvOrder:size(1)
local Xcv = torch.Tensor(cvM, X:size(2))
local Ycv = torch.Tensor(cvM, y:size(2))
for i = 1, cvM do
    Xcv[i] = X[cvOrder[i]]
    Ycv[i] = y[cvOrder[i]]
end

local testM = cvOrder:size(1)
local Xtest = torch.Tensor(testM, X:size(2))
local Ytest = torch.Tensor(testM, y:size(2))
for i = 1, testM do
    Xtest[i] = X[testOrder[i]]
    Ytest[i] = y[testOrder[i]]
end

-- In this part of the exercise, we load some pre-initialized
-- neural network parameters.

local net = nn.Sequential()  -- make a multi-layer perceptron
net:add(nn.Linear(input_layer_size, math.floor(hidden_layer_size * 1.5)))
net:add(nn.ReLU())
net:add(nn.Linear(math.floor(hidden_layer_size * 1.5), hidden_layer_size))
net:add(nn.ReLU())
net:add(nn.Linear(hidden_layer_size, math.floor(hidden_layer_size / 2)))
net:add(nn.ReLU())
net:add(nn.Linear(math.floor(hidden_layer_size / 2), num_labels))
net:add(nn.Tanh())

local costHistory = {}
local learningRate = 0.02
local criterion = nn.MSECriterion()
local trainErrorHistory = {}
local cvErrorHistory = {}
local accuracyHistory = {}
local cvErrorDiffHistory = {}
net:training()
local trainErrorFunc = function(pred, y)
    return criterion:forward(pred, y)
end
local y_labels = {}
for i = 1, num_labels do
    local y_label = torch.zeros(num_labels, 1)
    y_label[i] = 1
    y_labels[i] = y_label
end

local cvErrorFunc = function()
    local cvError = 0
    for k = 1, cvM do
        local y_label = y_labels[Ycv[k][1]]
        local predCv = net:forward(Xcv[k])
        cvError = cvError + criterion:forward(predCv, y_label)
    end
    return cvError / cvM
end

local targetErrorDiff = 0.01
local count = 1
local last_str = ''
while count <= 50 do
    local trainError = nnTorch(net, y_labels, Xtrain, Ytrain, criterion, learningRate, trainErrorFunc)
    local cost = net:getParameters():sum()
    local cvError = cvErrorFunc()
    costHistory[count] = cost
    trainErrorHistory[count] = trainError
    cvErrorHistory[count] = cvError
    local predict_max, pred = net:forward(Xtest):max(2)
    local accuracy = pred:eq(Ytest:long()):sum() / Ytest:size(1)
    accuracyHistory[count] = accuracy
    local lastCvError = cvErrorHistory[count - 1] or 0
    local cvErrorDiff = math.abs(cvError - lastCvError)

    io.write(('\b \b'):rep(#last_str))
    local str = "iter " .. count ..
            " | cost: " .. cost ..
            " | train error: " .. trainError ..
            " | cv error: " .. cvError ..
            " | cv error diff : " .. cvErrorDiff ..
            " | accuracy: " .. accuracy
    io.write(str)
    io.flush()
    last_str = str
    if lastCvError ~= 0 then
        cvErrorDiffHistory[#cvErrorDiffHistory + 1] = cvErrorDiff
        if trainError < targetErrorDiff then
            break
        end
    end
    count = count + 1
end
print('')

-- plot costHistory
plotTable({
    [1] = {
        data = costHistory,
        desc = 'Cost convergence',
        color = 'blue'
    }
}, 'Cost history', 'Iter times', 'Cost', 'costHistory')

-- plot error
plotTable({
    [1] = {
        data = trainErrorHistory,
        desc = 'Train error',
        color = 'blue'
    },

    [2] = {
        data = cvErrorHistory,
        desc = 'Cv error',
        color = 'red'
    }
}, 'Error history', 'Iter times', 'Error', 'errorHistory')

-- plot cvErrorDiffHistory
plotTable({
    [1] = {
        data = cvErrorDiffHistory,
        desc = 'Cv error diff',
        color = 'blue'
    }
}, 'Cv error diff history', 'Iter times', 'Diff', 'cvErrorDiffHistory')

-- Obtain Theta1 and Theta2 back from nn_params
local trained_nn_params = net:getParameters()
local endPos = hidden_layer_size * (input_layer_size + 1)
local Theta1 = torch.reshape(trained_nn_params[{ { 1, endPos } }], hidden_layer_size, input_layer_size + 1)

--  You can now "visualize" what the neural network is learning by
--  displaying the hidden units to see what features they are capturing in
--  the data.

print('Visualizing Neural Network... ')
local displayData = displayData(Theta1[{ {}, { 2, Theta1:size(2) } }])
image.save("Theta1.png", image.toDisplayTensor {
    input = displayData
})
os.execute(openCmd .. ' "' .. paths.cwd() .. '/Theta1.png"')

net:evaluate()
local predict_max, pred = net:forward(Xtest):max(2)
print('Training Set Accuracy: ', pred:eq(Ytest:long()):sum() / Ytest:size(1) * 100)