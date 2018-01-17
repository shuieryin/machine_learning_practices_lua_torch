---
--- Created by shuieryin.
--- DateTime: 18/12/2017 7:20 PM
---

package.path = package.path .. ";../../../lib/?.lua"

require "util"
require "ex4_func"

-- Setup the parameters you will use for this exercise
local input_layer_size = 400  -- 20x20 Input Images of Digits
local hidden_layer_size = 25   -- 25 hidden units
local num_labels = 10          -- 10 labels, from 1 to 10
-- (note that we have mapped "0" to label 10)

local X, y = loadData()
local m = X:size(1)
local trainingSetBoundary = math.floor(m * 0.9)
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

local miniBatchSize = 1
local net = nn.Sequential()  -- make a multi-layer perceptron
net:add(nn.Linear(input_layer_size * miniBatchSize, math.floor(hidden_layer_size * 2) * miniBatchSize))
net:add(nn.ReLU())
net:add(nn.Linear(math.floor(hidden_layer_size * 2) * miniBatchSize, math.floor(hidden_layer_size / 1.5) * miniBatchSize))
net:add(nn.ReLU())
--net:add(nn.Linear(hidden_layer_size * miniBatchSize, math.floor(hidden_layer_size / 2) * miniBatchSize))
--net:add(nn.ReLU())
net:add(nn.Linear(math.floor(hidden_layer_size / 1.5) * miniBatchSize, num_labels * miniBatchSize))
net:add(nn.Tanh())

local learningRate = 0.02
local criterion = nn.MSECriterion()
local trainErrorHistory = {}
local cvErrorHistory = {}
local accuracyHistory = {}
local errorDiffHistory = {}
net:training()
local errorFunc = function(pred, y)
    return criterion:forward(pred, y)
end
local y_labels = torch.eye(num_labels)

local YcvLabels = torch.Tensor(cvM, y_labels:size(1))
for j = 1, cvM do
    YcvLabels[j] = y_labels[Ycv[j][1]]
end
local XcvReshape = torch.reshape(Xcv, Xcv:size(1) / miniBatchSize, Xcv:size(2) * miniBatchSize)

local predict = function(curNet, X, y)
    local x = torch.reshape(X, X:size(1) / miniBatchSize, X:size(2) * miniBatchSize)
    local predData = curNet:forward(x)
    predData = torch.reshape(predData, X:size(1), num_labels)
    local predict_max, pred = predData:max(2)
    local accuracy = pred:eq(y:long()):sum() / y:size(1)
    return accuracy
end

local dropout = nn.Dropout(0.01)
local targetErrorDiff = 5e-6
local count = 1
local last_str = ''
while count <= 50 do
    local trainError = nnTorch(net, y_labels, Xtrain, Ytrain, criterion, learningRate, errorFunc, dropout, miniBatchSize)
    local predCv = net:forward(XcvReshape)
    local cvError = errorFunc(predCv, YcvLabels)
    trainErrorHistory[count] = trainError
    cvErrorHistory[count] = cvError
    local trainAccuracy = predict(net, Xtrain, Ytrain)
    local cvAccuracy = predict(net, Xcv, Ycv)
    local testAccuracy = predict(net, Xtest, Ytest)
    accuracyHistory[count] = testAccuracy
    local lastCvError = cvErrorHistory[count - 1] or 0
    local errorDiff = math.abs(trainError - cvError)

    io.write(('\b \b'):rep(#last_str + 1))
    local str = "\niter " .. count ..
            --" | train error: " .. trainError ..
            --" | cv error: " .. cvError ..
            " | error diff : " .. errorDiff ..
            " | train accuracy: " .. trainAccuracy ..
            " | cv accuracy: " .. cvAccuracy ..
            " | test accuracy: " .. testAccuracy
    io.write(str)
    io.flush()
    last_str = str
    if lastCvError ~= 0 then
        errorDiffHistory[#errorDiffHistory + 1] = errorDiff
        if errorDiff < targetErrorDiff then
            break
        end
    end
    count = count + 1
end
print('')

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
        data = errorDiffHistory,
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
local predData = net:forward(torch.reshape(Xtest, Xtest:size(1) / miniBatchSize, Xtest:size(2) * miniBatchSize))
predData = torch.reshape(predData, Xtest:size(1), num_labels)
local predict_max, pred = predData:max(2)
print('Training Set Accuracy: ', pred:eq(Ytest:long()):sum() / Ytest:size(1) * 100)