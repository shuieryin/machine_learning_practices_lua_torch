---
--- Created by shuieryin.
--- DateTime: 31/12/2017 5:45 PM
---

require "../../../lib/util"

require "optim"
require "util"
require "svm"
require "ex6_func"
require 'processEmail'
require 'getVocabList'
require 'emailFeatures'
local matio = require 'matio'

local function genSvmData(filename, X, y)
    local svmRawData = {}
    for i = 1, X:size(1) do
        local sign = "+"
        if y[i][1] == 0 then
            sign = "-"
        end
        local curRowData = {}
        for j = 1, X:size(2) do
            curRowData[#curRowData + 1] = j .. ":" .. X[i][j]
        end
        svmRawData[#svmRawData + 1] = sign .. "1 " .. table.concat(curRowData, " ")
        io.write(".")
        io.flush()
    end

    local file = io.open(filename, "w")
    file:write(table.concat(svmRawData, "\n"))
    file.close()

    print("\nDone!")
end

local trainDataFilename = "spamTrain.svmt"
if file_exists(trainDataFilename) == false then
    io.write("Loading training data...")
    io.flush()
    local tensors = matio.load('spamTrain.mat', { 'X', 'y' })
    local X = tensors['X']
    local y = tensors['y']

    genSvmData(trainDataFilename, X, y)
end

local testDataFilename = "spamTest.svmt"
if file_exists(testDataFilename) == false then
    io.write("Loading test data...")
    io.flush()
    local tensors = matio.load('spamTest.mat', { 'Xtest', 'ytest' })
    local Xtest = tensors['Xtest']
    local ytest = tensors['ytest']

    genSvmData(testDataFilename, Xtest, ytest)
end

local trainSvmDataFilename = "spamTrain.svm"
local trainSvmData
if file_exists(trainSvmDataFilename) == false then
    trainSvmData = svm.ascread(trainDataFilename)
    torch.save(trainSvmDataFilename, trainSvmData)
else
    trainSvmData = torch.load(trainSvmDataFilename)
end

local testSvmDataFilename = "spamTest.svm"
local testSvmData
if file_exists(testSvmDataFilename) == false then
    testSvmData = svm.ascread(testDataFilename)
    torch.save(testSvmDataFilename, testSvmData)
else
    testSvmData = torch.load(testSvmDataFilename)
end

local model = liblinear.train(trainSvmData)
liblinear.predict(testSvmData, model)

local filename = 'spamSample1.txt'

-- Read and predict
local file_contents = readFile(filename)
local word_indices = processEmail(file_contents)
local x = emailFeatures(word_indices)
local dataTBP = {
    [1] = {
        [2] = {
            [1] = torch.range(1, x:size(1)):int(),
            [2] = x:float()
        }
    }
}
local labels, accuracy, dec = liblinear.predict(dataTBP, model)
print('(1 indicates spam, -1 indicates not spam):', labels[1])