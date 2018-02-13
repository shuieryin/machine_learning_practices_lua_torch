---
--- Created by shuieryin.
--- DateTime: 30/12/2017 7:16 PM
---

require "../../../lib/util"

require "optim"
require "util"
require "ex6_func"
local matio = require 'matio'

--%% Machine Learning Online Class
--%  Exercise 6 | Spam Classification with SVMs
--%  Instructions
--%  ------------

--%  This file contains code that helps you get started on the
--%  exercise. You will need to complete the following functions:
--%
require 'gaussianKernel'
require 'dataset3Params'
require 'processEmail'
require 'emailFeatures'
require 'getVocabList'
require 'svmTrain'
require 'svmPredict'

--%  For this exercise, you will not need to change any code in this file,
--%  or any other files other than those mentioned above.

-- ==================== Part 1: Email Preprocessing ====================
--  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
--  to convert each email into a vector of features. In this part, you will
--  implement the preprocessing steps for each email. You should
--  complete the code in processEmail.m to produce a word indices vector
--  for a given email.

print('Preprocessing sample email (emailSample1.txt)')

-- Extract Features
local file_contents = readFile('emailSample1.txt')
local word_indices = processEmail(file_contents)

-- Print Stats
print('Word Indices:')
print(word_indices)

pause()

-- ==================== Part 2: Feature Extraction ====================
--  Now, you will convert each email into a vector of features in R^n.
--  You should complete the code in emailFeatures.m to produce a feature
--  vector for a given email.

print('Extracting features from sample email (emailSample1.txt)')

-- Extract Features
local features = emailFeatures(word_indices)

-- Print Stats
print('Length of feature vector: ', features:size(1))
print('Number of non-zero entries: ', torch.sum(features:gt(0)))

pause()

-- =========== Part 3: Train Linear SVM for Spam Classification ========
--  In this section, you will train a linear classifier to determine if an
--  email is Spam or Not-Spam.

-- Load the Spam Email dataset
-- You will have X, y in your environment
local tensors = matio.load('spamTrain.mat', { 'X', 'y' })
local X = tensors['X']
local y = tensors['y']

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

local C = 0.1
local model = svmTrain(X, y, C, { name = "linearKernel" })

local p = svmPredict(model, X)

print('Training Accuracy: ', p:eq(y):sum() / y:size(1) * 100)

-- =================== Part 4: Test Spam Classification ================
--  After training the classifier, we can evaluate it on a test set. We have
--  included a test set in spamTest.mat

-- Load the test dataset
-- You will have Xtest, ytest in your environment
tensors = matio.load('spamTest.mat', { 'Xtest', 'ytest' })
local Xtest = tensors['Xtest']
local ytest = tensors['ytest']

print('Evaluating the trained Linear SVM on a test set ...')

p = svmPredict(model, Xtest)

print('Test Accuracy: ', p:eq(ytest):sum() / ytest:size(1) * 100)

pause()

-- ================= Part 5: Top Predictors of Spam ====================
--  Since the model we are training is a linear SVM, we can inspect the
--  weights learned by the model to understand better how it is determining
--  whether an email is spam or not. The following code finds the words with
--  the highest weights in the classifier. Informally, the classifier
--  'thinks' that these words are the most likely indicators of spam.

-- Sort the weights and obtin the vocabulary list
--local model = torch.load("spam_model.sav")
local weight, idx = torch.sort(model.w, 1, true)
local vocabList = getVocabList()

print('Top predictors of spam: ')
for i = 1, 20 do
    print(vocabList[idx[i][1]], "\t", weight[i][1])
end

pause()

-- =================== Part 6: Try Your Own Emails =====================
--  Now that you've trained the spam classifier, you can use it on your own
--  emails! In the starter code, we have included spamSample1.txt,
--  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
--  The following code reads in one of these emails and then uses your
--  learned SVM classifier to determine whether the email is Spam or
--  Not Spam

-- Set the file to be read in (change this to spamSample2.txt,
-- emailSample1.txt or emailSample2.txt to see different predictions on
-- different emails types). Try your own emails as well!
local filename = 'spamSample1.txt'

-- Read and predict
file_contents = readFile(filename)
word_indices = processEmail(file_contents)
local x = emailFeatures(word_indices)
p = svmPredict(model, x)

print('Processed ', filename)
print('Spam Classification:', p[1][1])
print('(1 indicates spam, 0 indicates not spam)')