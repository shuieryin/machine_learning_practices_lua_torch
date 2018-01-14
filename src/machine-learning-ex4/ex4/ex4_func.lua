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

function sigmoidGradient(z)
    --SIGMOIDGRADIENT returns the gradient of the sigmoid function
    --evaluated at z
    --   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    --   evaluated at z. This should work regardless if z is a matrix or a
    --   vector. In particular, if z is a vector or matrix, you should return
    --   the gradient for each element.

    local g = torch.zeros(z:size())

    -- ====================== YOUR CODE HERE ======================
    -- Instructions: Compute the gradient of the sigmoid function evaluated at
    --               each value of z (z can be a matrix, vector or scalar).

    local sigmoidZ = sigmoid(z)
    g = torch.cmul(sigmoidZ, 1 - sigmoidZ)

    -- =============================================================
    return g
end

function displayData(X, example_width)
    --DISPLAYDATA Display 2D data in a nice grid
    --   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    --   stored in X in a nice grid. It returns the figure handle h and the
    --   displayed array if requested.
    local m = X:size(1)
    local n = X:size(2)

    -- Set example_width automatically if not passed in
    if type(example_width) ~= "table" or #example_width == 0 then
        example_width = math.floor(math.sqrt(n))
    end

    -- Gray Image
    --colormap(gray)

    -- Compute rows, cols
    local example_height = (n / example_width)

    -- Compute number of items to display
    local display_rows = math.floor(math.sqrt(m))
    local display_cols = math.ceil(m / display_rows)

    -- Between images padding
    local pad = 1

    -- Setup blank display
    local display_array = -torch.ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad))

    -- Copy each example into a patch on the display array
    local curr_ex = 1
    local max_val
    for j = 1, display_rows do
        for i = 1, display_cols do
            if curr_ex > m then
                break
            end
            -- Copy the patch

            -- Get the max value of the patch
            local curData = X[{ curr_ex, {} }]
            max_val = torch.max(torch.abs(curData))
            local targetM = pad + (j - 1) * (example_height + pad) + torch.range(1, example_height)
            local targetN = pad + (i - 1) * (example_width + pad) + torch.range(1, example_width)
            local reshaped = torch.reshape(curData, example_height, example_width) / max_val
            display_array[{ { targetM[1], targetM[#targetM] }, { targetN[1], targetN[#targetN] } }] = reshaped:t()
            curr_ex = curr_ex + 1
        end
        if curr_ex > m then
            break
        end
    end

    return display_array
end

y_labels_map = {}
function nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
    --NNCOSTFUNCTION Implements the neural network cost function for a two layer
    --neural network which performs classification
    --   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    --   X, y, lambda) computes the cost and gradient of the neural network. The
    --   parameters for the neural network are "unrolled" into the vector
    --   nn_params and need to be converted back into the weight matrices.
    --
    --   The returned parameter grad should be a "unrolled" vector of the
    --   partial derivatives of the neural network.

    -- Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    -- for our 2 layer neural network
    local theta1VecSize = hidden_layer_size * (input_layer_size + 1)
    local Theta1 = torch.reshape(nn_params[{ { 1, theta1VecSize } }], hidden_layer_size, input_layer_size + 1)
    local Theta2 = torch.reshape(nn_params[{ { theta1VecSize + 1, nn_params:numel() } }], num_labels, hidden_layer_size + 1)

    -- Setup some useful variables
    local m = X:size(1)

    -- You need to return the following variables correctly
    local J = 0
    local Theta1_grad = torch.zeros(Theta1:size())
    local Theta2_grad = torch.zeros(Theta2:size())

    -- ====================== YOUR CODE HERE ======================
    -- Instructions: You should complete the code by working through the
    --               following parts.
    --
    -- Part 1: Feedforward the neural network and return the cost in the
    --         variable J. After implementing Part 1, you can verify that your
    --         cost function computation is correct by verifying the cost
    --         computed in ex4.m
    --
    -- Part 2: Implement the backpropagation algorithm to compute the gradients
    --         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    --         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    --         Theta2_grad, respectively. After implementing Part 2, you can check
    --         that your implementation is correct by running checkNNGradients
    --
    --         Note: The vector y passed into the function is a vector of labels
    --               containing values from 1..K. You need to map this vector into a
    --               binary vector of 1's and 0's to be used with the neural network
    --               cost function.
    --
    --         Hint: We recommend implementing backpropagation using a for-loop
    --               over the training examples if you are implementing it for the
    --               first time.
    --
    -- Part 3: Implement regularization with the cost function and gradients.
    --
    --         Hint: You can implement this around the code for
    --               backpropagation. That is, you can compute the gradients for
    --               the regularization separately and then add them to Theta1_grad
    --               and Theta2_grad from Part 2.

    X = torch.ones(m, 1):cat(X, 2)
    local y_labels = y_labels_map[num_labels]
    if y_labels == nil then
        y_labels = {}
        y_labels_map[num_labels] = y_labels
    end
    for i = 1, m do

        --FEEDFORWARD

        -- X is 5000x401 going through 1 pic at a time
        local a1 = X[i]
        -- theta1 is 25x401 a1 is 1x401
        local z2 = Theta1 * a1 -- no need to tranpose a1 since X[i] already did
        -- a2 is sigmoid of z2 and add the ones for the next layer
        local a2 = sigmoid(z2)
        a2 = torch.ones(1):cat(a2, 1)
        -- calc z3 for the output layer
        local z3 = Theta2 * a2
        -- a3 (output) is sigmoid z3
        local a3 = sigmoid(z3)

        -- for the given i change y into a vector that is zero
        -- everywhere except its the index of its correct labels
        local pos = y[i]
        if type(pos) == "userdata" then
            pos = pos[1]
        end
        local y_label = y_labels[pos]
        if y_label == nil then
            y_label = torch.zeros(num_labels, 1)
            y_label[pos] = 1
            y_labels[pos] = y_label
        end

        --BACKPROPAGATION
        --for each output unit k in layer 3 set d3=(a3 - y)
        local d3 = a3 - y_label

        --For the hidden layer l=2 set d2 = t2:t()*d3.*sigmoidGrad(z2)
        z2 = torch.ones(1):cat(z2, 1)
        local d2 = torch.cmul(Theta2:t() * d3, sigmoidGradient(z2))
        d2 = d2[{ { 2, d2:size(1) } }]

        --accumulate the gradient from this example
        Theta1_grad = Theta1_grad + torch.mm(torch.reshape(d2, d2:size(1), 1), torch.reshape(a1, 1, a1:size(1)))
        Theta2_grad = Theta2_grad + torch.mm(torch.reshape(d3, d3:size(1), 1), torch.reshape(a2, 1, a2:size(1)))

        local cost = 0
        -- another for loop to add up cost by labels
        for k = 1, num_labels do
            local costk = -1 / m * (y_label[k] * torch.log(a3[k]) + (1 - y_label[k]) * torch.log(1 - a3[k]))
            cost = cost + costk[1]
        end

        --add the cost form that observation to the rest of the cost
        J = J + cost
    end

    local Theta1_ori = Theta1[{ {}, { 2, Theta1:size(2) } }]
    local Theta1_reg = Theta1_ori
    Theta1_reg = torch.pow(Theta1_reg, 2)
    local T1_regsum = Theta1_reg:sum()

    local Theta2_ori = Theta2[{ {}, { 2, Theta2:size(2) } }]
    local Theta2_reg = Theta2_ori
    Theta2_reg = torch.pow(Theta2_reg, 2)
    local T2_regsum = Theta2_reg:sum()

    --add the two thetas assuming we have one hidden layer
    --but the program works for any size theta
    local Reg = lambda / (2 * m) * (T2_regsum + T1_regsum)

    --add the reg term to the rest of our cost
    J = J + Reg

    --gradients
    Theta1_grad = 1 / m * Theta1_grad + lambda / m * torch.zeros(Theta1:size(1), 1):cat(Theta1_ori, 2)
    Theta2_grad = 1 / m * Theta2_grad + lambda / m * torch.zeros(Theta2:size(1), 1):cat(Theta2_ori, 2)

    -- =========================================================================

    -- Unroll gradients
    local grad = torch.reshape(Theta1_grad, Theta1_grad:numel(), 1):cat(torch.reshape(Theta2_grad, Theta2_grad:numel(), 1), 1)

    return J, grad
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

local function debugInitializeWeights(fan_out, fan_in)
    --DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    --incoming connections and fan_out outgoing connections using a fixed
    --strategy, this will help you later in debugging
    --   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
    --   of a layer with fan_in incoming connections and fan_out outgoing
    --   connections using a fix set of values
    --   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    --   the first row of W handles the "bias" terms

    -- Set W to zeros
    local W = torch.zeros(fan_out, 1 + fan_in)

    -- Initialize W using "sin", this ensures that W is always of the same
    -- values and will be useful for debugging
    return torch.reshape(torch.range(1, W:numel()):sin(), W:size()) / 10
end

local function computeNumericalGradient(J, theta)
    --COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    --and gives us a numerical estimate of the gradient.
    --   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    --   gradient of the function J around theta. Calling y = J(theta) should
    --   return the function value at theta.

    -- Notes: The following code implements numerical gradient checking, and
    --        returns the numerical gradient.It sets numgrad(i) to (a numerical
    --        approximation of) the partial derivative of J with respect to the
    --        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
    --        be the (approximately) the partial derivative of J with respect
    --        to theta(i).)

    local numgrad = torch.zeros(theta:size())
    local perturb = torch.zeros(theta:size())
    local e = 1e-6
    for p = 1, theta:numel() do
        -- Set perturbation vector
        perturb[p] = e
        local loss1 = J(theta - perturb)
        local loss2 = J(theta + perturb)
        -- Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    end
    return numgrad
end

function checkNNGradients(lambda)
    --CHECKNNGRADIENTS Creates a small neural network to check the
    --backpropagation gradients
    --   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
    --   backpropagation gradients, it will output the analytical gradients
    --   produced by your backprop code and the numerical gradients (computed
    --   using computeNumericalGradient). These two gradient computations should
    --   result in very similar values.

    lambda = lambda or 0

    local input_layer_size = 3
    local hidden_layer_size = 5
    local num_labels = 3
    local m = 5

    -- We generate some 'random' test data
    local Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    local Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    -- Reusing debugInitializeWeights to generate X
    local X = debugInitializeWeights(m, input_layer_size - 1)
    local y = 1 + torch.range(1, m):mod(num_labels)

    -- Unroll parameters
    local nn_params = torch.reshape(Theta1, Theta1:numel(), 1):cat(torch.reshape(Theta2, Theta2:numel(), 1), 1)

    -- Short hand for cost function
    local costFunc = function(p)
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
    end
    print('')

    local cost, grad = costFunc(nn_params)
    local numgrad = computeNumericalGradient(costFunc, nn_params)

    -- Visually examine the two gradient computations.  The two columns
    -- you get should be very similar.
    print(numgrad:cat(grad, 2))
    print("The above two columns you get should be very similar.")
    print("Left-Your Numerical Gradient, Right-Analytical Gradient)")

    -- Evaluate the norm of the difference between two solutions.
    -- If you have a correct implementation, and assuming you used EPSILON = 0.0001
    -- in computeNumericalGradient.m, then diff below should be less than 1e-9
    local diff = torch.norm(numgrad - grad) / torch.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then ')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference:')
    print(diff)
end

function predict(Theta1, Theta2, X)
    --PREDICT Predict the label of an input given a trained neural network
    --   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    --   trained weights of a neural network (Theta1, Theta2)

    -- Useful values
    local m = X:size(1)

    -- Instructions: Complete the following code to make predictions using
    --               your learned neural network. You should set p to a
    --               vector containing labels between 1 to num_labels.
    --
    -- Hint: The max function might come in useful. In particular, the max
    --       function can also return the index of the max element, for more
    --       information see 'help max'. If your examples are in rows, then, you
    --       can use max(A, [], 2) to obtain the max for each row.

    X = torch.ones(m, 1):cat(X, 2)

    local predict = sigmoid(X * Theta1:t())
    predict = torch.ones(predict:size(1)):cat(predict, 2)
    predict = sigmoid(predict * Theta2:t())
    local predict_max, p = predict:max(2)

    return p
end

function randInitializeWeights(L_in, L_out)
    --RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    --incoming connections and L_out outgoing connections
    --   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
    --   of a layer with L_in incoming connections and L_out outgoing
    --   connections.
    --
    --   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    --   the first column of W handles the "bias" terms
    -- Instructions: Initialize W randomly so that we break the symmetry while
    --               training the neural network.
    -- Note: The first column of W corresponds to the parameters for the bias unit

    -- You need to return the following variables correctly
    local W = torch.zeros(L_out, 1 + L_in)

    -- ====================== YOUR CODE HERE ======================
    -- Instructions: Initialize W randomly so that we break the symmetry while
    --               training the neural network.
    --
    -- Note: The first column of W corresponds to the parameters for the bias unit

    local epsilon_init = 0.12
    W = torch.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    -- =========================================================================

    return W
end