
local torch = require 'torch'
local nn = require 'nn'

require 'cunn'

local features = {6, 16, 120}
local inputs = 1
local outputs = 10

local model = nn.Sequential()

model:add(nn.SpatialConvolutionMM(inputs, features[1], 5, 5)) -- output size: 29x29
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- output size 14x14

-- stage 2 : filter bank -> non linear -> max pooling
model:add(nn.SpatialConvolutionMM(features[1], features[2], 5, 5)) -- output size: 10x10
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 5x5

model:add(nn.SpatialConvolutionMM(features[2], features[3], 5, 5)) -- output size: 1x1 (linear layer)
model:add(nn.ReLU(true))

model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolutionMM(features[3], outputs, 1, 1)) -- output size: 128x1x1
model:add(nn.ReLU(true))

model:add(nn.View(outputs))
model:add(nn.LogSoftMax())

model:cuda()

print(model)
torch.save("model.t7", model)
