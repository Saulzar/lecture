local torch = require 'torch'
local nn = require 'nn'
local optim = require 'optim'

local xlua = require 'xlua'

require 'cutorch'
require 'cunn'


local dataset = torch.load("train_32x32.t7", 'ascii')

local model = torch.load("model.t7")
local criterion = nn.ClassNLLCriterion():cuda()

print(dataset, model, criterion)

local batchSize = 256

local classes = {0,1,2,3,4,5,6,7,8,9}
local learningRate = 0.1


local function nextBatch(n)
  local size = (batchSize + n < dataset.data:size(1) and batchSize) or dataset.data:size(1) - n

  local batch = dataset.data:narrow(1, n + 1, size):cuda()
  local labels = dataset.labels:narrow(1, n + 1, size):cuda()

  return batch, labels
end


local parameters, gradParameters = model:getParameters()

local n = 0
local cost = 0

local confusion = optim.ConfusionMatrix(classes)
local trainSize = 16384

while(n < trainSize) do
  local batch, labels = nextBatch(n, dataset)

  gradParameters:zero()

  local output = model:forward(batch)
  confusion:batchAdd(output, labels)

  cost = cost + criterion:forward(output, labels)
  local gradient = criterion:backward(output, labels)

  model:backward(batch, gradient)

  parameters:add(-learningRate, gradParameters)

  xlua.progress(n, trainSize)
  n = n + batchSize

  collectgarbage()
end
xlua.progress(trainSize, trainSize)

print("error:", cost / trainSize)
print(confusion)
torch.save("model.t7", model)
