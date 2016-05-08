local torch = require 'torch'
local nn = require 'nn'
local optim = require 'optim'

local xlua = require 'xlua'

require 'cutorch'
require 'cunn'


local train = torch.load("train_32x32.t7", 'ascii')
local test = torch.load("test_32x32.t7", 'ascii')



local model = torch.load("model.t7")
local criterion = nn.ClassNLLCriterion():cuda()

print(test, train, model, criterion)

local batchSize = 256
local trainSize = 32768

local classes = {0,1,2,3,4,5,6,7,8,9}
local learningRate = 0.1


local function randomSubset(size, dataset)
  local indexes = torch.LongTensor(size):random(dataset.data:size(1))
  return {
    data = dataset.data:index(1, indexes),
    labels =  dataset.labels:index(1, indexes)
  }
end


local function nextBatch(n, dataset)
  local size = (batchSize + n < dataset.data:size(1) and batchSize) or dataset.data:size(1) - n

  local batch = dataset.data:narrow(1, n + 1, size):cuda()
  local labels = dataset.labels:narrow(1, n + 1, size):cuda()

  return batch, labels
end

local function runTraining(dataset)

  local parameters, gradParameters = model:getParameters()

  local n = 0
  local cost = 0

  local confusion = optim.ConfusionMatrix(classes)

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

  return cost / trainSize, confusion
end


local function runTests(dataset)

  local confusion = optim.ConfusionMatrix(classes)

  local n = 0
  local cost = 0
  local testSize = dataset.data:size(1)

  while(n < testSize) do
    local batch, labels = nextBatch(n, dataset)
    local output = model:forward(batch)

    cost = cost + criterion:forward(output, labels)
    confusion:batchAdd(output, labels)

    xlua.progress(n, testSize)
    n = n + batchSize

    collectgarbage()
  end

  xlua.progress(testSize, testSize)
  return cost  / testSize, confusion
end





local epoch = 1


while(true) do

  print ("training epoch", epoch)
  local next = randomSubset(trainSize, train)

  model:training()
  local trainCost, trainConfusion = runTraining(next, model, criterion)
  print(trainConfusion)

  print (string.format("done training, error: %f", trainCost))

  model:evaluate()
  local testCost, testConfusion = runTests(test, model, criterion)
  print(testConfusion)

  print (string.format("done testing, error: %f", testCost))

  torch.save("model.t7", model)


  epoch = epoch + 1

end
