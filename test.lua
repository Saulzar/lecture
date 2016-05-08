local torch = require 'torch'
local optim = require 'optim'

local xlua = require 'xlua'

require 'cutorch'
require 'cunn'

local dataset = torch.load("test_32x32.t7", 'ascii')
local model = torch.load("model.t7")

model = model:cuda()

print(dataset)

local batchSize = 256
local classes = {0,1,2,3,4,5,6,7,8,9}


local function nextBatch(n)
  local size = (batchSize + n < dataset.data:size(1) and batchSize) or dataset.data:size(1) - n

  local batch = dataset.data:narrow(1, n + 1, size):cuda()
  local labels = dataset.labels:narrow(1, n + 1, size):cuda()

  return batch, labels
end


local confusion = optim.ConfusionMatrix(classes)

local n = 0
local testSize = dataset.data:size(1)

while(n < testSize) do
  local batch, labels = nextBatch(n, dataset)
  local output = model:forward(batch)

  confusion:batchAdd(output, labels)

  xlua.progress(n, testSize)
  n = n + batchSize

end

print(confusion)
