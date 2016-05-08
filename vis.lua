local torch = require 'torch'
local nn = require 'nn'
local image = require 'image'

local cutorch = require 'cutorch'


local model = torch.load("model.t7")

local modules = model:findModules("nn.SpatialConvolutionMM")
print (modules)

local showWeights = function(m)

  local weights = m.weight:contiguous()
  weights = weights:reshape(m.nOutputPlane, m.nInputPlane, m.kW, m.kH)
  weights = weights:sum(1):squeeze()

  print(weights)

  image.display(weights)
end

showWeights(modules[2])
