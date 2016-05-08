local torch = require 'torch'
local image = require 'image'


local train = torch.load("train_32x32.t7", 'ascii')

print (train)

local batch = train.data:narrow(1, 1, 10)
local labels = train.labels:narrow(1, 1, 10)

print(labels)
image.display(batch)
