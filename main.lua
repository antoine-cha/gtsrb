require 'nn'
require 'cunn'
require 'cutorch'
--Custom imports
local learn = require 'learn'

-- training parameters
local filename = 'model.t7'
local dataPath = './dataset-43c-allex.t7'
local train_params = {
      network_file = '',
      batchSize = 128,
      only2classes = false,
      cudaOn = true,
      nb_epochs = 1000,
      learningRate = 2e-3,
      learningRateDecay = 1e-4,
      weightDecay = 0,
      momentum = 0.1
}

local network = learn.createNetwork() 

local mlp_ = nn.Sequential()
if train_params.network_file ~= '' then
  mlp_ = torch.load(network_file)
else
  if train_params.cudaOn then
    mlp_:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    mlp_:add(network)
    mlp_:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
  else
    mlp_:add(network)
  end

print(train_params)
--print(sgd_params)
learn.trainNetwork(mlp_, dataPath, train_params)

torch.save(filename, mlp_)
