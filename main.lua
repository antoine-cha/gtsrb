--Custom imports
local learn = require 'learn'
local path = require 'pl.path'
local dir = require 'pl.dir'

-- training parameters
local modelDir = './models/'
if not path.isdir(modelDir) then
  dir.makepath(modelDir)
end

local filename = 'model-mom2-l5-3.t7'
local dataPath = './dataset-43c-allex.t7'

local params = {
      filename = path.join(modelDir, filename),
      batchSize = 100,
      only2classes = false,
      cudaOn = true,
      nb_epochs = 10000,
      freq_save = 100,
      -- SGD parameters
      learningRate = 1e-3,
      learningRateDecay = 1e-4,
      weightDecay = 0,
      momentum = 0.1,
}


if path.exists(params.filename) then
  local name, ext = path.splitext(params.filename) 
  local n = 1
  while path.exists((name .. '-' .. '%.2i' .. ext):format(n)) do
    print((name .. '%.2i' .. ext):format(n))
    n = n + 1
  end
  params.filename = (name .. '-' .. '%.2i' .. ext):format(n)
end


local network = learn.createNetwork() 

local mlp_ = nn.Sequential()
mlp_:add(network)

print(params)
--print(sgd_paramslearn.trainNetwork(mlp_, dataPath, train_params)
learn.trainNetwork(mlp_, dataPath, params)

torch.save(params.filename, mlp_)
