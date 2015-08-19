require 'optim'
require 'nn'
require 'image'
require 'cunn'
require 'cutorch'

local comp = require 'pl.comprehension' . new()

-- Get a network and test it 
function show_conf(network, dataset_file, cudaOn)
  -- Show the confusion matrix

  local dataset = torch.load(dataset_file)
  conf = optim.ConfusionMatrix(43)
  conf:zero()
  local N = dataset.data:size(1)
  if not cudaOn then
    network:float()
  end

  for i=1, N do
    if i < 100 then
      print(dataset.data:index(1,i))
      local pred = network:forward(dataset.data:index(1,i))
      conf:add(pred, dataset.labels[i])
    end
  end
  --print(conf)
  image.display(conf:render('score', false,20))  
end


local model_file = './models/model-mom2-l5-3.t7'
local dataset_file = '../gtsrb/test/test_data.t7'
local cudaOn = true
local network = torch.load(model_file)

print('Compute the confusion matrix')
show_conf(network, dataset_file)

