require 'optim'
require 'nn'
require 'image'
require 'cunn'
require 'cutorch'
require 'sys'
require 'xlua'
local lapp = require 'pl.lapp'


local args = lapp [[
    Test the given model
    <model> (string) path to the model
]]
local comp = require 'pl.comprehension' . new()
print(args)
if not args.model then
  print('Give the model path as argument')
  os.exit()
end
-- Get a network and test it 
function show_conf(network, dataset_file, batchSize, preprocess)
  -- Show the confusion matrix

  local dataset = torch.load(dataset_file)
  local conf = optim.ConfusionMatrix(43)
  conf:zero()

  local indices = torch.range(1,
      dataset.data:size(1)):long():split(batchSize)
  -- remove last so that all batches have same size
  indices[#indices] = nil
  local base_size_ = dataset.data[1]:size()
  local targets = torch.CudaTensor(batchSize)
  local preds = torch.CudaTensor(batchSize)
  local samples = torch.CudaTensor(batchSize, 
                base_size_[1], base_size_[2], base_size_[3])
  network = network:cuda()

    xlua.progress(i, N)
    samples:copy(dataset.data:index(1,indices[i]))
    -- Do the same preprocessing as training data
    if preprocess then
      samples = samples - samples:mean(1)
    end
    preds = network:forward(samples)
    targets:copy(dataset.labels:index(1, indices[i]))
    conf:batchAdd(preds, targets)
    print(i)
  end

  conf:updateValids()
  io.write(('Test accuracy: '..'%.2f \n'):format(
            conf.totalValid * 100))
end


local dataset_file = '../gtsrb/test/test_data.t7'
local batchSize = 100
local network = torch.load(args.model)
print(torch.type(network))

print('Compute the confusion matrix')
show_conf(network, dataset_file, batchSize)

