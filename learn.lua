require 'nn'
require 'optim'
require 'cunn'
require 'sys'

-- define the dataset
local function getDataset(file, only2classes)
  -- Get the dataset object from the file.
  -- Format it if only 2 classes (change to 1, -1)
  -- Return a Tensor 
  -- -------------------------
  -- file (string)
  --  relative path to the t7 file 
  -- only2classes(bool)
  --  specify whether the dataset only contains 2 classes

  local dataset = torch.load(file)
  -- Modify dataset to get Tensors as classes
  if only2classes then
    for i, c in ipairs(dataset) do
      if dataset[i][2] == 2 then
        dataset[i][2] = torch.Tensor(1)
        dataset[i][2][1] = -1
      else
        dataset[i][2] = torch.Tensor(1)
        dataset[i][2][1] = 1
      end
    end
  end
  -- Needed by the algorithm
  function dataset:size() return #table[dataset] end
  return dataset
end


local function trainNetwork(network, params)
  -- Train a network given as input, with the given dataset and batchSize
  -- --------------------------
  -- Returns nothing :
  --  the network object is changed by the function
  -- --------------------------
  -- network (mlp)
  --  network to be trained
  -- params(table)
  --  contains the parameters
  local dataset = getDataset(params.dataPath, params.only2classes)
  dataset.data  = dataset.data:float()
  dataset.labels  = dataset.labels:float()
  local cudaOn = params.cudaOn
 
  ---- Training Criterion defined by the task
  if params.only2classes then
    criterion = nn.MSECriterion()
  else
    criterion = nn.ClassNLLCriterion()
  end
  print(criterion)

  if cudaOn then
    criterion:cuda()
    network:cuda()
  end
  local time = sys.clock()
  local time_0 = sys.clock()
  -- Some layers may have different behaviours during training
  network:training()
  local conf = optim.ConfusionMatrix(43)

  f, df = network:getParameters()
  local base_size_ = dataset.data[1]:size()
  local targets = torch.CudaTensor(params.batchSize)
  local inputs = torch.CudaTensor(params.batchSize, 
                base_size_[1], base_size_[2], base_size_[3])

  local indices = torch.range(1,
      dataset.data:size(1)):long():split(params.batchSize)
  -- remove last so that all batches have same size
  indices[#indices] = nil
  local _ind_ = 1

  -- Define the function to be optimized for SGD
  local feval = function(f_new)

    -- set x to x_new, if different
    -- (in this simple example, x_new will typically always point to x,
    -- so the copy is really useless)
    if f ~= f_new then
      f:copy(f_new)
    end
    -- reset gradients (gradients are always accumulated, to accomodate 
    -- batch methods)
    df:zero()
    --Get the indices of the next batch
    _ind_ = _ind_ + 1 
    if _ind_ == #indices then _ind_ = 1 end
    batch = indices[_ind_]

    inputs:copy(dataset.data:index(1, batch))
    targets:copy(dataset.labels:index(1, batch))

    if cudaOn then
      inputs:cuda()
      targets:cuda()
    end
    local outputs = network:forward(inputs)
    --evaluate the loss function and its derivative wrt x,
    local loss_x = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    network:backward(inputs, df_do)
    conf:batchAdd(outputs, targets)
    -- return loss(x) and dloss/dx
    return loss_x, df
  end


  -- Start training
  local current_loss = 0.0
  local max_batches = params.nb_epochs * #indices
  for i=1, max_batches do
    _, fs = optim.sgd(feval, f, params)
    current_loss = current_loss + fs[1]
    conf:updateValids()
    if i%10 == 0 then
      current_loss = current_loss / params.batchSize
      io.write(('%.2f / ' .. params.nb_epochs.. ' epochs , Average loss :' .. 
                ' %.6f // '):format(i/#indices, current_loss))
      io.write(('Train acc: '..'%.2f '):format(
              conf.totalValid * 100))
      io.write(('t=%.1f s \n'):format(sys.clock() - time))
      time = sys.clock()
      current_loss = 0.0
      conf:zero()
    end
    if i % params.freq_save == 0 then 
      torch.save(params.filename, network)
      torch.save(params.paramsFile, params)
      print('Model saved at ' .. params.filename)
    end
  end
  print(('Total experience time : %.1f s'):format(sys.clock() - time_0))
end

return {getDataset = getDataset,
        trainNetwork = trainNetwork,
      }
