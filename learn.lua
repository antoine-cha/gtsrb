require 'nn'
require 'optim'
require 'cunn'

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


local function createNetwork(nb_classes)
  -- Define the network as in the paper
  local nb_classes = nb_classes or 43
  -- Enables to test the output sizes
  local x = torch.Tensor(3,32,32)

  local mlp = nn.Sequential()
  kW = 3; kH=3;
  local conv1 = nn.SpatialConvolutionMM(3, 108, kW, kH)
  local subnor1 = nn.SpatialSubtractiveNormalization(108)
  local divnor1 = nn.SpatialDivisiveNormalization(108)
  local max1 = nn.SpatialMaxPooling(4, 4, 4, 4)
  -- Branching now
  local ways = nn.Concat(1)
  -- First branch with 2 layers of convolutions
  local way1 = nn.Sequential()
  local conv2 = nn.SpatialConvolutionMM(108, 108, kW, kH)
  local subnor2 = nn.SpatialSubtractiveNormalization(108)
  local divnor2 = nn.SpatialDivisiveNormalization(108)
  local max2 = nn.SpatialMaxPooling(4, 4, 4, 4)
  local conv3 = nn.SpatialConvolutionMM(108, 50, 1, 1) 
  way1:add(conv2)
  way1:add(nn.ReLU())
  way1:add(subnor2)
  way1:add(divnor2)
  way1:add(max2)
  way1:add(conv3)
  way1:add(nn.ReLU())

  local way2 = nn.Sequential()
  way2:add(nn.SpatialConvolution(108,50,7,7))
  way2:add(nn.Reshape(50,1)) 
  ways:add(way1)
  ways:add(way2)


  mlp:add(conv1)
  mlp:add(nn.ReLU())
  mlp:add(subnor1)
  mlp:add(divnor1)
  mlp:add(max1)
  -- Branching 
  mlp:add(ways)

  mlp:add(nn.Reshape(100))

  mlp:add(nn.Linear(100,nb_classes))
  mlp:add(nn.SoftMax())
  print('Network outputs :')
  print(mlp:forward(x):size())
  
  return mlp
end



local function trainNetwork(network, dataset_file, params)
  -- Train a network given as input, with the given dataset and batchSize
  -- --------------------------
  -- Returns nothing :
  --  the network object is changed by the function
  -- --------------------------
  -- network (mlp)
  --  network to be trained
  -- dataset_file (path)
  --  relative path to the dataset file
  -- only2classes(bool)
  --  specifies the task to format the dataset
  -- batchSize(int)
  local dataset = getDataset(dataset_file, params.only2classes)
  local cudaOn = params.cudaOn

  if params.cudaOn then
    network:cuda()
  end
  ---- Training Criterion defined by the task
  if params.only2classes then
    criterion = nn.MSECriterion()
    print('MSE Criterion')
  else
    criterion = nn.CrossEntropyCriterion()
    if cudaOn then
      criterion:cuda()
    end
    print('Cross-entropy Criterion')
  end

  if params.batchSize == 1 then
    local trainer = nn.StochasticGradient(network, criterion)
    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    print('Training using normal stochastic gradient')
    trainer:train(dataset)
  else
    f, df = network:getParameters()
    -- Define the function to be optimized for SGD
    _nidx_ = 1209 -- 9*128
    feval = function(f_new)
      -- set x to x_new, if different
      -- (in this simple example, x_new will typically always point to x,
      -- so the copy is really useless)
      if f ~= f_new then
        f:copy(f_new)
      end
      -- select a new training sample
      _nidx_ = (_nidx_ or 0) + 1
      if _nidx_ > (#dataset) then _nidx_ = 1 end
      local sample = dataset[_nidx_]
      local target = sample[2]
      local inputs = sample[1]
      if cudaOn then
        inputs:cuda()
        --target:cuda()
      end
      

      -- reset gradients (gradients are always accumulated, to accomodate 
      -- batch methods)
      df:zero()
      --evaluate the loss function and its derivative wrt x,
      local loss_x = criterion:forward(network:forward(inputs), target)
      network:backward(inputs, criterion:backward(network.output, target))
      -- return loss(x) and dloss/dx
      return loss_x, df
    end


    -- Start training
    for i=1, params.nb_epochs do
      current_loss = 0
      for i= 1,params.batchSize do
        _, fs = optim.sgd(feval, f, params)
        current_loss = current_loss + fs[1]
      end
      current_loss = current_loss / 100
      print(i ..'th iteration, current_loss :' .. current_loss)
    end
  end
end

return {getDataset = getDataset,
        createNetwork = createNetwork,
        trainNetwork = trainNetwork
      }
