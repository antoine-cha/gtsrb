require 'nn'
require 'optim'


-- define the dataset
function getDataset(file, yesNo)

    dataset = torch.load(file)
    -- Modify dataset to get Tensors as classes
    if yesNo then
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
    function dataset:size() return table.getn(dataset) end
    return dataset
end


-- define the network
function createNetwork()
    -- For testing purpose
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

    --Equivalent to a fully connected one
    --This removes the need for a reshape

    mlp:add(conv1)
    mlp:add(nn.ReLU())
    mlp:add(subnor1)
    mlp:add(divnor1)
    mlp:add(max1)
    -- Branching 
    mlp:add(ways)

    mlp:add(nn.Reshape(100))

    mlp:add(nn.Linear(100,3))
    mlp:add(nn.SoftMax())
    print(mlp:forward(x):size())

    return mlp
end

local yesNo = false
dataset = getDataset('1k3classes.t7', yesNo)
mlp = createNetwork()
---- Training Criterion
if yesNo then
    criterion = nn.MSECriterion()
else
    criterion = nn.CrossEntropyCriterion()
end

stoc = false
if stoc then
    trainer = nn.StochasticGradient(mlp, criterion)
    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer:train(dataset)
else
    f, df = mlp:getParameters()
    print(f:size())
    -- Define the function to be optimized for SGD
    --
    feval = function(f_new)
        -- set x to x_new, if differnt
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
        -- reset gradients (gradients are always accumulated, to accomodate 
        -- batch methods)
        df:zero()
        --evaluate the loss function and its derivative wrt x,
        local loss_x = criterion:forward(mlp:forward(inputs), target)
        mlp:backward(inputs, criterion:backward(mlp.output, target))
        -- return loss(x) and dloss/dx
        return loss_x, df
    end

    sgd_params = {
        learningRate = 1e-3,
        learningRateDecay = 1e-4,
        weightDecay = 0,
        momentum = 0
        }
    -- Start training
    for i=1,1e4 do
        current_loss = 0
        for i= 1,128 do
            _, fs = optim.sgd(feval, f, sgd_params)
            current_loss = current_loss + fs[1]
        end
        current_loss = current_loss / 100
        print(i ..'th iteration, current_loss :' .. current_loss)
    end
end





