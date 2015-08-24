--[[ This script extracts images to tensors in .t7 files.
--  The images will be sorted by classes, 1 file per class
--
--]]

-- Load packages 
local image = require('image')
local csv = require('csv')
local path = require('pl.path')
local dir = require('pl.dir')
local comp = require 'pl.comprehension' . new()
require 'nn'
require 'cunn'
-- Paths
local data_path = './train/GTSRB/Final_Training/Images/'
local img_path = '00000'
local meta_file = 'GT-00000.csv'

-- Functions
--
local function shuffle(t)
  local n = #t
  while n >= 2 do
    --n is now the last pertinent index
    local k = math.random(n) -- 1 <= k <= n
    -- Quick swap
    t[n], t[k] = t[k], t[n]
    n = n - 1
  end
  return t
end

local function shuffleTensors(t, labels)
  local n = t:size(1)
  local swap_ex = torch.Tensor(t[{1,{}}]:size())
  local swap_lab = 1
  while n >= 2 do
    --n is now the last pertinent index
    local k = math.random(n) -- 1 <= k <= n
    -- Quick swap
    swap_ex:copy(t[{k,{}}])
    t[{k, {}}]:copy(t[{n,{}}])
    t[{n, {}}]:copy(swap_ex)
    swap_lab = labels[k]
    labels[k] = labels[n]
    labels[n] = swap_lab
    n = n - 1
  end
  return t
end

local function csvToTable(meta_file)
  -- Returns a table (dict) of the metadata indexed by filenames
  -----------------------------------------
  -- meta_file (string)
  --      relative path to the file
  local f = csv.open(meta_file, {separator=";"})
  meta_data = {}
  for fields in f:lines() do
    -- Remove header
    if fields[1] ~= "Filename" then
      meta_data[fields[1]] = {
      width=fields[2],
      height=fields[3],
      x1=fields[4],
      y1=fields[5],
      x2=fields[6],
      y2=fields[7]
      }

      if fields[8] then
        meta_data[fields[1]].classId = tonumber(fields[8])
      end
    end
  end
  return meta_data
end



local function imagesToTensorFiles(origDir, destDir, size, augment_factor, prep)
  -- extract all the examples and write them per class
  -- The t7 files will be written at destDir
  ------------------------------------
  -- origDir (string)
  --      relative path to the directory containing the image files
  -- destDir (string)
  --      relative path to the destination directory, will contain the t7 files
  -- size (int) / OPTIONAL - default=32
  --      the size to which the images are rescaled

  local size = size or 32
  local augment_factor = augment_factor or 0

  for root, dirs, files in dir.walk(origDir) do
    -- Only check one level of recursion for now

    if root == origDir then
      if (table.getn(dirs) ~= 0) then
        for i,f in ipairs(dirs) do
          io.write(f .. ', ')
        end
        io.write('\n')
      end
      if (table.getn(files) ~= 0) then
        for i,f in ipairs(files) do
          io.write(f .. ', ')
        end
        io.write('\n')
      end
    else
      -- Here we will process a given class
      local class_ = path.splitext(path.basename(root))
      local meta_file = 'GT-' .. class_ .. '.csv'
      local meta_file = path.join(path.join(origDir, class_), meta_file)
      local meta_data = csvToTable(meta_file)
      local dataset = {}
      -- We need the classes to be 1-indexed
      local c = tonumber(class_) + 1
      io.write("Extracting class " .. c)
      -- the current index in the examples table
      local c_i = 1

      for i, f in ipairs(files) do
        local file_  = path.join(root, f)
        -- load image
        local name_, ext = path.splitext(file_)
        if ext == '.ppm' then
          local img = image.load(file_)
          local img_ = image.scale(img, size..'x'..size)
          img_ = image.rgb2yuv(img_)
          dataset[c_i] = torch.Tensor(3, size, size):copy(img_)
          c_i = c_i + 1
          -- Data augmentation
          if augment_factor ~= 0 then
            for j=1,augment_factor do
              img_ = image.rotate(img, torch.uniform(-15,15)*math.pi/180)
              local tr_step_x = torch.uniform(-2, 2)
              local tr_step_y = torch.uniform(-2, 2)
              img_ = image.translate(img_, tr_step_x, tr_step_y)
              local size_ = img_:size()
              local sc_factor = torch.uniform(0, 0.4)
              local border = {x=sc_factor*size_[2],
                              y=sc_factor*size_[3]}
              local start = {x=math.floor(math.max(border.x/2, 1)),
                             y=math.floor(math.max(border.y/2, 1))}
              local e = {y=math.floor(size_[2] - border.x/2),
                           x=math.floor(size_[3] - border.y/2)}
              img_ = image.crop(img_, start.x, start.y,
                                      e.x, e.y)
              img_ = image.scale(img_, size..'x'..size)
              img_ = image.rgb2yuv(img_)
              dataset[c_i] = torch.Tensor(3, size, size):copy(img_)
              c_i = c_i + 1
            end
          end
              local img_ = image.rgb2yuv(img_)
        end
      end
      local filename = path.join(destDir, "class_"..c..".t7")
      torch.save(filename, dataset)
      io.write('   ' .. filename .. " saved \n")
    end
  end
  return dataset
end

local function datasetFromClasses(origDir, ex_per_class, classes)
  -- Extract a given number of examples in the selected classes to create a dataset object.
  -- The dataset will be saved to a .t7 file. Examples are shuffled.
  ---------------------------------------------------
  -- classes (table OPTIONAL)
  --      array of the selected classes
  -- ex_per_class (int OPTIONAL)
  --      number of examples to be extracted from each class
  -- origDir (string)
  --      path to the directory containing the examples as tensors, 1 file per class
  local classes = classes or comp 'x for x=1, 43' ()
  local nb_classes = #classes
  local ex_per_class = ex_per_class or math.huge
  local dataset_ = {}
  local nb_total = 0

  for i_c, c in ipairs(classes) do
    local examples = torch.load(path.join(origDir,  "class_"..c..".t7"))
    nb_total = nb_total + #examples
  end

  local dataset = {}
  dataset.data = torch.Tensor(nb_total, 3, 32, 32)
  dataset.labels = torch.Tensor(nb_total)

  local current_i = 1
  for i_c, c in ipairs(classes) do
    xlua.progress(i_c, 43)
    --open file and take the first *ex_per_class* examples
    local examples = torch.load(path.join(origDir,  "class_"..c..".t7"))
    local nb = math.min(#examples, ex_per_class)
    for i=1, nb do
      dataset.data[{current_i, {}}]:copy(examples[i])
      dataset.labels[current_i] = c
      current_i = current_i + 1 
    end
  end
  -- Shuffle the dataset
  print('Shuffling the dataset')
  shuffleTensors(dataset.data, dataset.labels)


  return dataset, nb_classes, ex_per_class
end


local function formatTestData(test_orig, size, meta_test)
  -- get the test data in a correct dataset object
  local dataset_ = {}
  local metadata = csvToTable(meta_test)
  for root, dirs, files in dir.walk(test_orig) do
    local c_i = 1
    for i, f in ipairs(files) do
      local file_  = path.join(root, f)
      local name_, ext = path.splitext(file_)
      if ext == '.ppm' then
        dataset_[c_i] = {} 
        local img = image.load(file_)
        img = image.scale(img, size..'x'..size)
        img = image.rgb2yuv(img)
        dataset_[c_i][1] = torch.Tensor(3, size, size):copy(img)
        dataset_[c_i][2] = metadata[f].classId + 1
        c_i = c_i + 1
      end
    end     
  end

  local dataset = {}
  dataset.data = torch.Tensor(#dataset_, dataset_[1][1]:size(1),
                                         dataset_[1][1]:size(2),
                                         dataset_[1][1]:size(3))
  dataset.labels = torch.Tensor(#dataset_)
  for i, c in ipairs(dataset_) do
    dataset.data[{i,{}}] = c[1]
    dataset.labels[i] = c[2]
  end

  return dataset 
end

function preprocess(dataset)
      print('Preprocessing the data')
      --Global norm
      print('\t Global normalization')
      local mean_ =  dataset.data:mean()
      local std_ = dataset.data:std()
      for i=1, dataset.data:size(1) do
        dataset.data[{i, {}}] = dataset.data[{i, {}}] - mean_
        dataset.data[{i, {}}] = dataset.data[{i, {}}] / std_
      end
      print('Done')
      -- Local norm
      print('\t Local normalization')
      local indices = torch.range(1,
          dataset.data:size(1)):long():split(100)
      local inputs = torch.CudaTensor(100, 3, 32, 32)
      local outputs = torch.CudaTensor(100, 3, 32, 32)
      local mlp = nn.Sequential()
      mlp:add(nn.SpatialContrastiveNormalization(3))
      mlp:cuda()
      for i=1, #indices do
        if i==#indices then
            inputs = torch.CudaTensor(indices[i]:size(1), 3, 32, 32)
            outputs = torch.CudaTensor(indices[i]:size(1), 3, 32, 32)
        end
        inputs:copy(dataset.data:index(1,indices[i]))
        outputs = mlp:forward(inputs)
        dataset.data:index(1, indices[i]):copy(outputs)
      end

    return dataset
end


return {
  csvToTable = csvToTable,
  imagesToTensorFiles = imagesToTensorFiles,
  datasetFromClasses = datasetFromClasses,
  formatTestData = formatTestData,
  preprocess = preprocess
}
