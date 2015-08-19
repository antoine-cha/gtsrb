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

-- Paths
local data_path = './train/GTSRB/Final_Training/Images/'
local img_path = '00000'
local meta_file = 'GT-00000.csv'

-- Functions
--
function shuffle(t)
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

function csvToTable(meta_file)
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



function imagesToTensorFiles(origDir, destDir, size)
  -- extract all the examples and write them per class
  -- The t7 files will be written in the current dir
  ------------------------------------
  -- origDir (string)
  --      relative path to the directory containing the image files
  -- destDir (string)
  --      relative path to the destination directory, will contain the t7 files
  -- size (int) / OPTIONAL - default=32
  --      the size to which the images are rescaled

  local size = size or 32

  for root, dirs, files in dir.walk(origDir) do
    -- Only check one level of recursion for now

    if root == origDir then
      print(" Dirs :")
      print(table.getn(dirs))
      if (table.getn(dirs) ~= 0) then
        for i,f in ipairs(dirs) do
          io.write(f .. ', ')
        end
        io.write('\n')
      end
      print(" Files :")
      print(table.getn(files))
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
      print("Extracting class " .. c)
      -- the current index in the examples table
      local c_i = 1

      for i, f in ipairs(files) do
        local file_  = path.join(root, f)
        -- load image
        local name_, ext = path.splitext(file_)
        if ext == '.ppm' then
          local img = image.load(file_)
          img = image.scale(img, size..'x'..size)
          img = image.rgb2yuv(img)
          dataset[c_i] = torch.Tensor(3, size, size):copy(img)
          c_i = c_i + 1
        end
      end
      local filename = path.join(destDir, "class_"..c..".t7")
      torch.save(filename, dataset)
      print(filename .. " saved")
    end
  end
  return dataset
end

function datasetFromClasses(origDir, ex_per_class, classes)
  -- Extract a given number of examples in the selected classes to create a dataset object.
  -- The dataset will be saved to a .t7 file. Examples are shuffled.
  ---------------------------------------------------
  -- classes (table)
  --      array of the selected classes
  -- ex_per_class (int)
  --      number of examples to be extracted from each class
  -- origDir (string)
  --      path to the directory containing the examples as tensors, 1 file per class
  local classes = classes or comp 'x for x=1, 43' ()
  local nb_classes = #classes
  local ex_per_class = ex_per_class or math.huge
  local dataset_ = {}
  local current_i = 1

  for i_c, c in ipairs(classes) do
    --open file and take the first *ex_per_class* examples
    local examples = torch.load(path.join(origDir,  "class_"..c..".t7"))
    print(c, #examples)
    local nb = math.min(#examples, ex_per_class)
    nb = math.max(nb, 0)
    if nb == 0 then
      nb = #table[examples]
    end
    for i=1, nb do
      local ex = {examples[i]}
      if #ex == 0 then
        print(i .. '/' .. nb)
      end
      dataset_[current_i] = {}
      dataset_[current_i][1] = examples[i] 
      dataset_[current_i][2] = c
      current_i = current_i + 1 
    end
  end
  shuffle(dataset_)
  -- Transform dataset into 2 tensors to be sliced
  local dataset = {}
  dataset.data = torch.Tensor(#dataset_, dataset_[1][1]:size(1),
                                         dataset_[1][1]:size(2),
                                         dataset_[1][1]:size(3))
  dataset.labels = torch.Tensor(#dataset_)
  for i, c in ipairs(dataset_) do
    dataset.data[{i,{}}] = c[1]
    dataset.labels[i] = c[2]
  end

  return dataset, nb_classes, ex_per_class
end


function formatTestData(test_orig, size, meta_test)
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
        dataset_[c_i][2] = metadata[f].classId
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
----------------------------------------------------------------------------
----------------------------------------------------------------------------
local train_orig = './train/GTSRB/Final_Training/Images'
local test_orig = './test/GTSRB/Final_Test/Images'

local train_dest = './train/class_files'
local test_dest = './test/test_data.t7'

local meta_test = './gt/GT-final_test.csv'
----------------------------------------------------
----------------------------------------------------
local set = 'train'
-- Extract data from files to Tensors
extraction = false
-- Create the dataset tensor file
creation = true


if not path.isdir(train_dest) then
  dir.makepath(train_dest)
end

print(set)
if set == 'train'then
  if extraction then
    print('Extracting the images')
    dataset = imagesToTensorFiles(train_orig, train_dest, size)
  end
  if creation then
    print('Creating a dataset file')
    dataset, nb_classes, nb_examples = datasetFromClasses(train_dest)
    local nb_ex
  if nb_examples==math.huge then
    nb_ex = 'all'
  end
  torch.save('dataset-'.. nb_classes .. 'c-' .. nb_ex .. 'ex.t7', dataset)
  end
elseif set == 'test' then
  local testdata = formatTestData(test_orig, 32, meta_test)
  torch.save(test_dest, testdata)
end
   
