local path = require('pl.path')
local dir = require('pl.dir')
local data = require 'data'
local lapp = require 'pl.lapp'
local args = lapp [[
    Gets the data in the right format
    -s, --set (default "test")  Set to be processed
    -e, --extraction  (default true) extraction from ppm files ?
    -c, --creation (default true) creation of dataset from t7 files ?
    -d, --destination (default nil) target directory to write
       ]]

local targets = {}
targets.train_orig = './train/GTSRB/Final_Training/Images'
targets.test_orig = './test/GTSRB/Final_Test/Images'
targets.train_dest = args.destination or './train/class_files'
targets.test_dest = args.destination or './test/test_data.t7'
targets.meta_test = './gt/GT-final_test.csv'

local set = args.set
print(targets)

if not path.isdir(train_dest) then
  dir.makepath(train_dest)
end

if set == 'train'then
  print('Working on train data')
  if args.extraction then
    print('Extracting the images')
    dataset = data.imagesToTensorFiles(train_orig, train_dest, size)
  end
  if args.creation then
    print('Creating a dataset file')
    dataset, nb_classes, nb_examples = data.datasetFromClasses(train_dest)
    local nb_ex
  if nb_examples==math.huge then
    nb_ex = 'all'
  end
  torch.save('dataset-'.. nb_classes .. 'c-' .. nb_ex .. 'ex.t7', dataset)
  end
elseif set == 'test' then
  print('Setting up test data')
  local testdata = data.formatTestData(test_orig, 32, meta_test)
  torch.save(test_dest, testdata)
end
