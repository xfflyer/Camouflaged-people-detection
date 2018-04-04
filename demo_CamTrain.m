function fcnTrain(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'E:\FCNCam\FCN_CamTrainData\' ;
opts.dataDir = 'E:\FCNCam\FCN_CamTrainData\' ;
opts.modelType = 'fcn8s' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'cam_imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'cam_imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;

opts.numFetchThreads = 1 ; % not used yet

% training options (SGD)
opts.train = struct() ;
[opts, varargin] = vl_argparse(opts, varargin) ;

trainOpts.batchSize = 80 ;
trainOpts.numSubBatches = 10 ;
trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.prefetch = true ;
trainOpts.expDir = opts.expDir ;
trainOpts.learningRate = 0.0001 * ones(1,50) ;
trainOpts.numEpochs = numel(trainOpts.learningRate) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% segmentations
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = vocSetup('dataDir', opts.dataDir, ...
    'edition', opts.vocEdition, ...
    'includeTest', false, ...
    'includeSegmentation', true, ...
    'includeDetection', false) ;
  if opts.vocAdditionalSegmentations
    imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir) ;
  end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
% Get training and test/validation subsets
train = find(imdb.images.set == 1) ;%%训练图像的标号
val = find(imdb.images.set == 2) ;%%标定图像的标号
% Get dataset statistics
if exist(opts.imdbStatsPath)
  stats = load(opts.imdbStatsPath) ;
else
  stats = getDatasetStatistics(imdb) ;
  save(opts.imdbStatsPath, '-struct', 'stats') ;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Get initial model from VGG-VD-16

net =fcnInitializeModel_3Conv_Dense('sourceModelPath', opts.sourceModelPath);



% net = sal_fcnInitializeModel('sourceModelPath', opts.sourceModelPath) ;%%%自己编的
% if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
%   % upgrade model to FCN16s
%   net = sal_fcnInitializeModel16s(net) ;
% end
% if strcmp(opts.modelType, 'fcn8s')
%   % upgrade model fto FCN8s
%   net = sal_fcnInitializeModel8s(net) ;
% end

net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,21,'single') ;
bopts.rgbMean = stats.rgbMean ;
%bopts.useGpu = numel(opts.train.gpus) > 0 ;
bopts.useGpu = [1] ;
% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), ...
                     trainOpts, ....
                     'train', train, ...
                     'val', val, ...
                     opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
