function net = fcnInitializeModel_3Conv(varargin)
%FCNINITIALIZEMODEL8S Initialize the FCN-8S model from FCN-16S

opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
if ~exist(opts.sourceModelPath)
  fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
  mkdir(fileparts(opts.sourceModelPath)) ;
  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', opts.sourceModelPath) ;
end
net = vl_simplenn_tidy(load(opts.sourceModelPath)) ;

% for convt (deconv) layers, cuDNN seems to be slower?
net.meta.cudnnOpts = {'cudnnworkspacelimit', 512 * 1024^3} ;
%net.meta.cudnnOpts = {'nocudnn'} ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------

% Add dropout to the fully-connected layers in the source model
drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;
net.layers = [net.layers(1:33) drop1 net.layers(34:35) drop2 net.layers(36:end)] ;

% Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add more padding to the input layer
%net.layers(1).block.pad = 100 ;
net.layers(5).block.pad = [0 1 0 1] ;
net.layers(10).block.pad = [0 1 0 1] ;
net.layers(17).block.pad = [0 1 0 1] ;
net.layers(24).block.pad = [0 1 0 1] ;
net.layers(31).block.pad = [0 1 0 1] ;
net.layers(32).block.pad = [3 3 3 3] ;
% ^-- we could do [2 3 2 3] but that would not use CuDNN

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

% Modify the last fully-connected layer to have 21 output classes
% Initialize the new filters to zero
for i = [1 2]
  p = net.getParamIndex(net.layers(end-1).params{i}) ;
  if i == 1
    sz = size(net.params(p).value) ;
    sz(end) = 2 ;
  else
    sz = [2 1] ;
  end
  net.params(p).value = zeros(sz, 'single') ;
end
net.layers(end-1).block.size = size(...
  net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

% Remove the last loss layer
net.removeLayer('prob') ;
net.setLayerOutputs('fc8', {'x38'}) ;


%% Add skip layers on top of pool5
net.addLayer('skip5', ...
     dagnn.Conv('size', [1 1 512 2], 'pad', 0), ...
     'x30', 'x39', {'skip5f','skip5b'});
f = net.getParamIndex('skip5f') ;
net.params(f).value = zeros(1, 1, 512, 2, 'single') ;
net.params(f).learningRate = 0.1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip5b') ;
net.params(f).value = zeros(1, 1, 2, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

%% Add skip layers on top of pool4
net.addLayer('skip4', ...
     dagnn.Conv('size', [1 1 512 2], 'pad', 0), ...
     'x23', 'x40', {'skip4f','skip4b'});
 f = net.getParamIndex('skip4f') ;
net.params(f).value = zeros(1, 1, 512, 2, 'single') ;
net.params(f).learningRate = 0.1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip4b') ;
net.params(f).value = zeros(1, 1, 2, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

%% Add skip layers on top of pool3
net.addLayer('skip3', ...
     dagnn.Conv('size', [1 1 256 2]), ...
     'x17', 'x41', {'skip3f','skip3b'});

f = net.getParamIndex('skip3f') ;
net.params(f).value = zeros(1, 1, 256, 2, 'single') ;
net.params(f).learningRate = 0.01 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip3b') ;
net.params(f).value = zeros(1, 1, 2, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

filters = single(bilinear_u(64, 2, 2)) ;
net.addLayer('deconv_f1', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 32, ...
  'crop', [16 16 16 16], ...
  'numGroups', 2, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x38', 'x42', 'deconvf1') ;

f = net.getParamIndex('deconvf1') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

filters0 = single(bilinear_u(4, 1, 2)) ;
net.addLayer('deconv_f0', ...
  dagnn.ConvTranspose(...
  'size', size(filters0), ...
  'upsample', 2, ...
  'crop', 1, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'x38', 'x51', 'deconvf0') ;

f = net.getParamIndex('deconvf0') ;
net.params(f).value = filters0 ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% -------------------------------------------------------------------------
% sum layer
% -------------------------------------------------------------------------
net.addLayer('sum1', dagnn.Sum(), {'x51', 'x39'}, 'x43') ;
% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------

filters = single(bilinear_u(32, 2, 2)) ;
net.addLayer('deconv_f2', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 16, ...
                      'crop',  8, ...
                      'numGroups', 2, ...
                      'hasBias', false, ...
                      'opts', net.meta.cudnnOpts), ...
             'x43', 'x44', 'deconvf2') ;

f = net.getParamIndex('deconvf2') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;


%% Add the first deconv layer
filters = single(bilinear_u(4, 1, 2)) ;
net.addLayer('deconv1', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 2, ...
                      'crop', 1, ...
                      'hasBias', false), ...
             'x43', 'x45', 'deconv1') ;
f = net.getParamIndex('deconv1') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

net.addLayer('sum2', dagnn.Sum(), {'x40', 'x45'}, 'x46') ;

filters = single(bilinear_u(16, 2, 2)) ;
net.addLayer('deconv_f3', ...
dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 8, ...
                      'crop', 4, ...
                      'numGroups', 2, ...
                      'hasBias', false, ...
                      'opts', net.meta.cudnnOpts), ...
             'x46', 'x47', 'deconvf3') ;

f = net.getParamIndex('deconvf3') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% %% Add the first deconv layer
% filters = single(bilinear_u(4, 1, 2)) ;
% net.addLayer('deconv2', ...
%   dagnn.ConvTranspose('size', size(filters), ...
%                       'upsample', 2, ...
%                       'crop', 1, ...
%                       'hasBias', false), ...
%              'x46', 'x48', 'deconv2') ;
% f = net.getParamIndex('deconv2') ;
% net.params(f).value = filters ;
% net.params(f).learningRate = 0 ;
% net.params(f).weightDecay = 1 ;

net.addLayer('sum3', dagnn.Sum(), {'x41', 'x46'}, 'x48') ;

filters = single(bilinear_u(16, 2, 2)) ;
net.addLayer('deconv_f4', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 8, ...
                      'crop', 4, ...
                      'numGroups', 2, ...
                      'hasBias', false, ...
                      'opts', net.meta.cudnnOpts), ...
             'x48', 'x49', 'deconvf4') ;

f = net.getParamIndex('deconvf4') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

net.addLayer('sumdeconv', dagnn.Sum(), {'x42','x44','x47', 'x49'}, 'prediction') ;

net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;
% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

end
