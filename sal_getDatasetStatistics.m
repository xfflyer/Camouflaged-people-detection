clear all;clc;
train_imgsavepath='G:\MatConvNet\matconvnet-fcn-master\sal_data\saliency_train_img\';
file=dir([train_imgsavepath,'*jpg']);
for t=1:length(file)
  mm=t
  name=[train_imgsavepath,file(t).name];
  rgb = imread(name) ;
  rgb = single(rgb) ;
  [w,h,channel]=size(rgb);
  rgb2=zeros(w,h,3);
  if channel==1
      rgb2(:,:,1)=rgb;
      rgb2(:,:,2)=rgb;
      rgb2(:,:,3)=rgb;
  else
      rgb2=rgb;
  end
  z = reshape(permute(rgb2,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
end
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;

stats.rgbMean = rgbm1 ;
stats.rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
save('G:\MatConvNet\matconvnet-fcn-master\sal_data\exdir\imdbStats.mat', '-struct', 'stats') ;