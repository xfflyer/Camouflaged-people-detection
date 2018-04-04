clear all;clc;
basepath='E:\FCNCam\matconvnet-DDCN\result\DDCN2\';
impath='E:\Camouflage Dataset\img\';
gtpath='E:\Camouflage Dataset\label\';

imsavepath='E:\Camouflage Dataset\CamouflageData\img\';
gtsavepath='E:\Camouflage Dataset\CamouflageData\gt\';
% respath1='E:\FCNCam\matconvnet-DDCN\result\DDCN\';
respath2='E:\FCNCam\matconvnet-DDCN\result\Final\';
% respath3='E:\FCNCam\matconvnet-DDCN\result\DDCN2\';
respath4='E:\FCNCam\matconvnet-DDCN\result\Final2\';
imfile=dir([basepath,'*.png']);
for i=1:length(imfile)
    name=imfile(i).name;
    gtimg=imread([gtpath,name(1:end-4),'.png']);
    imwrite(gtimg,[gtsavepath,name(1:end-4),'.png']);
    img1=imread([impath,name(1:end-4),'.jpg']);
    imwrite(img1,[imsavepath,name(1:end-4),'.jpg']);
    resimg2=imread([respath2,name(1:end-4),'.png']);
    imwrite(resimg2,[respath4,name(1:end-4),'.png']);
end