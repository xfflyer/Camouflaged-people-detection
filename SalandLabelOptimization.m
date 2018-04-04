function optwCtr = SalandLabelOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, fgWeight,label,th1,th2)
% Solve the least-square problem in Equa(9) in our paper

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014
spNum = length(bgWeight);
adjcMatrix_nn = LinkNNAndBoundary(adjcMatrix, bdIds);
colDistM(adjcMatrix_nn == 0) = Inf;
Wn = Dist2WeightMatrix(colDistM, neiSigma);      %smoothness term
mu = 0.1;                                                   %small coefficients for regularization term
W = Wn + adjcMatrix * mu;                                   %add regularization term
D = diag(sum(W));

M=mean(fgWeight);
t=normalize(label);
label=normalize(label);
t(t>=th1)=1;t(t<th2)=0;
label(label>=th1)=1;
label(label<th2)=1;
label(label~=1)=0;
E_Label=diag(label);
bgLambda = 5;   %global weight for background term, bgLambda > 1 means we rely more on bg cue than fg cue.
%bgLambda = 1;
E_bg = diag(bgWeight * bgLambda);       %background term
E_fg = diag(fgWeight);          %foreground term

labelamda=1;
optwCtr =(D - W + E_bg + E_fg+labelamda*E_Label) \ (E_fg * ones(spNum, 1)+labelamda*E_Label*t);
optwCtr=normalize(optwCtr);