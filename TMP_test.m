% TMP with different low-rank denoisers
%------------------------------------------------
clear;
warning off;
randn('state',0);
rand('state',0);

% test parametrs
r=20; %rank
n1=250; %500
n2=250; %500
n=n1*n2;
rate=0.4;
m=fix(n*rate);
sparsity=0.1;

% sparse matrix 
s=1/sqrt(sparsity)*randn(n,1).*(rand(n,1)>(1-sparsity));
S=reshape(s,n1,n2);

% low-rank matrix
L=randn(n1,r)*randn(r,n2);
L=L/norm(L,'fro')*sqrt(n);

% noise level
sigma=sqrt(1e-5);

% set linear operator

% partial orthogonal 1
perm=randperm(n);
indexs=perm(1:m);
sign1=2*(rand(n,1)>0.5)-1;
sign2=2*(rand(n,1)>0.5)-1;
A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));
At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),n1,n2);

% partial orthogonal 2
% A=@(z) subsref(dct(z(:)),struct('type','()','subs',{{indexs}}));
% At=@(z) reshape(idct(put_vector(n,indexs,z)),n1,n2);

% random selection
% A=@(z) subsref(z(:),struct('type','()','subs',{{indexs}}));
% At=@(z) reshape(put_vector(n,indexs,z),size(L));

% Gaussian random
% GA=randn(m,n)/sqrt(n);
% A=@(z) GA*z(:);
% At=@(z) reshape(GA'*z,[n1,n2]);

% measurement 
y=A(L+S)+sigma*randn(m,1);
%-----------------------------
errL=@(z)norm(L-z,'fro')^2/norm(L,'fro')^2;
errS=@(z)norm(S-z,'fro')^2/norm(S,'fro')^2;
%---------test parameters-----------------
params.n1=n1;
params.n2=n2;
params.m=m;
params.sigma=sigma;
params.r=r;
params.iter=80;
params.inner_iter=3;
params.outer_iter=15;
params.Li=L;
params.Si=S;
params.errL=errL;
params.errS=errS;
params.beta=0.1;
params.psnr=1000;
params.adap=0;
% start running
tic;
params.vl = 1;
params.vs = 2;
[Lo,So,errl,errs]=TMP(A,At,y,params);
params.vl = 2;
params.vs = 2;
[Lo,So,errl1,errs1]=TMP_svt(A,At,y,params);
params.vl = 2;
params.vs = 2;
[Lo,So,errl2,errs2]=TMP_svht(A,At,y,params);
t=toc;

% plot
figure(1);
semilogy(1:length(errl),errl,'b -.',1:length(errs),errs,'r --',...
1:length(errl1),errl1,'g -d',1:length(errs1),errs1,'k -o',...
1:length(errl2),errl2,'y -^',1:length(errs2),errs2,'m ->','Markersize',5,'LineWidth',1.2);
legend('MSE of low-rank matrix (best rank-r)','MSE of sparse matrix (best rank-r)',...
    'MSE of low-rank matrix (SVST)','MSE of sparse matrix (SVST)',...
'MSE of low-rank matrix (SVHT)','MSE of sparse matrix (SVHT)');
xlabel('Iteration');
ylabel('MSE');
grid on;
set(gca,'FontSize',14);
set(gca, 'FontName', 'Times');