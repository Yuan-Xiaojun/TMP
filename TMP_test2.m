% algorithm comparisons
% by xuezhp
%------------------------------------------------
clear;
warning off;
r=2;%30
n1=256;%500
n2=256;%500
n=n1*n2;
rate=0.15;
m=fix(n*rate);
sparsity=0.02;
%-------------------------
s=1/sqrt(sparsity)*randn(n,1).*(rand(n,1)>(1-sparsity));
S=reshape(s,n1,n2);

L=randn(n1,r)*randn(r,n2);
L=L/norm(L,'fro')*sqrt(n);
%-------------------------
sigma=sqrt(1e-5);
perm=randperm(n);
indexs=perm(1:m);
sign1=2*(rand(n,1)>0.5)-1;
sign2=2*(rand(n,1)>0.5)-1;
A=@(z)subsref(idct(sign1.*dct(sign2.*z(:))),struct('type','()','subs',{{indexs}}));
At=@(z)reshape(sign2.*idct(sign1.*dct(put_vector(n,indexs,z))),n1,n2);
% A=@(z) subsref(dct(z(:)),struct('type','()','subs',{{indexs}}));
% At=@(z) reshape(idct(put_vector(n,indexs,z)),n1,n2);
% A=@(z)subsref(z(:),struct('type','()','subs',{{indexs}}));
% At=@(z)reshape(put_vector(n,indexs,z),size(L));
% GA=randn(m,n)/sqrt(n);
% A=@(z) GA*z(:);
% At=@(z) reshape(GA'*z,[n1,n2]);
y=A(L+S)+sigma*randn(m,1);
%-----------------------------
errL=@(z)norm(L-z,'fro')^2/norm(L,'fro')^2;
errS=@(z)norm(S-z,'fro')^2/norm(S,'fro')^2;
%---------test program-----------------
iter=30;
params.n1=n1;
params.n2=n2;
params.m=m;
params.sigma=sigma;
params.r=r;
params.iter=iter;
params.inner_iter=3;
params.outer_iter=15;
params.Li=L;
params.Si=S;
params.errL=errL;
params.errS=errS;
params.beta=0.05;
params.vl=1;
params.vs=2;
params.psnr=1000;
params.adap=0;
%-run TMP algorithm-
tic;
[Lo,So,errl,errs]=TMP(A,At,y,params);
t1=toc
%-run SPCP algorithm-
opts.sum=true;
opts.SVDstyle=3;
opts.size=[n1,n2];
opts.errL=errL;
opts.errS=errS;
opts.maxIts=iter;
A_cell={A,At};
lambda_S=1/sqrt(max(n1,n2));
epsilon=sqrt(m)*sigma;
AY=y;
tic;
[Lo2,So2,errHist,tau,errl2,errs2] = solver_RPCA_SPGL1(AY,lambda_S,epsilon,A_cell,opts);
t2=toc
%-run sparcs algorithm-
tic;
[L1,S1,errl1,errs1] = sparcs(y,r,sparsity*n, A, At, 'propack',1e-10, params.iter-1,0,-Inf,errL,errS);
t3=toc
% plot figure
figure(1);
semilogy(1:length(errl),errl,'b -.',1:length(errs),errs,'r --',...
1:length(errl1),errl1,'g -d',1:length(errs1),errs1,'k -o',...
1:length(errl2),errl2,'c -^',1:length(errs2),errs2,'m ->',...
'Markersize',10,'LineWidth',3);
legend('MSE of low-rank matrix (TMP)','MSE of sparse matrix (TMP)','MSE of low-rank matrix (SpaRCS)','MSE of sparse matrix (SpaRCS)','MSE of low-rank matrix (SPCP)','MSE of sparse matrix (SPCP)');
xlabel('Iteration');
ylabel('MSE');
grid on;
set(gca,'FontSize',15);
set(gca, 'FontName', 'Times');
