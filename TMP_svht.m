% compressed robust pca problem using turbo type message passing algorithm
% by xuezhp
% A: a partial orthogonal linear operator
% At: the transpose operator of A
% y: measurements
% with svht low-rank matrix denoiser
function [L,S,errl,errs]=TMP_svht(A,At,y,params)
n1=params.n1;
n2=params.n2;
m=params.m;
sigma=params.sigma;
% r=params.r; % do not require rank information.
iter=params.iter;
errL=params.errL;
errS=params.errS;
beta=params.beta;
psnr=params.psnr;
adap=params.adap;
vl=params.vl;
vs=params.vs;
%-----------------
n=n1*n2;
% initializations of mean and variance of L and S...
Lpri=zeros(n1,n2);
Spri=zeros(n1,n2);
vspri=1;
vlpri=1;
epsilon=1e-6;
Lext_old=zeros(n1,n2);
Sext_old=zeros(n1,n2);
vlext_old=0;
vsext_old=0;
for ii=1:iter
    %------------partial orthogonal linear operator (simplfied version)------------
    Lext=Lpri+n/m*At(y-A(Lpri+Spri));
    Sext=Spri+n/m*At(y-A(Lpri+Spri));
    vlext=n/m*(vspri+vlpri+sigma^2)-vlpri;
    vsext=n/m*(vspri+vlpri+sigma^2)-vspri;
    
    % dampling steps
    Lext=(1-beta)*Lext+beta*Lext_old;
    Sext=(1-beta)*Sext+beta*Sext_old;
    vlext=(1-beta)*vlext+beta*vlext_old;
    vsext=(1-beta)*vsext+beta*vsext_old;
    
    Lpri=Lext; Spri=Sext;
    vspri=vsext; vlpri=vlext;
    % (kl-divergence) auto damping
    kl1=n1*n2*(log(1/sqrt(vspri))+vspri/2)+norm(Spri,'fro')^2/2-n1*n2/2;
    kl2=n1*n2*(log(1/sqrt(vlpri))+vlpri/2)+norm(Lpri,'fro')^2/2-n1*n2/2;
    kl3=n1*n2*1/2*log(2*pi*sigma^2)+n1*n2*(vspri+vlpri)/(2*sigma^2)+norm(y-A(Lpri+Spri))^2/(2*sigma^2);
    klsum(ii)=kl1+kl2+kl3;
    if adap==1 && ii>1 && klsum(ii)>klsum(ii-1)
        beta=min(beta*1.3,0.95);
    else
        if adap==1 && ii>1 && klsum(ii)<klsum(ii-1)
        beta=max(beta*0.9,0.05);
        end
    end
    % caches
    Lext_old=Lext; vlext_old=vlext;
    Sext_old=Sext; vsext_old=vsext;
    %------------use singular value hard thresholding low-rank dernoiser-----------
    [U,sig,V]=svd(full(Lpri),'econ');
    sigg=diag(sig);
    THRESHOLD=linspace(1e-2,max(sigg),50);
    for ttt=1:length(THRESHOLD)
        threshold=THRESHOLD(ttt);
        r_y=sum(sigg>threshold);
        sig_x=zeros(size(sigg));
        sig_x(1:r_y)=sigg(1:r_y);
        div=div_svht(threshold,sigg,[n1 n2]); % diveregnce of svht
        a=sig_x-div/n*sigg;
        rr(ttt)=(a'*sigg)^2/(a'*a);
    end
    [~,indexmin]=max(rr(2:end));
    threshold=THRESHOLD(indexmin+1);
    r_y=sum(sigg>threshold);
    sig_x=zeros(size(sigg));
    sig_x(1:r_y)=sigg(1:r_y);
    div=div_svht(threshold,sigg,[n1 n2]);
    
    a=sig_x-div/n*sigg;
    c1=sum(a.*sigg)/sum(a.^2);
    sigxdf=c1*a;
    Lext=U*diag(sigxdf)*V';
    % vlext estimation
    if vl==1 % equ. (54c) asytopic
        vlext=vlpri*((1-r/n2*(1+n2/n1))/(1-div)^2-1);
    else
        if vl==2 % equ. (54b)
            %vlext=vlpri*norm(Lext,'fro')^2/norm(Lpri,'fro')^2;
            vlext=vlpri*norm(sigxdf)^2/norm(sigg)^2;
        else % equ. (54a)
            vlext=(norm(Lpri,'fro')^2-sum(sum(Lext.*Lpri))^2/norm(Lext,'fro')^2)/n;
        end
    end
    %------------sure-let sparse estimator part----------------
    [F,F_div]=Kernel_lin_1(reshape(Spri,n,1),vspri);
    %[F,F_div]=soft_thresholding(reshape(Spri,n,1),vspri);
    [len,~]=size(F);
    F_kernel=zeros(len,n);
    for jj=1:len
        F_kernel(jj,:)=F(jj,:)-F_div(jj)*reshape(Spri,1,n);
    end
    C_ff=F_kernel*F_kernel'/n+epsilon*eye(len);
    C_fy=conj(F_kernel)*reshape(Spri,n,1)/n;
    C_cal=mldivide(C_ff,C_fy);
    sext=F_kernel.'*C_cal;
    Sext=reshape(sext,n1,n2);
    % vsext estimation
    if vs==1
        vsext=norm(Sext-Spri,'fro')^2/n-vspri; % (41a)
    else
        vsext=(1*norm(y-A(Sext+Lext))^2/m-vlext-1*sigma^2); %(41b)
    end
    
    Lpri=Lext; Spri=Sext;
    vspri=vsext; vlpri=vlext;
    errl(ii)=errL(Lpri);
    errs(ii)=errS(Spri);
    if -20*log10(errl(ii))>psnr
        break;
    end
end
L=Lext;
S=Sext;
end