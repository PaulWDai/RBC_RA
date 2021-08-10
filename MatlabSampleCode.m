% Final Exam Matlab Code (Value Function Iteration and Calibration Part)
% Econ 242: Numerical Methods for Macro (lectured by Prof. Greenwood)
% Author: Weifeng Dai
% Date: Update in August 2020

clear all;
clc;
close;
%% Parameters

% Environment
alpha = 0.3;        % capital share in Cobb Douglas production function
beta = 1/(1.04);    % discount factor
delta = 0.08;       % depreciation rate
theta = 0.6;        % Frisch elasticity (disutility of work)
gamma = 2;          % CRRA coefficient (utility function)

% Markov chain
%{
z = 0;
pi = 1;
%}
z = 0.01735;    % calibrated from the model
pi = 0.775;   % calibrated from the model

% Setting grids
kmin = 3;
kmax = 4.5;
indexK = 500;
indexZ = 2;
kgrid = linspace(kmin,kmax,indexK);
zgrid = [1-z 1+z];
[z2D,k2D] = ndgrid(zgrid, kgrid); % 2-D grid

% Stimulation setting
T = 300;     % length of business cycle
rng('default'); % seed setting

% Graph setting
set(0,'defaultTextInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

%% Transitional Matrix
P = [pi 1-pi; 1-pi pi];

%% Value Function Iteration
% create 3D grid in order to introduce k'.
% 3D: (k, k', z)
k3D = repmat(k2D,1,1,indexK);
z3D = repmat(z2D,1,1,indexK);
h3D = ((1-alpha)*z3D.*k3D.^(alpha)).^(1/(alpha+theta));
o3D = z3D.*k3D.^alpha.*h3D.^(1-alpha);
r3D = alpha.*o3D./k3D;
w3D = (1-alpha).*o3D./h3D;
c3D = o3D + (1-delta)*k3D - permute(k3D,[1 3 2]) - h3D.^(1+theta)/(1+theta);
c3D(c3D<0) = 10^-8;
u3D = c3D.^(1-gamma)/(1-gamma);
tol = 10^-5;
iter = 0;
maxIter = 2000;
V = zeros(indexZ,indexK);
err = 1;
tic
while err > tol && iter < maxIter
    iter = iter +1;
    Vlast = V;
    EV = P*V;
    W = u3D + beta*permute(repmat(EV,1,1,indexK),[1 3 2]);
    [V,I] = max(W,[],3);
    err = max(abs(V-Vlast),[],'all');
end
toc
optK = kgrid(I);

%% Simulation

kS = zeros(1,T);
kS(1) = 3.9168; % steady state level of k when pi = 1, z = 0
% create discrete time markov process
mc = dtmc(P);
zSindex = simulate(mc,T);
zS = zgrid(zSindex);
zS = zS(1:T);
idx = zeros(1,T);
kSindex = zeros(1,T);
kSindex(1) = round((kS(1)-kmin)/(kmax-kmin)*(indexK-1))+1;
kSindex(kSindex<1) = 1;
kSindex(kSindex>indexK) = indexK;
for j = 1:T-1
    idx(j) = sub2ind(size(optK),zSindex(j),kSindex(j));
    kS(j+1) = optK(idx(j));
    kSindex(j+1) = round((kS(j+1)-kmin)/(kmax-kmin)*(indexK-1))+1;
    kSindex(kSindex<1) = 1; % index the outlier
    kSindex(kSindex>indexK) = indexK;
end
% to reshape the vector with length of T-1;
hS = ((1-alpha)*zS.*kS.^(alpha)).^(1/(alpha+theta));
oS = zS.*kS.^alpha.*hS.^(1-alpha);
hS = hS(1:T-1);
oS = oS(1:T-1);
zS = zS(1:T-1);
iS(1:T-1) = kS(2:T) - (1-delta)*kS(1:T-1);
iS(iS<=0) = 10^-3; % available for taking log
prodS = oS./hS;
cS(1:T-1) = oS + (1-delta)*kS(1:T-1) - kS(2:T) - hS.^(1+theta)/(1+theta);
cS(cS<=0) = 10^-3; % available for taking log
kS = kS(1:T-1);


%% Moments

% All variables are in logs.
% Autocorrelations
auto_o = corr(log(oS(1:T-2))',log(oS(2:T-1))');
auto_c = corr(log(cS(1:T-2))',log(cS(2:T-1))');
auto_i = corr(log(iS(1:T-2))',log(iS(2:T-1))');
auto_h = corr(log(hS(1:T-2))',log(hS(2:T-1))');
auto_prod = corr(log(prodS(1:T-2))',log(prodS(2:T-1))');
% Standard deviations in percent
std_o = std(log(oS))*100;
std_c = std(log(cS))*100;
std_i = std(log(iS))*100;
std_h = std(log(hS))*100;
std_prod = std(log(prodS));
% Correlation with output
corr_o = corr(log(oS)',log(oS)');
corr_c = corr(log(cS)',log(oS)');
corr_i = corr(log(iS)',log(oS)');
corr_h = corr(log(hS)',log(oS)');
corr_prod = corr(log(prodS)',log(oS)');
% Results
disp('Model Moments')
ModelMoment = ...
      [std_o     auto_o      corr_o      ;...
      std_c     auto_c      corr_c      ;...
      std_i     auto_i      corr_i      ;...
      std_h     auto_h      corr_h      ;...
      std_prod  auto_prod   corr_prod   ];
disp(round(ModelMoment,2));
disp('Data Moments')
DataMoment =  ...
     [3.5       0.66        1.00        ;...
      2.2       0.72        0.74        ;...
      10.5      0.25        0.68        ;...
      2.1       0.39        0.81        ;...
      2.2       0.77        0.82        ];
disp(round(DataMoment,2));
%% Figures

% Check policy function
figure(1);
subplot(1,2,1);
plot(kgrid,kgrid,'r:',kgrid,optK(1,:),'b-','LineWidth',2);
title('Policy function for low TFP $1-z$')
subplot(1,2,2);
plot(kgrid,kgrid,'r:',kgrid,optK(1,:),'b-','LineWidth',2);
title('Policy function for low TFP $1+z$')

% Check value function
figure(2);
subplot(1,2,1);
plot(kgrid,V(1,:),'b-','LineWidth',2);
title('Value function for low TFP $1-z$')
subplot(1,2,2);
plot(kgrid,V(1,:),'b-','LineWidth',2);
title('Value function for low TFP $1+z$')
% End
