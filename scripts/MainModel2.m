%% MainModel2.m    Estimates the "Convenience Yield Model" from Global Trends in
%                  Interest Rates. Results are saved as
%                  "OutputModel2.mat."


%% Initial setup
clear;

filename = '../results/OutputModel2.mat';  % Output filename

addpath Routines

[DATA,TEXT] = xlsread('../indata/DataInflShortLongConsUpdated.xlsx');

Year = DATA(:,1);

Ndraws  =  100000;      % Number of MCMC draws
p = 1;                  % Number of lags in the VAR for the cycle;



Mnem = TEXT(2:end);
%1     'cpi_usa'
%2     'stir_usa'
%3     'ltir_usa'
%4     'rconpc_usa'
%5     'cpi_deu'
%6     'stir_deu'
%7     'ltir_deu'
%8     'rconpc_deu'
%9     'cpi_uk'
%10    'stir_uk'
%11    'ltir_uk'
%12    'rconpc_uk'
%13    'cpi_fr'
%14    'stir_fr'
%15    'ltir_fr'
%16    'rconpc_fr'
%17    'cpi_ca'
%18    'stir_ca'
%19    'ltir_ca'
%20    'rconpc_ca'
%21    'cpi_it'
%22    'stir_it'
%23    'ltir_it'
%24    'rconpc_it'
%25    'cpi_jp'
%26    'stir_jp'
%27    'ltir_jp'
%28    'rconpc_jp'
%29    'baa_usa'
%30    'cpi_au'
%31    'stir_au'
%32    'ltir_au'
%33    'rconpc_au'
%34    'cpi_be'
%35    'stir_be'
%36    'ltir_be'
%37    'rconpc_be'
%38    'cpi_fi'
%39    'stir_fi'
%40    'ltir_fi'
%41    'rconpc_fi'
%42    'cpi_ie'
%43    'stir_ie'
%44    'ltir_ie'
%45    'rconpc_ie'
%46    'cpi_nl'
%47    'stir_nl'
%48    'ltir_nl'
%49    'rconpc_nl'
%50    'cpi_no'
%51    'stir_no'
%52    'ltir_no'
%53    'rconpc_no'
%54    'cpi_ch'
%55    'stir_ch'
%56    'ltir_ch'
%57    'rconpc_ch'
%58    'cpi_se'
%59    'stir_se'
%60    'ltir_se'
%61    'rconpc_se'
%62    'cpi_es'
%63    'stir_es'
%64    'ltir_es'
%65    'rconpc_es'
%66    'cpi_pt'
%67    'stir_pt'
%68    'ltir_pt'
%69    'rconpc_pt'
%70    'cpi_dk'
%71    'stir_dk'
%72    'ltir_dk'
%73    'rconpc_dk'


Country = {'US','DE','UK','FR','CA','IT','JP','AU','BE', 'FI', 'IE', 'NL', 'NO', 'CH', 'SE', 'ES', 'PT'};
codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt';
Nc = numel(codes);
X = DATA(:,2:end);

country_start = [0, 4, 8, 12, 16, 20, 24, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69];

for i = 1:Nc
    code = codes{i};
    start_col = country_start(i);
    
    eval(sprintf('Price_%s = X(:, start_col + 1);', code));
    eval(sprintf('Stir_%s = X(:, start_col + 2);', code));
    eval(sprintf('Ltir_%s = X(:, start_col + 3);', code));
    eval(sprintf('Cons_%s = X(:, start_col + 4);', code));
    eval(sprintf('Dcons_%s = [NaN; (Cons_%s(2:end, :) ./ Cons_%s(1:end - 1) - 1)*100];', code, code, code));
    eval(sprintf('Infl_%s = [NaN; (Price_%s(2:end) ./ Price_%s(1:end-1)-1)*100];', code, code, code));
end

Baa_us = X(:,29);

Y = [
    Stir_us...
    Stir_de...
    Stir_uk...
    Stir_fr...
    Stir_ca...
    Stir_it...
    Stir_jp...
    Stir_au...  % Added Australia
    Stir_be...  % Added Belgium
    Stir_fi...
    Stir_ie...
    Stir_nl...
    Stir_no...
    Stir_ch...
    Stir_se...
    Stir_es...
    Stir_pt...
    Infl_us...
    Infl_de...
    Infl_uk...
    Infl_fr...
    Infl_ca...
    Infl_it...
    Infl_jp...
    Infl_au...  % Added Australia
    Infl_be...  % Added Belgium
    Infl_fi...
    Infl_ie...
    Infl_nl...
    Infl_no...
    Infl_ch...
    Infl_se...
    Infl_es...
    Infl_pt...
    Ltir_us...
    Ltir_de...
    Ltir_uk...
    Ltir_fr...
    Ltir_ca...
    Ltir_it...
    Ltir_jp...
    Ltir_au...  % Added Australia
    Ltir_be...  % Added Belgium
    Ltir_fi...
    Ltir_ie...
    Ltir_nl...
    Ltir_no...
    Ltir_ch...
    Ltir_se...
    Ltir_es...
    Ltir_pt...
    Baa_us
    ];


Mnem = {
    'Stir_us'...
    'Stir_de'...
    'Stir_uk'...
    'Stir_fr'...
    'Stir_ca'...
    'Stir_it'...
    'Stir_jp'...
    'Stir_au'...  % Added Australia
    'Stir_be'...  % Added Belgium
    'Stir_fi'...
    'Stir_ie'...
    'Stir_nl'...
    'Stir_no'...
    'Stir_ch'...
    'Stir_se'...
    'Stir_es'...
    'Stir_pt'...
    'Infl_us'...
    'Infl_de'...
    'Infl_uk'...
    'Infl_fr'...
    'Infl_ca'...
    'Infl_it'...
    'Infl_jp'...
    'Infl_au'...  % Added Australia
    'Infl_be'...  % Added Belgium
    'Infl_fi'...
    'Infl_ie'...
    'Infl_nl'...
    'Infl_no'...
    'Infl_ch'...
    'Infl_se'...
    'Infl_es'...
    'Infl_pt'...
    'Ltir_us'...
    'Ltir_de'...
    'Ltir_uk'...
    'Ltir_fr'...
    'Ltir_ca'...
    'Ltir_it'...
    'Ltir_jp'...
    'Ltir_au'...  % Added Australia
    'Ltir_be'...  % Added Belgium
    'Ltir_fi'...
    'Ltir_ie'...
    'Ltir_nl'...
    'Ltir_no'...
    'Ltir_ch'...
    'Ltir_se'...
    'Ltir_es'...
    'Ltir_pt'...
    'Baa_us'
    };


Y(abs(Y)>30)=NaN;

T0 = 100;
T1 = 144;

[T,n] = size(Y);

T0pre = find(Year==1870);
T1pre = find(Year==1899);
disp(['Avg. and std in the presample: 1954-1959'])
disp([(1:n)' nanmean(Y(T0pre:T1pre,:))' nanstd(Y(T0pre:T1pre,:))'])

disp('mean Stir')
disp(nanmean(nanmean(Y(T0pre:T1pre,1:Nc))))

disp('mean Infl')
disp(nanmean(nanmean(Y(T0pre:T1pre,Nc+1:Nc*2))))

disp('mean Ltir')
disp(nanmean(nanmean(Y(T0pre:T1pre,Nc*2+1:Nc*3))))

disp('mean Baa us')
disp(nanmean(nanmean(Y(T0pre:T1pre,Nc*3+1))))


disp('std Stir')
disp(nanmean(nanstd(Y(T0pre:T1pre,1:Nc))))

disp('std Infl')
disp(nanmean(nanstd(Y(T0pre:T1pre,Nc+1:Nc*2))))

disp('std Ltir')
disp(nanmean(nanstd(Y(T0pre:T1pre,Nc*2+1:Nc*3))))

disp('std Baa')
disp(nanmean(nanstd(Y(T0pre:T1pre,Nc*3+1))))


%% Setup model, initial conditions

T0 = find(Year==1870);
T1 = find(Year==2024);


Y = Y(T0:T1,:);
Year = Year(T0:T1);
y=Y;
[T,n] = size(y);

%        m_us  pi_wrd  ts_wrd     cy_us
Ctr =[
    1       1      0          -1   %     Stir_us...
    1       1      0          -1   %     Stir_de...
    1       1      0          -1   %     Stir_uk...
    1       1      0          -1   %     Stir_fr...
    1       1      0          -1   %     Stir_ca...
    1       1      0          -1   %     Stir_it...
    1       1      0          -1   %     Stir_jp...     
    1       1      0          -1   %     Stir_au...
    1       1      0          -1   %     Stir_be...
    1       1      0          -1   %     Stir_fi...
    1       1      0          -1   %     Stir_ie...
    1       1      0          -1   %     Stir_nl...
    1       1      0          -1   %     Stir_no...
    1       1      0          -1   %     Stir_ch...
    1       1      0          -1   %     Stir_se...
    1       1      0          -1   %     Stir_es...
    1       1      0          -1   %     Stir_pt...
    0       1      0           0   %     Infl_us...
    0       1      0           0   %     Infl_de...
    0       1      0           0   %     Infl_uk...
    0       1      0           0   %     Infl_fr...
    0       1      0           0   %     Infl_ca...
    0       1      0           0   %     Infl_it...
    0       1      0           0   %     Infl_jp...
    0       1      0           0   %     Infl_au...
    0       1      0           0   %     Infl_be...
    0       1      0           0   %     Infl_fi...
    0       1      0           0   %     Infl_ie...
    0       1      0           0   %     Infl_nl...
    0       1      0           0   %     Infl_no...
    0       1      0           0   %     Infl_ch...
    0       1      0           0   %     Infl_se...
    0       1      0           0   %     Infl_es...
    0       1      0           0   %     Infl_pt...
    1       1      1          -1   %     Ltir_us...
    1       1      1          -1   %     Ltir_de...
    1       1      1          -1   %     Ltir_uk...
    1       1      1          -1   %     Ltir_fr...
    1       1      1          -1   %     Ltir_ca...
    1       1      1          -1   %     Ltir_it...
    1       1      1          -1   %     Ltir_jp...
    1       1      1          -1   %     Ltir_au...
    1       1      1          -1   %     Ltir_be...
    1       1      1          -1   %     Ltir_fi...
    1       1      1          -1   %     Ltir_ie...
    1       1      1          -1   %     Ltir_nl...
    1       1      1          -1   %     Ltir_no...
    1       1      1          -1   %     Ltir_ch...
    1       1      1          -1   %     Ltir_se...
    1       1      1          -1   %     Ltir_es...
    1       1      1          -1   %     Ltir_pt...
    1       1      1           0   %     Baa_us
    ];

%Adding country specific trends to real rates (cy)
% Cadd1                   =    zeros(n,7);
% Cadd1(1:7,1:7)          =    eye(7); %Stir
% Cadd1(15:21,1:7)        =    eye(7); %Ltir
% Cadd1(22,1)             =    0;      %no convenience yield of Baa
Cadd1 = zeros(n, Nc);
Cadd1(1:Nc, 1:Nc) = eye(Nc); %Stir, Nc countries
Cadd1(Nc*2+1:Nc*3, 1:Nc) = eye(Nc); %Ltir, Nc countries
Cadd1(Nc*3+1, 1) = 0; %no convenience yield of Baa

%Adding the country specific trends in inflation rates
% Cadd2              =    zeros(n,7);
% Cadd2(1:7,1:7)     =    eye(7); %Stir
% Cadd2(8:14,1:7)    =    eye(7); %Infl
% Cadd2(15:21,1:7)   =    eye(7); %Ltir
% Cadd2(22,1)        =    1;      %Baa is nominal hence loads on same pi_us
Cadd2 = zeros(n, Nc);
Cadd2(1:Nc, 1:Nc) = eye(Nc);
Cadd2(Nc+1:2*Nc, 1:Nc) = eye(Nc);
Cadd2(2*Nc+1:3*Nc, 1:Nc) = eye(Nc);
Cadd2(3*Nc+1, 1) = 1;

%Adding country specific trends to term spread
% Cadd3                 =    zeros(n,7);
% Cadd3(15:21,1:7)      =    eye(7); %Ltir
% Cadd3(22,1)           =    1;      %Baa has long maturities, hence it loads on ts_us
Cadd3 = zeros(n, Nc);
Cadd3(2*Nc+1: Nc*3, 1:Nc) = eye(Nc);
Cadd3(3*Nc+1, 1) = 1;


Ctr           = [Ctr Cadd1 Cadd2 Cadd3];
Ccyc          = zeros(n,n*p);
Ccyc(1:n,1:n) = eye(n);
C             = [Ctr Ccyc];


r = size(Ctr,2);

b0          = zeros(n*p,n);
b0(1:n,1:n) = eye(n)*0;

df0tr = 100;

%                rs_us       pi_wrd    ts_wrd  cy         rs_idio        pi_idio            ts_idio
% SC0tr =     ([   1/sqrt(2)   sqrt(2)   1       1/sqrt(2)  1*ones(1,7)    sqrt(2)*ones(1,7)  1*ones(1,7)   ]).^2/100;
% S0tr  =      [   1.5         2         1       1          zeros(1,7)     zeros(1,7)         zeros(1,7)    ]';
% P0tr  = diag([   1           2         1       1          1*ones(1,7)/2  2*ones(1,7)/2      1*ones(1,7)/2 ].^2);
SC0tr =     ([   1/sqrt(2)   sqrt(2)   1       1/sqrt(2)  1*ones(1,Nc)    sqrt(2)*ones(1,Nc)  1*ones(1,Nc)   ]).^2/100;
S0tr  =      [   1.5         2         1       1          zeros(1,Nc)     zeros(1,Nc)         zeros(1,Nc)    ]';
P0tr  = diag([   1           2         1       1          1*ones(1,Nc)/2  2*ones(1,Nc)/2      1*ones(1,Nc)/2 ].^2);


%                 stir        infl         ltir       cy
Psi =       (2*[  ones(1,Nc)   2*ones(1,Nc)  ones(1,Nc)  1]).^2;

S0cyc = zeros(n*p,1);

Atr  = eye(r);
Qtr  = diag(SC0tr);

% Initialize  cyclic component
My             = ones(T,1)*nanmean(y);
yint           = y;
yint(isnan(y)) = My(isnan(y));
[Trend,Ycyc]   = hpfilter(yint,1000);
[beta, sigma]  = BVAR(Ycyc, p, b0, Psi, .2, 0);

Acyc                  = zeros(n*p);
Acyc(n+1:end,1:end-n) = eye(n*(p-1));
Acyc(1:n,:)           = beta';

Qcyc          = zeros(n*p);
Qcyc(1:n,1:n) = (sigma + sigma') / 2;  % Symmetric
P0cyc         = dlyap(Acyc,Qcyc);

% Initialize transition matrix
A                  = zeros(r+n*p);
A(1:r,1:r)         = Atr;
A(r+1:end,r+1:end) = Acyc;


% Initialize variance-covariance matrix of transition equation
Q                  = zeros(r+n*p);
Q(1:r,1:r)         = Qtr;
Q(r+1:end,r+1:end) = Qcyc;


R = eye(n)*1e-12;


% Starting conditions for the Kalman recursion
S0                  = [S0tr;S0cyc];
P0                  = zeros(r+n*p);
P0(1:r,1:r)         = P0tr;
P0(r+1:end,r+1:end) = P0cyc;

tic

% Store MCMC
States = ones(T,r+n*p,Ndraws)*NaN;
Trends = ones(T,n,Ndraws)*NaN;
LogLik = ones(1,Ndraws)*NaN;
SS0    = ones(r,Ndraws)*NaN;
Theta  = ones(1,Ndraws);
AA     = ones(r+n*p,r+n*p,Ndraws)*NaN;
QQ     = ones(r+n*p,r+n*p,Ndraws)*NaN;
CC     = ones(n,r+n*p,Ndraws)*NaN;
RR     = ones(n,n,Ndraws)*NaN;

P_acc_var = ones(1,Ndraws)*NaN;
P_acc     = ones(1,Ndraws)*NaN;

% Set prior to paper's lambda (coefficient to inf_wrld)
mean_theta = 1;
std_theta = .5;

% C(1:21,2)        = mean_theta;
% C([1:7 15:21],1) = mean_theta;
C(1:Nc*3, 2) = mean_theta;
C([1:Nc 2*Nc+1:3*Nc], 1) = mean_theta;


lambda = .2;  % Minnesota prior

logML = -inf;


%% Begin estimation

for jm = 1:Ndraws  % MCMC draws
    
    % ----------------------Block 1a -------------------------------------
    
    kf = KF(y,C,R,A,Q,S0,P0);
    loglik = kf.LogLik;
    
    theta_old = [];
    % theta_old = [theta_old; C(1:7,2)] ;% Loadings on global inflation
    theta_old = [theta_old; C(1:Nc, 2)];

    theta_new = theta_old + randn(size(theta_old))*std_theta/5;
    
    C_new = C;
    % C_new(1:7,2)   = theta_new(1:7);   % loadings of stir on pi_wrd
    % C_new(8:14,2)  = theta_new(1:7);   % loadings of pi on pi_wrd
    % C_new(15:21,2) = theta_new(1:7);   % loadings of ltir on pi_wrd
    % C_new(22,2)    = theta_new(1);     % loadings of baa  on pi_wrd
    C_new(1:Nc, 2) = theta_new(1:Nc);
    C_new(Nc+1:2*Nc, 2) = theta_new(1:Nc);
    C_new(2*Nc+1:3*Nc, 2) = theta_new(1:Nc);
    C_new(3*Nc+1, 2)= theta_new(1);

    kf_new     = KF(y, C_new, R, A, Q, S0, P0);
    loglik_new = kf_new.LogLik;
    
    log_rat =   (loglik_new + sum(log(normpdf(theta_new, mean_theta, std_theta^2)))) ...
              - (loglik     + sum(log(normpdf(theta_old, mean_theta, std_theta^2))));
    p_acc = min(exp(log_rat),1);
    
    if rand<=p_acc
        C      = C_new;
        loglik = loglik_new;
        kf     = kf_new;
    end;
    
    % ----------------------------- Block 2a ------------------------------
    
    kc   = KC(kf);
    Ytr  = [kc.S0(1:r)'; kc.S(:,1:r)]; % Trend components
    Ycyc = kc.S(:,r+1:r+n);            % Cyclical component
    
    % ----------------------------- Block 2b ------------------------------
    
    SCtr                = CovarianceDraw(diff(Ytr),df0tr,diag(SC0tr));
    Q(1:r,1:r)          = SCtr;
    
    
    
    for jp=1:p   % Organize cycle components by including the initial values
        Ycyc = [kc.S0(r+(jp-1)*n+1:r+n*jp)'; Ycyc];
    end;
    
    % Estimate cycle component
    [beta,sigma]       = BVAR(Ycyc, p, b0, Psi, lambda, 1);
    Acyc_new           = Acyc;
    Acyc_new(1:n,:)    = beta';
    
    if max(abs(eigs(Acyc_new))) < 1  % Convergence criteria
       
        Qcyc_new               = Qcyc;
        Qcyc_new(1:n,1:n)      = (sigma+sigma')/2;  % Symmetric
        P0cyc_new              = dlyap(Acyc_new, Qcyc_new);
        P0cyc_new              = (P0cyc_new + P0cyc_new')/2;
        Y0cyc = kc.S0(r+1:end);
        
        rat = mvnpdf(Y0cyc, Y0cyc*0, P0cyc_new) / mvnpdf(Y0cyc, Y0cyc*0, P0cyc);
        
        p_acc_var = min(rat,1);
        
        
        if rand<=p_acc_var  % Accept
            Acyc  = Acyc_new;
            Qcyc  = Qcyc_new;
            P0cyc = P0cyc_new;
            A(r+1:end,r+1:end)  = Acyc;
            Q(r+1:end,r+1:end)  = Qcyc;
            P0(r+1:end,r+1:end) = P0cyc;
        end;
        

        
    else
        p_acc_var =0;
    end
    
    % Store MCMC
    States(:,:,jm) = kc.S;
    Trends(:,:,jm) = kc.S(:,1:r)*C(:,1:r)';
    LogLik(jm) = loglik;
    LogML(jm) = logML;
    SS0(:,jm) = S0(1:r);
    AA(:,:,jm) = A;
    QQ(:,:,jm) = Q;
    CC(:,:,jm) = C;
    RR(:,:,jm) = R;
    Lambda(jm) = lambda;
    
    P_acc(jm)    = p_acc;
    P_acc_var(jm)= p_acc_var;
    
        if mod(jm,10)==0  % Print to command window
        if jm>1
            if jm <=1000
                disp([num2str(jm),'th draw of ',num2str(Ndraws),'; Elapsed time: ',num2str(toc),' seconds'])
                disp(['Acceptance rate so far: ',num2str(mean(P_acc(1:jm)))])
                disp(['Acceptance rate so far: ',num2str(mean(P_acc_var(1:jm)))])
            elseif jm <10000
                disp([num2str(jm),'th draw of ',num2str(Ndraws),'; Elapsed time: ',num2str(toc),' seconds'])
                disp(['Acceptance rate of the last 1k draws: ',num2str(mean(P_acc(jm-1000+1:jm)))])
                disp(['Acceptance rate of the last 1k draws: ',num2str(mean(P_acc_var(jm-1000+1:jm)))])
            else
                disp([num2str(jm),'th draw of ',num2str(Ndraws),'; Elapsed time: ',num2str(toc),' seconds'])
                disp(['Acceptance rate of the last 10k draws: ',num2str(mean(P_acc(jm-5000+1:jm)))])
                disp(['Acceptance rate of the last 10k draws: ',num2str(mean(P_acc_var(jm-5000+1:jm)))])
            end;
        end;
        
    end;
end;

Ndraws = length(Lambda)-1;

skip = 1;
Discard = floor(Ndraws/2);  % Burn-in index

% Remove burn-in draws
States = States(:,:,Discard+1:Ndraws);
Trends = Trends(:,:,Discard+1:Ndraws);
AA     = AA(:,:,Discard+1:Ndraws);
QQ     = QQ(:,:,Discard+1:Ndraws);
CC     = CC(:,:,Discard+1:Ndraws);
RR     = RR(:,:,Discard+1:Ndraws);
LogLik = LogLik(:,Discard+1:Ndraws);
LogML  = LogML(:,Discard+1:Ndraws);
SS0    = SS0(:,Discard+1:Ndraws);


CommonTrends = States(:,1:r,:);
Cycles       = States(:,r+1:r+n,:);

mStates = nanmean(States,3);


save(filename, '-v7.3')
