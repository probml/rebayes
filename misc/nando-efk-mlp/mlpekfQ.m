function [y,theta,thetaRecord,PRecord,OutputVariance,qplot] = mlpekfmax(x,d,s1,s2,Rparameter,Qparameter,KalmanP,initVar,window,tsteps)
% PURPOSE: To simulate a standard EKF-MLP training algorithm.
% INPUTS  : - x = The network input.
%           - d = The network target vector.
%           - s1 = Number of neurons in the hidden layer.
%           - s2 = Number of neurons in the output layer (1).
%           - Rparameter = EKF measurement noise hyperparameter.
%           - Qparameter = EKF process noise hyperparameter.
%           - KalmanP = Initial EKF covariance.
%           - initVar = Prior variance of the weights.
%           - window = Window length to compute time covariance.
%           - tsteps = Number of time steps (input error checking).
% OUTPUTS : - y = The network output.
%           - theta = The final weights.
%           - thetaRecord = The weights at each time step.
%           - PRecord = The EKF covariance at each time step.
%           - OutputVariance = The innovations covariance.
%           - qplot = Qparameter at each time step.   

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECKING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 10, error('Not enough input arguments.'); end

% Check that the size of input (x) is N by L, where N is the dimension
% of the input and L is the length of the data (number of samples).
[N,L] = size(x);
[D,L] = size(d);
if (L ~= tsteps), error('d must be of size 1x(time steps).'), end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALISATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N,L] = size(x);
[D,L] = size(d);

T  = s2*(s1+1) + s1*(N+1);                % No weights. The 1 is for the bias terms.
theta = sqrt(initVar)*(randn(T,1));       % Parameter vector.
H = zeros(T,D);                           % Jacobian Matrix.
K = zeros(T,D);                           % Kalman Gain matrix.
P = sqrt(KalmanP)*eye(T,T);               % Weight covariance matrix.
R = Rparameter*eye(D);                    % Measurement noise covariance.
Q = Qparameter*eye(T,T);                  % Process noise covariance.
o1 = zeros(s1,1);
y = zeros(s2,L);
w2 = zeros(s2,s1+1);
w1 = zeros(s1,N+1);

thetaRecord=zeros(T,L);
PRecord=zeros(T,T,L);
r = zeros(s2,L);
qplot=zeros(1,L);
Htime=zeros(L,T);
OutputVariance=zeros(1,L);


     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN SAMPLES LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for samples = 1:L,
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FEED FORWARD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % fill in weight matrices using the parameter vector: 

  for i = 1:s2,
    for j = 1:(s1+1),
      w2(i,j)= theta(i*(s1+1)+j-(s1+1),1);
    end;
  end;
  for i = 1:s1,
    for j = 1:(N+1),
      w1(i,j)= theta(s2*(s1+1) +i*(N+1)+j-(N+1),1);
    end;
  end;

  % Compute the network outputs for each layer:
  u1 = w1*[1 ; x(:,samples)]; 
  o1 = 1./(1+exp(-u1));
  u2 = w2*[1 ; o1];
  y(:,samples)=u2;  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FILL H %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % output layer:
    for i = 1:s2,
      for j = 1:(s1+1),
        if j==1
          H(i*(s1+1) + j - (s1+1) ,1)= 1;
        else
          H(i*(s1+1) + j - (s1+1) ,1)= o1(j-1,1);
        end;
      end;
    end;
    
    % Second layer:
    for i = 1:s1,
      for j = 1:(N+1),
        rhs = w2(1,i+1)*o1(i,1)*(1-o1(i,1));
        if j==1
          H(s2*(s1+1) + i*(N+1) + j - (N+1) ,1) = rhs;
        else
          H(s2*(s1+1) + i*(N+1) + j - (N+1) ,1)= rhs * x(j-1,samples);
        end;
      end;
    end;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%% E - KALMAN EQUATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  Pold=P;
  Htime(samples,:)=H(:,1)';
  r = d-y; 
  
  K = (P+Q) *H * ((R + H'*(P+Q)*H)^(-1));
  theta = theta + K * (d(:,samples) - y(:,samples));
  P = P -  K*H'*(P+Q) + Q;

  thetaRecord(:,samples)=theta;
  PRecord(:,:,samples)=P;

  OutputVariance(1,samples) = R + H'*(P)*H;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UPDATE Q %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if samples > window
    S = zeros(1,T);
    sumS = 0;
    for l = 1:window,
      if l==1,
        S = (1/window) * (Htime(samples,:)) ./ R.^(1/2);  
      else
        S = (1/window) * sum( (Htime(samples-(l-1):samples,:)) ./ R.^(1/2) ); 
      end;
      sumS=sumS + S*S';
    end;
    CovTime=((1/window)*sum(r(:,samples-window+1:samples) ./ R.^(1/2))).^(2);
    CovEnsemble=S*Pold*S' + (1/window);
    CovDifference = CovTime - CovEnsemble;
    
    if CovDifference > 0
      Qparameter = CovDifference / sumS;
    else
      Qparameter = 0;
    end;
    Q=Qparameter*eye(T,T);
  end;
  qplot(1,samples)=Qparameter;  

end;












