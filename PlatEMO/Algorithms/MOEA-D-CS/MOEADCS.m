function MOEADCS(Global)
% <algorithm> <M>
% MOEA/D based on differential evolution
% delta --- 0.9 --- The probability of choosing parents locally
% pa    --- 0.25 --- The global random walk probability
% nr    ---   2 --- Maximum number of solutions replaced by each offspring

%------------------------------- Reference --------------------------------
% H. Li and Q. Zhang, Multiobjective optimization problems with complicated
% Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on Evolutionary
% Computation, 2009, 13(2): 284-302.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    %% Parameter setting
    [delta,pa,nr,rate,C,WindowSize,D] = Global.ParameterSet(0.9,0.25,2,1,5,ceil(Global.N/2),1);
    %% Operators and Operator List
    % grw
    alpha = 1.5;
    sigY1 = gamma(1+alpha)*sin(alpha*pi/2);
    sigY2 = gamma((1+alpha)/2)*alpha*2^((alpha-1)/2);
    sigY = (sigY1/sigY2)^(1/alpha);
    
    getStepSize = @(sz) normrnd(0,1,sz)./(abs(normrnd(0,sigY,sz)).^(1./alpha));
    grw = @(Decsi, Decsj) Decsi + rate.*getStepSize(size(Decsi)).*(Decsj-Decsi);
    % lrw
    lrw = @(Decsi, Decsj, Decsk) Decsi + rand(size(Decsi)).*heaviside(pa-rand(size(Decsi))).*(Decsj-Decsk);
    % list
    opList = {grw,lrw};

    %% Generate the weight vectors
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    T = ceil(Global.N/10);

    %% Detect the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);

    %% Generate random population
    Population = Global.Initialization();
    Z = min(Population.objs,[],1);
    [~,FrontNo,~] = EnvironmentalSelection(Population,Global.N);
    Zmax = max(Population(FrontNo==1).objs,[],1);
    disp(Population);
    
    %% Optimization
    FRR = zeros(1,2);	% Credit value of each operator
    SW  = zeros(2,WindowSize);	% Sliding window
    
    %% Calculate max angles per weights (psi in the paper)
    angles = ComputeAngles(Population.objs-Z,W);
    psi = max(angles,[],1);
    while Global.NotTermination(Population)
        
        % For each solution
        for i = 1 : Global.N
            % Choose the parents
            if rand < delta
                P = B(i,randperm(end));
            else
                P = randperm(Global.N);
            end
            
            j = randsample(P(P~=i), 1);
            k = randsample(P(and((P~=i),(P~=j))),1);
            
            Xi = Population(i).dec;
            Xj = Population(j).dec;
            Xk = Population(k).dec;
            
            opIdx = FRRMAB(FRR,SW,C);
            op = opList{opIdx};
            
            if opIdx == 1
                X = op(Xi, Xj);
                % Polynomial Mutation
                [proM, disM] = deal(1,20);
                [Lower, Upper] = deal(Global.lower, Global.upper);
                Site  = rand([1,Global.D]) < proM/Global.D;
                mu    = rand([1,Global.D]);
                temp  = Site & mu<=0.5;
                X       = min(max(X,Lower),Upper);
                X(temp) = X(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                                  (1-(X(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
                temp = Site & mu>0.5; 
                X(temp) = X(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                                  (1-(Upper(temp)-X(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))); 
            else
                X = op(Xi, Xj, Xk);
            end
            
             
            Offspring = INDIVIDUAL(X);
            
            % Update the ideal point
            Z = min(Z,Offspring.obj);

            % Update the solutions in P by Tchebycheff approach
%             g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
%             g_new = max(repmat(abs(Offspring.obj-Z),length(P),1).*W(P,:),[],2);
%                     Tchebycheff approach with normalization
%             Zmax  = max(Population.objs,[],1);
            g_old = max(abs(Population(P).objs-repmat(Z,length(P),1))./repmat(Zmax-Z,length(P),1).*W(P,:),[],2);
            g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),length(P),1).*W(P,:),[],2);
            gmma = (g_old - g_new)./g_old;
            
            psiOld = psi(P);
            psiNew = ComputeAngles(Offspring.obj, W(P,:));
            dt = psiOld - psiNew;
            
            %% Replace Population with solution if better
            I1 = gmma>0;
            I2 = transpose(dt>0);
            I = and(I1, I2);
            replace = I;
            Population(P(replace)) = Offspring;
            
            %% Update credit
            FIR = sum((g_old(replace)-g_new(replace))./g_old(replace));
            SW  = [SW(1,2:end),opIdx;SW(2,2:end),FIR];
            FRR = CreditAssignment(SW,D);
        end
        angles = ComputeAngles(Population.objs-Z,W);
        psi = max(angles,[],1);
        [~,FrontNo,~] = EnvironmentalSelection(Population,Global.N);
        Zmax = max(Population(FrontNo==1).objs,[],1);    
    end
end