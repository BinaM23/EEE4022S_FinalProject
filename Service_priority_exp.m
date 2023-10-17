%Experiment 2: Varied service priority
%Author: Bina E Mukuyamba
%Date: 17/10/2023

%Experiment was conducted twice, 1st time using static network conditions
%Repeated using dynamic network conditions

%NB the terms vectors and arrays will be used interchangably
%   treat the matrices as vectors of vectors
%all the implementation carried out in run_experiment method
%input the number of users to simulate and priority of each service
numUsers=input("enter number of users to simulate"); %used 1000 for the report
mydata1=run_experiment(numUsers,5,1,3); %simulate scenario 1...
mydata2=run_experiment(numUsers,5,3,1);
mydata3=run_experiment(numUsers,1,5,3);
mydata4=run_experiment(numUsers,3,5,1);
mydata5=run_experiment(numUsers,1,3,5);
mydata6=run_experiment(numUsers,3,1,5);
mydata7=run_experiment(numUsers,5,1,1);
mydata8=run_experiment(numUsers,1,5,1);
mydata9=run_experiment(numUsers,1,1,5);
mydata10=run_experiment(numUsers,5,5,5); %simulate scenario 10

%create bar charts for plotting
mybar=bar(mydata1);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 1: voice=5 video=1 web browsing=3")
mybar2=bar(mydata2);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 2: voice=5 video=3 web browsing=1")

mybar3=bar(mydata3);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 3: voice=1 video=5 web browsing=3")

mybar4=bar(mydata4);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 4: voice=3 video=5 web browsing=1")

mybar5=bar(mydata5);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 5: voice=1 video=3 web browsing=5")

mybar6=bar(mydata6);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 6: voice=3 video=1 web browsing=5")
mybar7=bar(mydata7);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 7: voice=5 video=1 web browsing=1")

mybar8=bar(mydata8);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 8: voice=1 video=5 web browsing=1")

mybar9=bar(mydata9);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 9: voice=1 video=1 web browsing=5")

mybar10=bar(mydata10);
xlabel("RAT-N");
ylabel("No of users(MMTs) admitted");
title("Scenario 10: voice=5 video=5 web browsing=5")

%% 
% 

%The following are the utility functions for each
%criterion as defined in Table 3 in [11]
%each returns a single utility value u
%for a particular service and performance rating x
%%
function u = f_x(service,x)
e=exp(1);
switch service
    case "voice"
    a=0.25;b=48;
    u=1/(1.0+e^(-a*(x-b)));
    case "video"
    a=0.003;b=2000;
    u=1/(1.0+e^(-a*(x-b)));
    otherwise %web browsing
    a=0.01;b=564;
    u=1/(1.0+e^(-a*(x-b)));

end

end
function u = g_x(service,x)
e=exp(1);
switch service
    case "voice"
    a=0.1;b=75;
    u=1-(1/(1.0+e^(-a*(x-b))));
    case "video"
    a=0.1;b=112.5;
    u=1-(1/(1.0+e^(-a*(x-b))));
    otherwise %web browsing
    a=0.03;b=375;
    u=1-(1/(1.0+e^(-a*(x-b))));

end

end
function u = h_x(service,x)
switch service
    case "voice"
    g=1/30.0;
    u=1-(g*x);
    case "video"
    g=1/30.0;
    u=1-(g*x);
    otherwise %web browsing
    g=1/30.0;
    u=1-(g*x);

end

end
%updated
function u = h_x2(service,x)
switch service
    case "voice"
    g=1/50.0;
    u=1-(g*x);
    case "video"
    g=1/50.0;
    u=1-(g*x);
    otherwise %web browsing
    g=1/50.0;
    u=1-(g*x);

end

end

%function to assign weights
%Function returns a 1x3 cell which consists of vectors of user defined weights
%for each service
% each element of the cell corresponds to a different service
%assume w{1}=voice weights for user i
function w = assignWeights(N) %N = number of users you want

w=cell(1,N); %creates 1xN cell where each row = 1, column= N 1x3 sets of user weights for user i
weights=[1,2,3,4,5,6,7,8,9];
for i =1:length(w) %for N users
    for j = 1:3 % loop through weight vector for each service (x3 per user)
            arr=zeros(1,4);
            for k=1:4 %loop through weight for each criterion (x4 per weight vector)
            arr(k)=randi([weights(1),weights(9)]);
            end 
            users_i_Weights{j}=arr; %assign each completed vector to set of vectors
    end
    w{i}=users_i_Weights; %each user-i will have 3 sets of randomized weights
end

end

%returns normalized weights for a cell of user weight vectors
%returns a 1x3 cell consisting of vectors of normalized user weights for each
%service for each user
function Norm = Normalize(userCell) 
userCellSize=size(userCell);
Norm=cell(1,userCellSize(2)); %should be the same size as input cell
for i =1:length(Norm) %for N users
    for j = 1:3 % loop through weight vector for each service (x3 per user)
            arr=zeros(1,4);
            for k=1:4 %loop through weight for each criterion (x4 per weight vector)
            arr(k)=userCell{i}{j}(k)/sum(userCell{i}{j});%normalize using equation
            end 
            users_i_Weights{j}=arr; %assign each completed vector to set of vectors
    end
    Norm{i}=users_i_Weights; %each user-i will have 3 sets of normalized weights
end

end

%Function that calculates Wu for N users
%inputs normalized weights (cell) for all users, normalized user specified priority vector
%returns a  1xN cell of Wu vectors for all users
function WuVec = calculate_Wu(normalizedWeights,Puvec)
normSize=size(normalizedWeights);
WuVec=cell(1,normSize(2)); %store Wu vector for all users
for i=1:normSize(2) %find weight user weight vector for each user i
%wunormvec=the weight vectors for each service for user i placed in one
%vector for easier calculation
wunormvec=[normalizedWeights{i}{1};normalizedWeights{i}{2};normalizedWeights{i}{3}]; %each row is the normalized weight vector for each service 
Wu=zeros(1,4); %Wu user specified weight vector for group of services
total=0;
for j=1:4
    colvec=wunormvec(:,j); %break down the rieman sum into steps
    for g=1:3 %loop through services
        total=total+(Puvec(g)*colvec(g)); %sum of Pu_norm * uij^g
    end
    Wu(j)=total;
    total=0;
end
WuVec{i}=Wu; %assign calculated Wu vector to user i
end
end

%this function uses TOPSIS to calculate the scores of each RAT
%for all users
%returns a 1XN cell of vectors of RAT scores for each user
function all_RAT_scores_per_user = TOPSIS(Decision_matrices)
num=size(Decision_matrices);
all_RAT_scores_per_user=cell(1,num(2));
for u = 1:num(2) %calculate score for all u users
    D=Decision_matrices{u}; %each user has unique D
    %TOPSIS code
    Dplus=zeros(1,4);
    Dminus=zeros(1,4);
    %store the ideal solutions D+ and D-
    for i=1:4
    colvec=D(:,i);
        if i<=2
        Dplus(i)=max(colvec);
        Dminus(i)=min(colvec);
        end
        if i>2
        Dplus(i)=min(colvec);
        Dminus(i)=max(colvec);
        end
    
    end
    %calculate euclidean distances Si+ and Si-
    %for each RAT to ideal solution
    Siplus=zeros(1,4);
    Siminus=zeros(1,4);%vectors to store each Si value
    tally1=0;
    tally2=0;
    for i=1:4
        for j=1:4
        tally1=tally1+(Dplus(j)-D(i,j))^2; %sum of squared diff
        tally2=tally2+(D(i,j)*Dminus(j))^2; %sum of products (22)
        end
        Siplus(i)=sqrt(tally1);
        Siminus(i)=sqrt(tally2);
        tally1=0;
        tally2=0;
    end
    SC=zeros(1,4); %array to store each RATs score
    for i=1:4
        SC(i)=Siminus(i)/(Siminus(i)+Siplus(i));
    end
    all_RAT_scores_per_user{u}=SC; %assign array of RAT scores for user i
end
end

%This function calculates the best RAT for N users 
%input:a cell consisting of the vectors of each users RAT scores
%output: 1xN cell of the best RAT for each user
function best_RAT_per_user = UserRats(RATscores)
num=size(RATscores);
best_RAT_per_user=cell(1,num(2));
for k=1:num(2) %find best RAT for all users 
    Best_RAT=0;
    UserRatScores=RATscores{k};
    for i=1:4
    score=UserRatScores(i);
    if score>=max(UserRatScores)
    Best_RAT=i;
    end
    end
    best_RAT_per_user{k}=Best_RAT;
end
end
%used to count the number of users admitted into each RAT
%returns an array of number of users admitted to each RAT
%input = 1xN array of the best RATs for N users
function p = plotRats(URarray)
RAT1=sum(URarray==1);
RAT2=sum(URarray==2);
RAT3=sum(URarray==3);
RAT4=sum(URarray==4);
p=[RAT1,RAT2,RAT3,RAT4];
end


%This function is used to run the entire algorithm
%In this case for varied service priority scenarios for N users
%inputs: N= number of users, p1=voice priority, p2=video priority, p3=web
%browsing priority
%returns array of the number of users admitted into each RAT
function data=run_experiment(N,p1,p2,p3)
%Defining the RATs
%RAT-N = [bandwidth,cost,delay,loss rate] are our criteria
%3G
RAT_1 =[1.350,3.5,30,6];
%4G
RAT_2=[2.400,4.5,60,13];
%WLAN
RAT_3=[4.500,0.5,80,10];
%5G
RAT_4=[1000,7,13,4];
%MODULE 1: WEIGHTING
%Attribute matrix= Mt
%Each row of the matrices corresponds to a particular RAT
%row_N = RAT-N, col_x = criterion_x value
Mt=[RAT_1;RAT_2;RAT_3;RAT_4];
%normalized net attribute matrix Mtbar normalizing using (2)
Mt_norm=zeros(4,4);
for j=1:4
    for i=1:4
        Mt_norm(i,j)=Mt(i,j)/sum(Mt(:,j));
    end
end
Mt_norm;

%ENTROPY calculation using (3)
Entropy_values=zeros(1,4);
k=-1/log(4); %constant
vecsum=0; %initializing
for j=1:4
    Mtvec=Mt_norm(:,j); %column vector
    for i=1:4
        vecsum=vecsum+sum(Mtvec(i)*log(Mtvec(i))); %intermediate sums
    end
    Entropy_values(j)=k*vecsum; %entropy value
    vecsum=0; %don't forget to reset the running total
end
Entropy_values;
%Objective weights calculation using (4)
ObjectiveWeights=zeros(1,4);

for j=1:4
    ObjectiveWeights(j)=(1-Entropy_values(j))/(4-sum(Entropy_values));
end
ObjectiveWeights;
users=assignWeights(N); % numUsers MMTs are created with random weights for each service
%This vector is changed for each experiment using p1,p2 and p3
Pu=[p1,p2,p3]; %user specified service priority vector, this will be the same for all MMTs
Ps=Pu; %service determined priority=user specified priority, will be manually changed

%normalizing weights and priority
NormedUweights=Normalize(users);

sp=sum(Pu);
Pu_norm=zeros(1,3);%normalized vector
for i=1:3
Pu_norm(i)=Pu(i)/sp;
end
Ps_norm=Pu_norm;%normalized prority vectors
%Wu user specified weight vector calculation
%each user will have unique Wu due to having unique weights per service
Wu_allUsers=calculate_Wu(NormedUweights,Pu_norm); %stores Wu for each user in a cell


%Service determined weights by FAHP
%Fuzzy comparison matrices for each service defined below(modify to be
%dynamic)
%update replaced security with cost for all matrices
%all elements which were for security were changed
FCM_voice=[0.5,0.5,0.1,0.4;0.5,0.5,0.1,0.4;0.9,0.9,0.5,0.8;0.6,0.6,0.2,0.5];
FCM_video=[0.5,0.7,0.3,0.75;0.3,0.5,0.1,0.55;0.7,0.9,0.5,0.95;0.25,0.45,0.05,0.5];
FCM_web=[0.5,0.5,0.75,0.4;0.5,0.5,0.75,0.4;0.25,0.25,0.5,0.15;0.6,0.6,0.85,0.5];
%weight calculation using (9)
wSvoice=zeros(1,4);
wSvideo=zeros(1,4);
wSweb=zeros(1,4); %vectors to store calculated weights of criteria per service
%tally=0;
n=4; %no of crietria
for i=1:4
    %voice
    k=(2*FCM_voice(i,1))^2; %2*ai1^k
    tally=sum(FCM_voice(i,:));%sum of row i
    soln=(tally+(n/2.0)-1)/(n*(n-1));
    wSvoice(i)=soln*k;
    tally=0;
    %video
    k2=(2*FCM_video(i,1))^2; %2*ai1^k
    tally2=sum(FCM_video(i,:));%sum of row i
    soln2=(tally2+(n/2.0)-1)/(n*(n-1));
    wSvideo(i)=soln2*k2;
    tally2=0;
    %web browsing
    k3=(2*FCM_web(i,1))^2; %2*ai1^k
    tally3=sum(FCM_web(i,:));%sum of row i
    soln3=(tally3+(n/2.0)-1)/(n*(n-1));
    wSweb(i)=soln3*k3;
    tally3=0;
end
%resulting weight vectors
wSvoice;
wSvideo;
wSweb;
%Ws service determined weight vector calculation
%Procedure is exactly the same as Wu calculation
wsnormvec=[wSvoice;wSvideo;wSweb]; %each row is the normalized weight vector for each service
Ps_norm;
Ws=zeros(1,4); %initialize 
total=0;
for j=1:4
    colvec=wsnormvec(:,j); %break down the rieman sum into steps
    for g=1:3
        total=total+(Ps_norm(g)*colvec(g));
    end
    Ws(j)=total;
    total=0;
end
%Ws stores the overall weight of each criterion for multiservice i.e. each
%service's weight for criterion cj  is combined
Ws;
%Comprehensive weight vector calculation
%Combines Wo(objective weights),Ws and Wu using weight proportion parameters
%alpha=a, beta=b, gamma=y
%W(comprehensive weights)=aWu+bWo+yWs
Wvec=cell(1,N); %vector to store comprehensive weights for each user
a=0.2;b=0.3;y=0.5; %updated based on experiment 1 results
ObjectiveWeights;
Ws;
for i=1:N
    W=(a*Wu_allUsers{i})+(b*ObjectiveWeights)+(y*Ws); %using formula for each user i
    Wvec{i}=W;
end
%MODULE 2: UTILITY
Mt;
Mt(:,1)=Mt(:,1)*1000; %in kbps for utility

%utility is calculated by substituing the performance rating from Mt into
%the appropriate utility function and constants

u1=zeros(4,4); %utility value matrices for each service (voice)
u2=zeros(4,4); %video
u3=zeros(4,4); %web browsing
%voice utility

for i=1:4
    colvec=Mt(:,i);%colum vector
    for j=1:4
    %colvec(j);
    if i==1
    u1(i,j)=f_x("voice",colvec(j));
    end
    if i==2
    u1(i,j)=h_x2("voice",colvec(j));    
    end
    if i==3
        u1(i,j)=g_x("voice",colvec(j)); %use appropriate formula per col
    end
    if i==4
        u1(i,j)=h_x("voice",colvec(j));
    end
   
    end
    
end
u1=transpose(u1);%get the elements the right way round
%video utility
for i=1:4
    colvec=Mt(:,i);%colum vector
    for j=1:4
    %colvec(j);
    if i==1
    u2(i,j)=f_x("video",colvec(j));
    end
    if i==2
    u2(i,j)=h_x2("video",colvec(j));    
    end
    if i==3
        u2(i,j)=g_x("video",colvec(j)); %use appropriate formula per col
    end
    if i==4
        u2(i,j)=h_x("video",colvec(j));
    end
    
    end
    
end
u2=transpose(u2);
%web browsing utility
for i=1:4
    colvec=Mt(:,i);%colum vector
    for j=1:4
    %colvec(j);
    if i==1
    u3(i,j)=f_x("web",colvec(j));
    end
    if i==2 %cost not security function 
    u3(i,j)=h_x2("web",colvec(j));    
    end
    if i==3
        u3(i,j)=g_x("web",colvec(j)); %use appropriate formula per col
    end
    if i==4
        u3(i,j)=h_x("web",colvec(j));
    end
    
    end
    
end
u3=transpose(u3);
%Comprehensive utility value matrix calculation
U=zeros(4,4);
uvec={u1,u2,u3}; %using Matlab matrices
%disp(uvec{1})

%3 way loop i,j and g vary
tally=0;
for i=1:4
    for j=1:4
        for g=1:3
            usomething=uvec{g}; %u_ij for a specific service g
            tally=tally+usomething(i,j)*Ps_norm(g); %break down the sum into parts
        end
         U(i,j)=tally;
         tally=0;

    end
end
U ;%final comprehensive utility value matrix
%MODULE 3: NETWORK RANKING AND SELECTION
%Create normalized decison matrix (NDM)
NDM=U;
%create weighted NDM for each user by multiplying each row of U with W for each user
Dvec=cell(1,N); %cell to store D for each user
for k=1:N
D=zeros(4,4);
W=Wvec{k};
for i=1:4
    for j=1:4
        D(i,j)=NDM(i,j)*W(j); %element of D
    end
end
Dvec{k}=D; %assigns decision matrix to user k
end
Dvec{1}; %for debugging
%Use TOPSIS to calculate the scores of all the RATs for each user
RAT_scores=TOPSIS(Dvec);

%selects and displays the best RAT for each user based on TOPSIS 
%    disp(RATs{1}) to check
RATs=UserRats(RAT_scores);

%converts the cell to an array for plotting
Best_RAT_array=cell2mat(RATs);
data=plotRats(Best_RAT_array);
end