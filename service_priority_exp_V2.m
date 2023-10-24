%RAT selection algorithm 
%Experiment 2:Varying priority
%Author: Bina E Mukuyamba
%Date: 24/10/2023

%Varying service priority alternate version
%each user detects different network conditions
%To simulate a dynamic network environment
%Ws FAHP weights are calculated in another program for each scenario then
%substituted into this program
%can make multiple of these to store results for each scenario

%Run_experiment(P1,P2,P3,FAHP_weights,NumUsers)
%P1,P2,P3 = priorities for each service
%FAHP_weights = service determined weights Ws
%NumUsers=Number of users to simulate, 1000 for experiment
%Ws_i values found by running FAHP_EAM.m using appropriate [P1,P2,P3]
%values then paste results in appropriate variable

Ws_1=[0.1980,0.4837,0.1820,0.1363]; %Ws for scenario 1 [5,1,3]
Ws_2=[0.2614,0.4274,0.2657,0.0454]; %Ws for scenario 2 [5,3,1]
Ws_3=[0.4620,0.1643,0.2374,0.1363]; %Ws for scenario 3 [1,5,3]
Ws_4=[0.3934,0.2677,0.2934,0.0454]; %Ws for scenario 4 [3,5,1]
Ws_5=[0.3985,0.2206,0.1536,0.2272]; %Ws for scenario 5 [1,3,5]
Ws_6=[0.2665,0.3803,0.1259,0.2272]; %Ws for scenario 6 [3,1,5]
Ws_7=[0.1581,0.5495,0.2340,0.0584]; %Ws for scenario 7 [5,1,1]
Ws_8=[0.4975,0.1389,0.3052,0.0584]; %Ws for scenario 8 [1,5,1]
Ws_9=[0.3344,0.2836,0.0899,0.2922]; %Ws for scenario 9 [1,1,5]
Ws_10=[0.3300,0.3240,0.2097,0.1363]; %Ws for scenario 10 [5,5,5]

NumUsers=input("Enter number of users to simulate")
mydata1=Run_experiment(5,1,3,Ws_1,NumUsers); %scenario1
mydata2=Run_experiment(1,5,3,Ws_2,NumUsers);
mydata3=Run_experiment(1,5,3,Ws_3,NumUsers);
mydata4=Run_experiment(3,5,1,Ws_4,NumUsers);
mydata5=Run_experiment(1,3,5,Ws_5,NumUsers);
mydata6=Run_experiment(3,1,5,Ws_6,NumUsers);
mydata7=Run_experiment(5,1,1,Ws_7,NumUsers);
mydata8=Run_experiment(1,5,1,Ws_8,NumUsers);
mydata9=Run_experiment(1,1,5,Ws_9,NumUsers);
mydata10=Run_experiment(5,5,5,Ws_10,NumUsers); %scenario 10

%plot results for all scenarios
Big_Data=[mydata1;mydata2;mydata3;mydata4;mydata5;mydata6;mydata7;mydata8;mydata9;mydata10];
x=1:10;
bar(x,Big_Data);
xlabel("Scenario")
legend("3G","4G","WLAN","5G")
ylabel("No of users(MMTs) admitted")
title("Number of MMTs admitted into each RAT per scenario")

%plot results scenario by scenario
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

%NB updated utility functions to include cost
%Utility functions as defined in table 3 in [11]
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

%Not used in this program
%Function to calculate convex degree of possibility
%returns degree of possibility between 2 TFNs
%inputs 2 vectors where S1=(l1,m1,u1) and S2=(l2,m2,u2)
%output Vvalue= 0,1 or some value inbetween
function Vvalue = M2greaterM1(M1,M2)
l1=M1(1);m1=M1(2);u1=M1(3);
l2=M2(1);m2=M2(2);u2=M2(3);
if (m2>=m1)
    Vvalue=1;
else
if (l1>=u2)
        Vvalue=0;
else
Vvalue=(l1-u2)/((m2-u2)-(m1-l1));
end
end
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
%updated for cost
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
function w = assignWeights(N) %N = number of users you want
%w=zeros(N,3); %w is a vector of 3 weight vectors for each user
%assume each element of the cell corresponds to a different service
%assume w{1}=voice weights for user i
w=cell(1,N); %creates 1xN cell where each row = 1, column= N 1x3 sets of user weights for user i
%basically a 1D array of 1x3 columns
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
function Norm = Normalize(userCell) 
userCellSize=size(userCell);
Norm=cell(1,userCellSize(2)); %should be the same size as input cell
for i =1:length(Norm) %for N users
    for j = 1:3 % loop through weight vector for each service (x3 per user)
            arr=zeros(1,4);
            for k=1:4 %loop through weight for each criterion (x4 per weight vector)
            arr(k)=userCell{i}{j}(k)/sum(userCell{i}{j});
            end 
            users_i_Weights{j}=arr; %assign each completed vector to set of vectors
    end
    Norm{i}=users_i_Weights; %each user-i will have 3 sets of normalized weights
end

end
%Not used in this program
function T = find_gain(Umat,alpha,beta,gamma,WU,WS,WO)
%a,b,y are varied to determine the best gain for our model
%using (30) from [11]
Wvec=(alpha*WU)+(beta*WO)+(gamma*WS);
gain=zeros(1,4);
total=0;
for i=1:4
    for j=1:4
        total=total+(Umat(i,j)*Wvec(j));
    end
    gain(i)=total;
    total=0;
end
T=sum(gain);

end

%Function to run experiment 2
%Implements all the above functions to calculate weights and unity
%returns an array which stores the number of users in each RAT after
%running the selections for 1000 users
%inputs P1,P2,P3 = service priorities
%FAHP_Weights = Ws for a particular priority vector [P1,P2,P3]
%Num_Users = Number of users to simulate
%Output The_RATS = array to store number of users per RAT
function The_RATS = Run_experiment(P1,P2,P3,FAHP_weights,NumUsers)
The_RATS=zeros(1,4); %array to store number of users in each RAT
for iterations=1:NumUsers %no of users to simulate

%Defining the RATs
%Dynamically changing values
%net attr will be randomly selected from table 2 for each iteration/user
%NB cost is only criterion that will remain the same
%RAT-N = [bandwidth,cost,delay,loss rate]
%3G
RAT_1 =[(randi([7,20]))/10,3.5,randi([10,50]),randi([2,10])]; %divide by 10 to generate random decimals
%4G
RAT_2=[(randi([8,1000]))/10,4.5,randi([40,80]),randi([6,20])];
%WLAN
RAT_3=[randi([1,100]),0.5,randi([70,90]),randi([4,15])];
%5G
RAT_4=[randi([100,1000]),7,randi([1,25]),(randi([1,80]))/10];

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
sum(Entropy_values);
%Objective weights calculation using (4)
ObjectiveWeights=zeros(1,4);

for j=1:4
    ObjectiveWeights(j)=(1-Entropy_values(j))/(4-sum(Entropy_values));
end
%Subjective user preference weights calculation
%assume 9 pt weight scale [1,9] user specified weights are randomized
weights=[1,2,3,4,5,6,7,8,9];

wu1=[randi([weights(1),weights(9)]),randi([weights(1),weights(9)]),randi([weights(1),weights(9)]),randi([weights(1),weights(9)])]; %user defined weights for service 1(voice)
wu2=[randi([weights(1),weights(9)]),randi([weights(1),weights(9)]),randi([weights(1),weights(9)]),randi([weights(1),weights(9)])]; %user defined weights for service 2(video)
wu3=[randi([weights(1),weights(9)]),randi([weights(1),weights(9)]),randi([weights(1),weights(9)]),randi([weights(1),weights(9)])]; %user defined weights for service 3(web browsing)

%Now a variable passed into function
Pu=[P1,P2,P3]; %user specified service priority vector
Ps=Pu; %service determined priority=user specified priority, will be manually changed

%normalizing weights and priority
wu1norm=zeros(1,4); %normalized weight vectors
wu2norm=zeros(1,4);
wu3norm=zeros(1,4);
s1=sum(wu1);
s2=sum(wu2);
s3=sum(wu3);
for i=1:4
    wu1norm(i)=wu1(i)/s1;
    wu2norm(i)=wu2(i)/s2;
    wu3norm(i)=wu3(i)/s3;
end
sp=sum(Pu);
Pu_norm=zeros(1,3);%normalized vector
for i=1:3
Pu_norm(i)=Pu(i)/sp;
end
Ps_norm=Pu_norm;%normalized prority vectors
%Wu user specified weight vector calculation
wunormvec=[wu1norm;wu2norm;wu3norm]; %each row is the normalized weight vector for each service 
Wu=zeros(1,4);
total=0;
for j=1:4
    colvec=wunormvec(:,j); %break down the rieman sum into steps
    for g=1:3
        total=total+(Pu_norm(g)*colvec(g));
    end
    Wu(j)=total;
    total=0;
end
Wu;
%======================================================================
%This was an old method used to determine Ws, not used anymore used Extent
%analysis instead, see FAHP_EAM.m
%Fuzzy comparison matrices for each service defined below
%update replaced security with cost for all matrices
%all elements which were for security were changed
%FCM_voice=[0.5,0.5,0.1,0.4;0.5,0.5,0.1,0.4;0.9,0.9,0.5,0.8;0.6,0.6,0.2,0.5];
%FCM_video=[0.5,0.7,0.3,0.75;0.3,0.5,0.1,0.55;0.7,0.9,0.5,0.95;0.25,0.45,0.05,0.5];
%FCM_web=[0.5,0.5,0.75,0.4;0.5,0.5,0.75,0.4;0.25,0.25,0.5,0.15;0.6,0.6,0.85,0.5];
%%weight calculation using (9)
%wSvoice=zeros(1,4);
%wSvideo=zeros(1,4);
%wSweb=zeros(1,4); %vectors to store calculated weights of criteria per service
%%tally=0;
%n=4; %no of crietria
%for i=1:4
%    %voice
%    k=(2*FCM_voice(i,1))^2; %2*ai1^k
%    tally=sum(FCM_voice(i,:));%sum of row i
%    soln=(tally+(n/2.0)-1)/(n*(n-1));
%    wSvoice(i)=soln*k;
%    tally=0;
%    %video
%    k2=(2*FCM_video(i,1))^2; %2*ai1^k
%    tally2=sum(FCM_video(i,:));%sum of row i
%    soln2=(tally2+(n/2.0)-1)/(n*(n-1));
%    wSvideo(i)=soln2*k2;
%    tally2=0;
%    %web browsing
%    k3=(2*FCM_web(i,1))^2; %2*ai1^k
%    tally3=sum(FCM_web(i,:));%sum of row i
%    soln3=(tally3+(n/2.0)-1)/(n*(n-1));
%    wSweb(i)=soln3*k3;
%    tally3=0;
%end
%%resulting weight vectors
%wSvoice;
%wSvideo;
%wSweb;
%%Ws service determined weight vector calculation
%%Procedure is exactly the same as Wu calculation
%wsnormvec=[wSvoice;wSvideo;wSweb]; %each row is the normalized weight vector for each service
%Ps_norm;
%Ws=zeros(1,4); %initialize 
%total=0;
%for j=1:4
%    colvec=wsnormvec(:,j); %break down the rieman sum into steps
%    for g=1:3
%        total=total+(Ps_norm(g)*colvec(g));
%    end
%    Ws(j)=total;
%    total=0;
%end
%%Ws stores the overall weight of each criterion for multiservice i.e. each
%%service's weight for criterion cj  is combined
%Ws;
%============================================================================

Ws=FAHP_weights;
%Comprehensive weight vector calculation
%Combines Wo(objective weights),Ws and Wu using weight proportion parameters
%alpha=a, beta=b, gamma=y
%W(comprehensive weights)=aWu+bWo+yWs
a=0.2;b=0.5;y=0.3; %from Experiment 1
Wu;
ObjectiveWeights;
Ws;
W=(a*Wu)+(b*ObjectiveWeights)+(y*Ws);
W;

%MODULE 2: UTILITY
Mt(:,1)=Mt(:,1)*1000; %in kbps for utility
u1=zeros(4,4); %utility value matrices for each service (voice)
u2=zeros(4,4); %video
u3=zeros(4,4); %web browsing

%voice utility
%updated utility function for cost for all services
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
u1=transpose(u1); %get the elements the right way round

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
U; %final comprehensive utility value matrix
 
%MODULE 3: NETWORK RANKING AND SELECTION
%Create normalized decison matrix (NDM)
NDM=U;
%create weighted NDM by multiplying each row with W
D=zeros(4,4);
for i=1:4
    for j=1:4
        D(i,j)=NDM(i,j)*W(j);
    end
end
D;
%TOPSIS code
Dplus=zeros(1,4);
Dminus=zeros(1,4);
%store the ideal solutions D+ and D-
for i=1:4
    colvec=D(:,i);
        if i==1%update
        Dplus(i)=max(colvec);
        Dminus(i)=min(colvec);
        end
        if i>1 %update
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
        %Fixed TOPSIS
    tally1=tally1+(D(i,j)-Dplus(j))^2; %sum of squared diff
    tally2=tally2+(D(i,j)-Dminus(j))^2; %sum of products (22)
    end
    Siplus(i)=sqrt(tally1);
    Siminus(i)=sqrt(tally2);
    tally1=0;
    tally2=0;
end
%Use (23) to calculate the score of each RAT
SC=zeros(1,4); %array to store each RATs score
for i=1:4
    SC(i)=Siminus(i)/(Siminus(i)+Siplus(i));
end

%The best RAT is the one with the highest score
Best_RAT=0;
for i=1:4
score=SC(i);
if score==max(SC)
    Best_RAT=i; %assign RAT number of best RAT to variable
end
end
%count number of users in RAT 1
if (Best_RAT==1)
 The_RATS(1)=The_RATS(1)+1;
end
%count number of users in RAT 2
if (Best_RAT==2)
 The_RATS(2)=The_RATS(2)+1;
end
%count number of users in RAT 3
if (Best_RAT==3)
 The_RATS(3)=The_RATS(3)+1;
end
%count number of users in RAT 4
if (Best_RAT==4)
 The_RATS(4)=The_RATS(4)+1;
end

end %end of a single selection event for one user

The_RATS;

end