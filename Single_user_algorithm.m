%RAT selection algorithm for a single user
%Author: Bina E Mukuyamba
%Date: 06/10/2023
%Implements the worked example in chapter 3 of report in code

%Update 20/10/2023
%changed TOPSIS equations
%No difference from using Extent analysis method

%Defining the RATs
%updated security with cost, bandwidth are in Mbps
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
Mt=[RAT_1;RAT_2;RAT_3;RAT_4]
%normalized net attribute matrix Mtbar normalizing using (2)
Mt_norm=zeros(4,4);
for j=1:4
    for i=1:4
        Mt_norm(i,j)=Mt(i,j)/sum(Mt(:,j))
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
    Entropy_values(j)=k*vecsum %entropy value
    vecsum=0; %don't forget to reset the running total
end
sum(Entropy_values)
%Objective weights calculation using (4)
ObjectiveWeights=zeros(1,4);

for j=1:4
    ObjectiveWeights(j)=(1-Entropy_values(j))/(4-sum(Entropy_values))
end
%Subjective user preference weights calculation
%example is for a single user but can be extended to 100-1000 MMTs
%assume 9 pt weight scale [1,9] user specified weights are randomized
%can't use 0
%create the user

wu1=[6,8,9,5]; %user defined weights for service 1(voice)
wu2=[9,5,6,8]; %user defined weights for service 2(video)
wu3=[6,9,9,6]; %user defined weights for service 3(web browsing)

Pu=[3,3,3]; %user specified service priority vector, this will be the same for all MMTs
Ps=Pu; %service determined priority=user specified priority, will be manually changed

%normalizing weights and priority
wu1norm=zeros(1,4); %normalized weight vectors
wu2norm=zeros(1,4);
wu3norm=zeros(1,4);
s1=sum(wu1);
s2=sum(wu2);
s3=sum(wu3);
for i=1:4
    wu1norm(i)=wu1(i)/s1
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
wunormvec=[wu1norm;wu2norm;wu3norm] %each row is the normalized weight vector for each service 
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
Wu
%======================================================================
%OLD METHOD NOT USED ANYMORE, used extent analysis instead
%see FAHP_EAM.m
%Service determined weights by AHP
%Fuzzy comparison matrices for each service defined below(modify to be
%dynamic)
%update replaced security with cost for all matrices
%all elements which were for security were changed
%FCM_voice=[0.5,0.5,0.1,0.4;0.5,0.5,0.1,0.4;0.9,0.9,0.5,0.8;0.6,0.6,0.2,0.5]
%FCM_video=[0.5,0.7,0.3,0.75;0.3,0.5,0.1,0.55;0.7,0.9,0.5,0.95;0.25,0.45,0.05,0.5]
%FCM_web=[0.5,0.5,0.75,0.4;0.5,0.5,0.75,0.4;0.25,0.25,0.5,0.15;0.6,0.6,0.85,0.5]
%weight calculation using (9)
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
%wsnormvec=[wSvoice;wSvideo;wSweb] %each row is the normalized weight vector for each service
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
%Ws
%=====================================================================================
%run FAHP_EAM, with Pu=[3,3,3] copy results here
Ws=[0.3300,0.3240,0.2097,0.1363]

%Comprehensive weight vector calculation
%Combines Wo(objective weights),Ws and Wu using weight proportion parameters
%alpha=a, beta=b, gamma=y
%W(comprehensive weights)=aWu+bWo+yWs
a=0.2;b=0.5;y=0.3; %default value

Wu
ObjectiveWeights
Ws
W=(a*Wu)+(b*ObjectiveWeights)+(y*Ws);
W
%MODULE 2: UTILITY
Mt(:,1)=Mt(:,1)*1000 %in kbps for utility
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
u1=transpose(u1) %get the elements the right way round
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
u2=transpose(u2) 
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
u3=transpose(u3) 
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
U %final comprehensive utility value matrix
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
D
%TOPSIS code
Dplus=zeros(1,4);
Dminus=zeros(1,4);
%store the ideal solutions D+ and D-
% update c2 is now a cost criterion !
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
Dplus
Dminus
%calculate euclidean distances Si+ and Si-
%for each RAT to ideal solution
Siplus=zeros(1,4);
Siminus=zeros(1,4);%vectors to store each Si value
tally1=0;
tally2=0;
for i=1:4
    for j=1:4
        %Used standard TOPSIS equations instead
    tally1=tally1+(D(i,j)-Dplus(j))^2; %sum of squared diff
    tally2=tally2+(D(i,j)-Dminus(j))^2; 
    end
    Siplus(i)=sqrt(tally1);
    Siminus(i)=sqrt(tally2);
    tally1=0;
    tally2=0;
end
Siplus %Si+
Siminus %Si-
% calculate the score of each RAT
SC=zeros(1,4); %array to store each RATs score
for i=1:4
    SC(i)=Siminus(i)/(Siminus(i)+Siplus(i));
end
SC

%The best RAT is the one with the highest score
Best_RAT=0;
for i=1:4
score=SC(i);
if score==max(SC)
    Best_RAT=i;
end
end

disp("The selected optimal RAT is: ")
disp(num2str(Best_RAT))
%%
%NB updated utility functions to include cost
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
%update
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
function Norm = Normalize(userCell) %returns normalized weights for a cell of user weight vectors
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