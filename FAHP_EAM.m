%Program to workout FAHP by extent analysis method
%Author Bina Mukuyamba
%Date:22/10/2023

%vary priority as needed
%Copied FCM from tables 6-9 in [11]
%Note order is different from tables in [11] due to order of criteria
%equal =(1,1,1) rather than (1,1,3)
%matrix=[BW,C,D,PLR]
%used cells for easier computation

%Fuzzy comparison matrices for each service
FCMvoice={[1,1,1],[1/8,1/6,1/4],[1/4,1/2,1],[1,2,4];[4,6,8],[1,1,1],[3,5,7],[5,7,9];[1,2,4],[1/7,1/5,1/3],[1,1,1],[1,3,5];
    [1/4,1/2,1],[1/9,1/7,1/5],[1/5,1/3,1],[1,1,1];}
FCMvideo={[1,1,1],[4,6,8],[1,2,4],[4,6,8];[1/8,1/6,1/4],[1,1,1],[1/5,1/3,1],[1,1,1];
    [1/4,1/2,1],[1,3,5],[1,1,1],[1,3,5];[1/8,1/6,1/4],[1,1,1],[1/5,1/3,1],[1,1,1];}
FCMwebBrowsing={[1,1,1],[1,2,4],[3,5,7],[1/4,1/2,1];[1/4,1/2,1],[1,1,1],[2,4,6],[1/5,1/3,1];
    [1/7,1/5,1/3],[1/6,1/4,1/2],[1,1,1],[1/8,1/6,1/4];[1,2,4],[1,3,5],[4,6,8],[1,1,1];}
N=4; %4 RATs
wSvoice=zeros(1,4);
wSvideo=zeros(1,4);
wSweb=zeros(1,4); %vectors to store calculated weights of criteria per service
Pu=[2,2,2] %user specified service priority vector, this will be the same for all MMTs
Ps=Pu;
sp=sum(Pu);
Pu_norm=zeros(1,3);%normalized vector
for i=1:3
Pu_norm(i)=Pu(i)/sp;
end
Ps_norm=Pu_norm;

%computation for voice weights
rowsum=cell(1,4);
for i=1:4
    ai=FCMvoice(i,:); %work out sum of TFN for each row
    rowsum{i}=[ai{1}(1)+ai{2}(1)+ai{3}(1)+ai{4}(1),ai{1}(2)+ai{2}(2)+ai{3}(2)+ai{4}(2),ai{1}(3)+ai{2}(3)+ai{3}(3)+ai{4}(3)]
end
rowsum; %stores sum of TFN(1x3 vector) for all 4 rows/criteria
colsum = zeros(1, 3);
% Iterate through the cell array and sum the elements
for i = 1:numel(rowsum)
   currentVector = rowsum{i}
   colsum = colsum + currentVector
end
colsum;
colsum=flip(colsum); %calculate the inverse by rearranging order of TFN
inverseTFN=[1/colsum(1),1/colsum(2),1/colsum(3)]; %take reciprocal to get inverse
%calculate Si values
Si_values=cell(1,4); %store Si values
for k=1:4
    Si_values{k}=rowsum{k}.*inverseTFN; %si=row i sum * inverse TFN
end
Si_values
S1=Si_values{1};
S2=Si_values{2};
S3=Si_values{3};
S4=Si_values{4};
%Calculate degree of possibility
%deviated from equation (10)
degreesOfpossibility=cell(4,3);
%S1 comparisions
%M2greaterM1(M1,M2) = solves for M2>M1
degreesOfpossibility{1,1}=M2greaterM1(S2,S1); % Solves S1>S2
degreesOfpossibility{1,2}=M2greaterM1(S3,S1); % Solves S1>S3
degreesOfpossibility{1,3}=M2greaterM1(S4,S1); %Solves S1>S4
%S2 comparisions
degreesOfpossibility{2,1}=M2greaterM1(S1,S2); % Solves S2>S1
degreesOfpossibility{2,2}=M2greaterM1(S3,S2); % Solves S2>S3
degreesOfpossibility{2,3}=M2greaterM1(S4,S2); % Solves S2>S4
%S3 comparisions
degreesOfpossibility{3,1}=M2greaterM1(S1,S3); % Solves S3>S1
degreesOfpossibility{3,2}=M2greaterM1(S2,S3); % Solves S3>S2
degreesOfpossibility{3,3}=M2greaterM1(S4,S3); % Solves S3>S4
%S4 comparisions
degreesOfpossibility{4,1}=M2greaterM1(S1,S4); % Solves S4>S1
degreesOfpossibility{4,2}=M2greaterM1(S2,S4); % Solves S4>S2
degreesOfpossibility{4,3}=M2greaterM1(S3,S4); % Solves S4>S3

degreesOfpossibility
%calculate degree of possibility for convex Si being greater than k TFNs
%find the minimum from each group of comparisons
minVs=zeros(1,4);
for o=1:4
    row=cell2mat(degreesOfpossibility(o,:));
    minVs(o)=min(row);
end
minVs
%Normalize the values in MinVs to obatin weight vector
tot=sum(minVs);
for x=1:4
    wSvoice(x)=minVs(x)/tot;
end
wSvoice
%Repeat the same process for video
%computation for video weights
rowsum=cell(1,4);
for i=1:4
    ai=FCMvideo(i,:); %work out sum of TFN for each row
    rowsum{i}=[ai{1}(1)+ai{2}(1)+ai{3}(1)+ai{4}(1),ai{1}(2)+ai{2}(2)+ai{3}(2)+ai{4}(2),ai{1}(3)+ai{2}(3)+ai{3}(3)+ai{4}(3)];
end
rowsum; %stores sum of TFN(1x3 vector) for all 4 rows/criteria
colsum = zeros(1, 3);
%% Iterate through the cell array and sum the elements
for i = 1:numel(rowsum)
   currentVector = rowsum{i};
   colsum = colsum + currentVector;
end
colsum;
colsum=flip(colsum);
inverseTFN=[1/colsum(1),1/colsum(2),1/colsum(3)];
%calculate Si values
Si_values=cell(1,4);
for k=1:4
    Si_values{k}=rowsum{k}.*inverseTFN;
end
Si_values;
S1=Si_values{1};
S2=Si_values{2};
S3=Si_values{3};
S4=Si_values{4};
%Calculate degree of possibility
%deviated from equation (10)
degreesOfpossibility=cell(4,3);
%S1 comparisions
%M2greaterM1(M1,M2) = solves for M2>M1
degreesOfpossibility{1,1}=M2greaterM1(S2,S1); % Solves S1>S2
degreesOfpossibility{1,2}=M2greaterM1(S3,S1); % Solves S1>S3
degreesOfpossibility{1,3}=M2greaterM1(S4,S1); %Solves S1>S4
%S2 comparisions
degreesOfpossibility{2,1}=M2greaterM1(S1,S2); % Solves S2>S1
degreesOfpossibility{2,2}=M2greaterM1(S3,S2); % Solves S2>S3
degreesOfpossibility{2,3}=M2greaterM1(S4,S2); % Solves S2>S4
%S3 comparisions
degreesOfpossibility{3,1}=M2greaterM1(S1,S3); % Solves S3>S1
degreesOfpossibility{3,2}=M2greaterM1(S2,S3); % Solves S3>S2
degreesOfpossibility{3,3}=M2greaterM1(S4,S3); % Solves S3>S4
%S4 comparisions
degreesOfpossibility{4,1}=M2greaterM1(S1,S4); % Solves S4>S1
degreesOfpossibility{4,2}=M2greaterM1(S2,S4); % Solves S4>S2
degreesOfpossibility{4,3}=M2greaterM1(S3,S4); % Solves S4>S3

degreesOfpossibility;
%calculate degree of possibility for convex Si being greater than k TFNs
%find the minimum from each group of comparisons
minVs=zeros(1,4);
for o=1:4
    row=cell2mat(degreesOfpossibility(o,:));
    minVs(o)=min(row);
end
minVs;
%Normalize the values in MinVs to obatin weight vector
tot=sum(minVs);
for x=1:4
    wSvideo(x)=minVs(x)/tot;
end
wSvideo;
%repeat for web browsing
%computation for web browsing
rowsum=cell(1,4);
for i=1:4
    ai=FCMwebBrowsing(i,:); %work out sum of TFN for each row
    rowsum{i}=[ai{1}(1)+ai{2}(1)+ai{3}(1)+ai{4}(1),ai{1}(2)+ai{2}(2)+ai{3}(2)+ai{4}(2),ai{1}(3)+ai{2}(3)+ai{3}(3)+ai{4}(3)];
end
rowsum; %stores sum of TFN(1x3 vector) for all 4 rows/criteria
colsum = zeros(1, 3);
%% Iterate through the cell array and sum the elements
for i = 1:numel(rowsum)
   currentVector = rowsum{i};
   colsum = colsum + currentVector;
end
colsum;
colsum=flip(colsum);
inverseTFN=[1/colsum(1),1/colsum(2),1/colsum(3)];
%calculate Si values
Si_values=cell(1,4);
for k=1:4
    Si_values{k}=rowsum{k}.*inverseTFN;
end
Si_values;
S1=Si_values{1};
S2=Si_values{2};
S3=Si_values{3};
S4=Si_values{4};
%Calculate degree of possibility
%deviated from equation (10)
degreesOfpossibility=cell(4,3);
%S1 comparisions
%M2greaterM1(M1,M2) = solves for M2>M1
degreesOfpossibility{1,1}=M2greaterM1(S2,S1); % Solves S1>S2
degreesOfpossibility{1,2}=M2greaterM1(S3,S1); % Solves S1>S3
degreesOfpossibility{1,3}=M2greaterM1(S4,S1); %Solves S1>S4
%S2 comparisions
degreesOfpossibility{2,1}=M2greaterM1(S1,S2); % Solves S2>S1
degreesOfpossibility{2,2}=M2greaterM1(S3,S2); % Solves S2>S3
degreesOfpossibility{2,3}=M2greaterM1(S4,S2); % Solves S2>S4
%S3 comparisions
degreesOfpossibility{3,1}=M2greaterM1(S1,S3); % Solves S3>S1
degreesOfpossibility{3,2}=M2greaterM1(S2,S3); % Solves S3>S2
degreesOfpossibility{3,3}=M2greaterM1(S4,S3); % Solves S3>S4
%S4 comparisions
degreesOfpossibility{4,1}=M2greaterM1(S1,S4); % Solves S4>S1
degreesOfpossibility{4,2}=M2greaterM1(S2,S4); % Solves S4>S2
degreesOfpossibility{4,3}=M2greaterM1(S3,S4); % Solves S4>S3

degreesOfpossibility;
%calculate degree of possibility for convex Si being greater than k TFNs
%find the minimum from each group of comparisons
minVs=zeros(1,4);
for o=1:4
    row=cell2mat(degreesOfpossibility(o,:));
    minVs(o)=min(row);
end
minVs;
%Normalize the values in MinVs to obatin weight vector
tot=sum(minVs);
for x=1:4
    wSweb(x)=minVs(x)/tot;
end
wSweb;
wsnormvec=[wSvoice;wSvideo;wSweb]; %each row is the normalized weight vector for each service
Ws=zeros(1,4);
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
Ws

%Function to calculate convex degree of possibility
%returns degree of possibility between 2 TFNs
%inputs 2 vectors where S1=(l1,m1,u1) and S2=(l2,m2,u2)
%output Vvalue= 0,1 or some value inbetween

%% 
% 
% 
% 

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