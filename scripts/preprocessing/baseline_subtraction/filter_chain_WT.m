cd 'C:\Users\1501020012\Desktop\signals'   % change this with the location of dataset files in your computer
d=dir('*.csv');   % dir struct of all pertinent .csv files
n=length(d);        % how many there were
data=cell(1,n);
filtered_data = cell(1,n);% preallocate a cell array to hold results

for i=1:n
data(i)={csvread(d(i).name, 1, 0)};  % read each file

a = cell2mat(data(1,i));
b = a(:,1);
ecg_filtered = sgolayfilt(b,4,11);

Wn= [5,14]/(290/2);
[b,a]=butter(4,Wn,'bandpass');

filteredECG=filtfilt(b,a,ecg_filtered);

N=8;
[C,L]=wavedec(filteredECG,N,'db4');
[thr,sorh,keepapp]=ddencmp('den','wv',filteredECG); 
filteredECG = wdencmp('gbl',C,L,'db4',8,thr,sorh,keepapp);

filtered_data(i) = {filteredECG};
csvwrite(d(i).name, filtered_data(i));
end

a = cell2mat(data(1,1));
b = a(:,1);

ecg_filtered = sgolayfilt(b,4,11);

Wn= [5,14]/(290/2);
[b,a]=butter(4,Wn,'bandpass');

filteredECG=filtfilt(b,a,ecg_filtered);
% filteredECG=xECG; %REMOVE
%  plot(filteredECG);

N=8;
[C,L]=wavedec(filteredECG,N,'db4');
[thr,sorh,keepapp]=ddencmp('den','wv',filteredECG); 
filteredECG=wdencmp('gbl',C,L,'db4',8,thr,sorh,keepapp);

plot(filtered_data{1,20});
figure
plot(filtered_data{1,30});