cd 'D:\Desktop\dataframes'
d=dir('*.csv');   % dir struct of all pertinent .csv files
n=length(d);        % how many there were
data=cell(1,n);
gsrfiltered = cell(1,n);% preallocate a cell array to hold results
Fs=290;
for i=1:n
    data(i)={csvread(d(i).name, 1, 0)};  % read each file
    %csvwrite(d(i).name, filtered_data(i));
    
    [b,a]=ellip(4,0.1,40,4*2/Fs); % Elliptic filter design
    [H,w]=freqz(b,a,gsrlen); %frequency response
    temp = data{1,3};
    gsr = temp(:,2);
    gsrfiltered=filter(b,a,gsr);?
    csvwrite(d(i).name, gsrfiltered(i));
end

for j=1:n
temp = data{1,j};
[b,a]=ellip(4,0.1,40,4*2/Fs); % Elliptic filter design
[H,w]=freqz(b,a,size(temp,1)); %frequency response

gsr = temp(:,2);
gsr_filtered = filter(b,a,gsr);
gsr_filtered_sliced = gsr_filtered(250:end,1);

csvwrite(d(j).name, gsr_filtered_sliced);
end
