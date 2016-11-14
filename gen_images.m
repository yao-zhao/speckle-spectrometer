% generate simulated spectra and intensity pair
datapath = 'data';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};
count = 1;

for ifile = 1%:length(filenames)
   filename = fullfile(datapath, filenames{ifile});
   [~, label, ~]= fileparts(filenames{ifile});
   load_data = load(filename);
   T = load_data.T;
   [numpix, numspec] = size(T);
   imagepath = fullfile(datapath, label);
   mkdir(imagepath);
   % only train on all single wavelength composition
   for i=1:numspec
   end
%    for i=1:count
%        spec = smooth(rand(1, numspec),5);
%        spec = spec/sum(spec);
%        plot(spec)
%    end
end