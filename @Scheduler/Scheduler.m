classdef Scheduler < handle
    % SCHEDULER for experiments
    % scheduler read all possible transmission matrix
    % and all possible models matched to each transmission matrix
    % runs training of models and saves output
    properties
        filenames
        compatiblemodels
        modelnames
        numoutputs
        datapath = 'data'
        modelpath = 'models'
    end
    
    methods
        % constructor
        function obj = Scheduler()
            % load transmission matrix
            files = dir(fullfile(obj.datapath, '*.mat'));
            obj.filenames = {files.name};
            % load models
            models = dir(fullfile(obj.modelpath));
            models = models(3:end);
            obj.modelnames = {models.name};
            numModels = length(obj.modelnames);
            obj.numoutputs = zeros(1, numModels);
            for im = 1:numModels
                tmpsplit = strsplit(obj.modelnames{im}, '-');
                obj.numoutputs(im) = str2double(tmpsplit(end));
            end
            % find possible models
            numFiles = length(obj.filenames);
            obj.compatiblemodels = cell(1, numFiles);
            for ifile = 1:numFiles
                filename = fullfile(obj.datapath, obj.filenames{ifile});
                dl = DataLoader(filename);
                obj.compatiblemodels{ifile} = obj.modelnames(obj.numoutputs == dl.numSpec);
            end
        end
        
        % train all models
        function train(obj)
            numFiles = length(obj.filenames);
            expcounter = 1;
            display('Preview Experiments and Test loading ----------------------------------');
            for ifile = 1:numFiles
                filename = fullfile(obj.datapath, obj.filenames{ifile});
                dl = DataLoader(filename);
                cm_names = obj.compatiblemodels{ifile};
                for im = 1:length(cm_names)
                    display(['experiment No. ', num2str(expcounter)])
                    display(['transmission matrix name: ', filename])
                    display(['model name: ', cm_names{im}])
                    cm = CaffeModel(cm_names{im}, dl);
                    cm.test_initialization();
                    display('initialization passed')
                    expcounter = expcounter + 1;
                end
            end
        end
        
    end
    
end

