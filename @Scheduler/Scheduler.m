classdef Scheduler < handle
    % SCHEDULER for experiments
    % scheduler read all possible transmission matrix
    % and all possible models matched to each transmission matrix
    % runs training of models and saves output
    properties
        filenames
        compatiblemodels
        modelnames
        savenames
        numoutputs
        datapath = 'data'
        modelpath = 'models'
        resultpath = 'results'
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
        
        % test initialization traning of all models
        function train(obj, is_test)
            numFiles = length(obj.filenames);
            expcounter = 1;
            obj.savenames = [];
            display('Preview Experiments and Test loading ----------------------------------');
            for ifile = 1:numFiles
                filename = fullfile(obj.datapath, obj.filenames{ifile});
                dl = DataLoader(filename);
                cm_names = obj.compatiblemodels{ifile};
                dlsavename = dl.getSaveName();
                for im = 1:length(cm_names)
                    display(['experiment No. ', num2str(expcounter)])
                    display(['transmission matrix name: ', filename])
                    display(['model name: ', cm_names{im}])
                    display(['save name: ', dlsavename, cm_names{im}])
                    foldername = fullfile(obj.resultpath, [dlsavename, cm_names{im}]);
                    mkdir(foldername);
                    obj.savenames = [obj.savenames, {foldername}];
                    cm = CaffeModel(cm_names{im}, dl);
                    if is_test
                        cm.test_initialization();
                    else
                        cm.train()
                        copyfile(fullfile(obj.modelpath, cm_names{im},...
                            'stage_0_final_0.caffemodel'),...
                            fullfile(foldername, 'trained.caffemodel'))
                    end
                    cm.save(foldername)
                    display('initialization passed')
                    fprintf('\n');
                    expcounter = expcounter + 1;
                end
            end
        end
        
        % validation 
        
    end
    
end

