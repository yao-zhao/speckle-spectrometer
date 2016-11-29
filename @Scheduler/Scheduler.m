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
        % path parameter
        datapath = 'data'
        modelpath = 'models'
        resultpath = 'results'
        % validation parameter
        num_vals = 2;
        num_displays = 2;
        % training parameter
        force_train = false;
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
                    if exist(fullfile(foldername, 'trained.caffemodel')) && ~obj.force_train
                        display('result exist, ignore training')
                        fprintf('\n');
                    else
                        mkdir(foldername);
                        obj.savenames = [obj.savenames, {foldername}];
                        cm = CaffeModel(cm_names{im}, dl);
                        if is_test
                            cm.test_initialization();
                        else
                            cm.train()
                            copyfile(fullfile(obj.modelpath, cm_names{im},...
                                ['stage_0_iter_', num2str(cm.solver.max_iter()), '.caffemodel']),...
                                fullfile(foldername, 'trained.caffemodel'))
                            %                         copyfile(fullfile(obj.modelpath, cm_names{im},...
                            %                             ['stage_0_iter_', num2str(cm.solver.max_iter()), '.solverstate']),...
                            %                             fullfile(foldername, 'trained.solverstate'))
                        end
                        cm.save(foldername)
                        display('initialization passed')
                        fprintf('\n');
                    end
                    expcounter = expcounter + 1;
                end
            end
        end
        
        % validation
        function validate(obj)
            files = dir(obj.resultpath);
            files = files([files.isdir]);
            files = files(3:end);
            caffe.set_mode_gpu();
            caffe.set_device(0);
            for ifile = 1:length(files)
                filename = files(ifile).name;
                display(['validate trained model in ', filename]);
                % load data and model
                traindata = load(fullfile(obj.resultpath, filename, 'training.mat'));
                dl = traindata.dataloader;
                dl.batchsize = obj.num_vals;
                caffe.reset_all();
                net = caffe.Net(fullfile(obj.modelpath, ....
                    traindata.modelname, 'deploy_0.prototxt'), ...
                    fullfile(obj.resultpath, filename, 'trained.caffemodel'), ...
                    'test');
                [img_batch, spectra_batch] = dl.getBatch();
                % model inference
                model_spectra = zeros(size(spectra_batch));
                tstart = tic;
                for ibatch = 1:dl.batchsize
                    % net inference
                    net.blobs('data').set_data(img_batch(:, :, :, ibatch));
                    net.forward_prefilled();
                    model_spectra(:, ibatch) =...
                        double(net.blobs(net.outputs{1}).get_data());
                end
                model_time = toc(tstart);
                fprintf('model inference time of batch size %d is %2.4f\n', dl.batchsize, model_time);
                % opt inference
                om = OptModel(dl.T);
                [opt_spectra, opt_time] = om.inference(img_batch);
                fprintf('optimization time of batch size %d is %2.4f\n', dl.batchsize, opt_time);
                % plot
                close all;
                figure('Position', [100, 100, 1200, 800]);
                for ibatch = 1:obj.num_displays
                    clf;
                    specvec = (1:size(spectra_batch, 1))';
                    plot(specvec, spectra_batch(:, ibatch)); hold on;
                    plot(specvec, model_spectra(:, ibatch)); hold on;
                    plot(specvec, opt_spectra(:, ibatch)); hold on;
                    xlabel('wavelength');
                    ylabel('spectrum');
                    title(filename);
                    legend('label', 'model', 'optimization');
                    print(fullfile(obj.resultpath, filename,...
                        ['val_', num2str(ibatch), '.pdf']), '-dpdf', '-bestfit')
                    pause(.1)
                end
                % loss
                model_loss = mean(sum((model_spectra - spectra_batch).^2, 1));
                opt_loss = mean(sum((opt_spectra - spectra_batch).^2, 1));
                fprintf('average sum loss of model is %2.4f\n', model_loss);
                fprintf('average sum loss of optimization is %2.4f\n', opt_loss);
                model_inference_time_per_image = model_time / obj.num_vals;
                optimization_time_per_image = opt_time / obj.num_vals;
                save(fullfile(obj.resultpath, filename, 'validation.mat'),...
                    'model_loss', 'opt_loss', 'model_inference_time_per_image',...
                    'optimization_time_per_image');
                
            end
        end
        
    end
    
end

