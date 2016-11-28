classdef CaffeModel < handle
    % Hold models for training
    
    properties
        modelpath = 'models'
        name
        dataloader
        % training parameters
        solver
        totaliter = 100
        % result
        losses
        training_time
    end
    
    methods
        
        % constructor
        function obj = CaffeModel(modelname, dataloader)
            obj.name = modelname;
            caffe.reset_all();
            caffe.set_mode_gpu();
            caffe.set_device(0);
            obj.solver = caffe.Solver(fullfile(obj.modelpath, obj.name, 'solver_0.prototxt'));
            obj.totaliter = obj.solver.max_iter();
            obj.dataloader = dataloader;
        end
        
        % function train
        function train(obj)
            loss = zeros(obj.totaliter, 1);
            tstart = tic;
            while obj.solver.iter() < obj.solver.max_iter()
                iter = obj.solver.iter()+1;
                [ img_batch, spectra_batch ] = obj.dataloader.getBatch();
                % load data
                obj.solver.net.blobs('data').set_data(img_batch);
                obj.solver.net.blobs('label').set_data(spectra_batch);
                % step 1
                obj.solver.step(1);
                % get result
                loss(iter) = double(obj.solver.net.blobs('loss').get_data());
                % display progress
                if mod(iter, 100) == 0
                    display(['iteration ', num2str(iter), ' loss ', num2str(loss(iter))]);
                end
            end
            % save
            obj.solver.snapshot();
            obj.training_time = toc(tstart);
            obj.losses = loss;
        end
        
        
        % function train
        function test_initialization(obj)
            [ img_batch, spectra_batch ] = obj.dataloader.getBatch();
            % load data
            obj.solver.net.blobs('data').set_data(img_batch);
            obj.solver.net.blobs('label').set_data(spectra_batch);
            % step 1
            obj.solver.step(1);
            % get result
            double(obj.solver.net.blobs('loss').get_data());
            % save
            obj.solver.snapshot();
        end
        
        % load from net
        function loadNet(obj, savename)
            obj.solver.net.copy_from(fullfile('results',savename,'trained.caffemodel'));
        end
        
        % function output
        function save(obj, savepath)
            losses = obj.losses;
            training_time = obj.training_time;
            dataloader = obj.dataloader;
            modelname = obj.name;
            save(fullfile(savepath, 'training.mat'),...
                'losses', 'training_time', 'dataloader', 'modelname');
        end
    end
    
end

