%% run experiments group1
addpath('/home/yz/caffe3/matlab');
clear all;
close all;
clc;
sch = Scheduler();
sch.force_train=false;
sch.train(0)


%%
addpath('/home/yz/caffe3/matlab');
sch = Scheduler();
sch.num_displays = 5;
sch.num_vals = 5;
sch.validate()

%% run experiments group2
addpath('/home/yz/caffe3/matlab');
clear all;
close all;
clc;
sch = SchedulerRI();
sch.force_train=false;
sch.train(0)


%%
addpath('/home/yz/caffe3/matlab');
sch = SchedulerRI();
sch.num_displays = 5;
sch.num_vals = 5;
sch.validate()