% run experiments
addpath('/home/yz/caffe3/matlab');
%%
clear all;
close all;
clc;
sch = Scheduler();
sch.force_train=false;
sch.train(0)


%%
sch = Scheduler();
sch.validate()