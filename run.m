% run experiments

clear all;
close all;
clc;
sch = Scheduler();
sch.train(0)


%%
addpath('/home/yz/caffe3/matlab');
sch = Scheduler();
sch.validate()