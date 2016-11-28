% run experiments

clear all;
close all;
clc;
sch = Scheduler();
sch.train(0)
sch.validate()