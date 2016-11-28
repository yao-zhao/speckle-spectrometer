import numpy as np
import os
import argparse
import sys
caffe_path = '/home/yz/caffe3/'
sys.path.append(caffe_path+'python')
import caffe
import build_net as b
import os
os.environ["GLOG_minloglevel"] = "2"

# net 1
def net0(n, final_output):
    n.add_input(batch_size = 256, data_dim = [1, 2101, 1], label_dim = [final_output])
    n.add_fc(final_output)
    n.add_euclidean()
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 6e4, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-4, gamma = 0.1, stepsize = 2e4,
                display = 10, snapshot = 5e3)

# net 1
def net1(n):
    n.add_input(batch_size = 256, data_dim = [1, 2101, 1], label_dim = [final_output])
    n.add_conv_1d(16, kernel_size=7, stride=2, pad=3, bias_term=True)
    n.add_relu()
    n.add_conv_1d(24, bias_term=True)
    n.add_relu()
    n.add_conv_1d(32, bias_term=True)
    n.add_relu()
    n.add_fc(final_output)
    n.add_euclidean()
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 6e4, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-4, gamma = 0.1, stepsize = 2e4,
                display = 10, snapshot = 5e3)

for final_output in [121, 401, 1001]:
    net = b.BuildNet(lambda n: net0(n, final_output=final_output),
        name = 'fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()
    net = b.BuildNet(lambda n: net0(n, final_output=final_output),
        name = 'conv_7x16_3x24_3x32_fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()