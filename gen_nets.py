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

# net 0
def net0(n, final_output):
    n.add_input(batch_size = 256, data_dim = [1, 2101, 1],
        label_dims = [[final_output]])
    n.add_fc(final_output)
    n.add_relu()
    # n.add_softmax_op()
    n.add_euclidean(name='', label='label0')
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 6e3, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-6, gamma = 0.1, stepsize = 2e3,
                display = 10, snapshot = 100e3)

# # net 1
# def net1(n, final_output):
#     n.add_input(batch_size = 256, data_dim = [1, 2101, 1], label_dims = [final_output])
#     n.add_conv_1d(16, kernel_size=11, stride=2, pad=5, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(24, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(32, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(40, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(48, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_fc(final_output)
#     n.add_relu()
#     # n.add_softmax_op()
#     n.add_euclidean(name='')
#     n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
#                 max_iter = 1500, base_lr = 0.01, momentum = 0.9,
#                 weight_decay = 1e-6, gamma = 0.1, stepsize = 500,
#                 display = 10, snapshot = 100e3)

# # net 1
# def net2(n, final_output):
#     n.add_input(batch_size = 256, data_dim = [1, 2101, 1], label_dims = [final_output])
#     n.add_conv_1d(32, kernel_size=7, stride=2, pad=3, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(32, bias_term=True)
#     n.add_conv_1d(32, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(64, bias_term=True)
#     n.add_conv_1d(64, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(128, bias_term=True)
#     n.add_conv_1d(128, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(256, bias_term=True)
#     n.add_conv_1d(256, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_fc(final_output)
#     n.add_relu()
#     # n.add_softmax_op()
#     n.add_euclidean(name='')
#     n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
#                 max_iter = 1500, base_lr = 0.01, momentum = 0.9,
#                 weight_decay = 1e-6, gamma = 0.1, stepsize = 500,
#                 display = 10, snapshot = 100e3)

# # net 1
# def net3(n, final_output):
#     n.add_input(batch_size = 256, data_dim = [1, 2101, 1], label_dims = [final_output])
#     n.add_conv_1d(32, kernel_size=7, stride=2, pad=3, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(32, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(64, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(128, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(256, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(512, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(1024, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_conv_1d(2048, stride=2, bias_term=True)
#     n.add_relu()
#     n.add_fc(final_output)
#     n.add_relu()
#     # n.add_softmax_op()
#     n.add_euclidean(name='')
#     n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
#                 max_iter = 1500, base_lr = 0.01, momentum = 0.9,
#                 weight_decay = 1e-6, gamma = 0.1, stepsize = 500,
#                 display = 10, snapshot = 100e3)

def net4(n, final_output):
    n.add_input(batch_size = 256, data_dim = [1, 2101, 1],
        label_dims = [[final_output]])
    n.add_conv_1d(32, kernel_size=11, stride=2, pad=5, bias_term=True)
    n.add_relu()
    n.add_conv_1d(64, stride=2, bias_term=True)
    n.add_relu()
    n.add_conv_1d(128, stride=2, bias_term=True)
    n.add_relu()
    n.add_conv_1d(256, stride=2, bias_term=True)
    n.add_relu()
    n.add_conv_1d(512, stride=2, bias_term=True)
    n.add_relu()
    n.add_fc(8192)
    n.add_relu()
    n.add_fc(1024)
    n.add_relu()
    n.add_fc(final_output)
    n.add_relu()
    # n.add_softmax_op()
    n.add_euclidean(name='', label='label0')
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 6e3, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-6, gamma = 0.1, stepsize = 2e3,
                display = 10, snapshot = 100e3)

def net5(n, final_output):
    n.add_input(batch_size = 256, data_dim = [1, 2101, 1],
        label_dims = [[final_output]])
    n.add_conv_1d(32, kernel_size=11, stride=2, pad=5, bias_term=True)
    n.add_relu()
    n.add_conv_1d(64, stride=2, bias_term=True)
    n.add_relu()
    n.add_conv_1d(128, stride=2, bias_term=True)
    n.add_relu()
    n.add_conv_1d(256, stride=2, bias_term=True)
    n.add_relu()
    n.add_conv_1d(512, stride=2, bias_term=True)
    n.add_relu()
    n.add_fc(4096)
    n.add_relu()
    n.add_fc(4096)
    n.add_relu()
    n.add_fc(final_output)
    # n.add_relu()
    # n.add_softmax_op()
    n.add_euclidean(name='', label='label0')
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 6e3, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-6, gamma = 0.1, stepsize = 2e3,
                display = 10, snapshot = 100e3)

def net6(n, final_output, spec_dim = 425, ri_dim = 11):
    n.add_input(batch_size = 256, data_dim = [1, spec_dim, 1],
        label_dims = [[final_output], [1]])
    n.add_conv_1d(64, kernel_size=7, stride=2, pad=3, bias_term=True)
    n.add_relu()
    for num in [128, 256, 512, 1024]:
      n.add_conv_1d(num, stride=2, bias_term=True)
      n.add_relu()
    branch1 = n.bottom
    n.add_fc(2048)
    n.add_relu()
    n.add_fc(2048)
    n.add_relu()
    n.add_fc(final_output)
    n.add_euclidean(name='spec_', label='label0', loss_weight=1)
    n.bottom = branch1
    n.add_fc(512)
    n.add_relu()
    n.add_fc(512)
    n.add_relu()
    n.add_fc(ri_dim)
    n.add_softmax(name='ri_', label='label1', loss_weight=0.1)
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 80e3, base_lr = 0.1, momentum = 0.9,
                weight_decay = 1e-6, gamma = 0.1, stepsize = 40e3,
                display = 10, snapshot = 100e3)

def net7(n, final_output, spec_dim = 425, ri_dim = 11):
    n.add_input(batch_size = 256, data_dim = [1, spec_dim, 1],
        label_dims = [[final_output], [1]])
    n.add_conv_1d(64, kernel_size=7, stride=2, pad=3, bias_term=True)
    n.add_relu()
    for num in [128, 256, 512, 1024]:
      n.add_conv_1d(num, stride=2, bias_term=True)
      n.add_relu()
    branch1 = n.bottom
    n.add_fc(2048)
    n.add_relu()
    n.add_fc(2048)
    n.add_relu()
    n.add_fc(512)
    n.add_relu()
    n.add_fc(512)
    n.add_relu()
    n.add_fc(final_output)
    n.add_euclidean(name='spec_', label='label0', loss_weight=1)
    n.bottom = branch1
    n.add_fc(1024)
    n.add_relu()
    n.add_fc(1024)
    n.add_relu()
    n.add_fc(256)
    n.add_relu()
    n.add_fc(256)
    n.add_relu()
    n.add_fc(ri_dim)
    n.add_softmax(name='ri_', label='label1', loss_weight=0.1)
    n.add_solver_sdg(test_interval = 1e5, test_iter = 1, iter_size = 1,
                max_iter = 150e3, base_lr = 0.1, momentum = 0.9,
                weight_decay = 1e-6, gamma = 0.1, stepsize = 50e3,
                display = 10, snapshot = 100e3)


for final_output in [121, 401, 1001]:
    net = b.BuildNet(lambda n: net0(n, final_output=final_output),
        name = 'group1/fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()
    # net = b.BuildNet(lambda n: net1(n, final_output=final_output),
    #     name = 'conv1_fc-'+str(final_output),
    #     caffe_path = caffe_path)
    # net.save()
    # net = b.BuildNet(lambda n: net2(n, final_output=final_output),
    #     name = 'conv2_fc-'+str(final_output),
    #     caffe_path = caffe_path)
    # net.save()
    # net = b.BuildNet(lambda n: net3(n, final_output=final_output),
    #     name = 'conv3_fc-'+str(final_output),
    #     caffe_path = caffe_path)
    # net.save()
    net = b.BuildNet(lambda n: net4(n, final_output=final_output),
        name = 'group1/conv4_fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()
    net = b.BuildNet(lambda n: net5(n, final_output=final_output),
        name = 'group1/conv5_fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()

    ## second group
for final_output in [34]:
    net = b.BuildNet(lambda n: net6(n, final_output=final_output),
        name = 'group2/conv6_fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()
    net = b.BuildNet(lambda n: net7(n, final_output=final_output),
        name = 'group2/conv7_fc-'+str(final_output),
        caffe_path = caffe_path)
    net.save()