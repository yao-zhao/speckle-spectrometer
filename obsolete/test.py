import sys
sys.path.append('/home/yz/caffe3/python')
import caffe

model_def = 'model/cnn/train.prototxt'

net = caffe.Net(model_def, caffe.TEST);