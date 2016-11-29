# spectrometer-cnn
using CNN for spectrum reconstruction using all fiber spectromter

# analysis stream line

- DataLoader, contains different methods for loading data

- CaffeModel, contains wrapper for training using different caffe models

# results
## old
- in single line reconstruction, cnn is able to reconstruct sub correlation images with noiseless image
- in multiple line (5 lines) reconstruction, cnn perform similar to linear reconstruction

## nerual network training
- for the last layer, using softmax cant converge (may be try L1 normalize)
- for the last layer, without relu reaches much better loss

## new
- linear reconstruction does not work well, not nearly as good as optimization


# facts
- noise is about 1e-3 of the whole mean value.

# control experiments include
- noise options: yes or no (prepared)
- method options: linear, neural net, and optimization (need optimization)
- spectrum options: single, multiple, and continuous (need continous)
- spectrum density: correlated and uncorrelated (prepared, using different transimission matrix)

# goals of next steps
- limit the neural net parameters
- finish control experiments
- temperature: fixed and sampled
