# spectrometer-cnn
using CNN for spectrum reconstruction using all fiber spectromter

# folder contents

- Model, saving caffe models

- Data, transmission matrices

- results, trained model, plots, etc

- archive, old results

- obsolete, old codes

# analysis stream line

- DataLoader, contains different methods for loading data

- CaffeModel, contains wrapper for training using different caffe models

- Scheduler, orgnize batch model traning and model validation

- OptModel, using optimization, \sum (T*S - I)^2 + \lambda * \sum S^2

- RI sufix, refractive index data.

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
- noise options: yes or no (done)
- method options: linear, neural net, and optimization (done)
- spectrum options: single, multiple, and continuous (need continous)
- spectrum density: correlated and uncorrelated (done, using different transimission matrix)

# next steps

