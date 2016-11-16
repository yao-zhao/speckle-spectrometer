# spectrometer-cnn
using CNN for spectrum reconstruction using all fiber spectromter

# results
- in single line reconstruction, cnn is able to reconstruct sub correlation images with noiseless image
- in multiple line (5 lines) reconstruction, cnn perform similar to linear reconstruction

# facts
- noise is about 1e-3 of the whole mean value.

# control experiments include
- noise options: yes or no
- method options: linear, neural net, and optimization
- spectrum options: single, multiple, and continuous
- spectrum density: correlated and uncorrelated

# goals of next steps
- limit the neural net parameters
- finish control experiments
- temperature: fixed and sampled
