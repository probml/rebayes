# data from https://www.mathworks.com/help/deeplearning/ug/train-a-convolutional-neural-network-for-regression.html
# save('rotated_digits.mat', 'XTrain', 'YTrain', 'XValidation', 'YValidation')

import scipy
dat = scipy.io.loadmat('/Users/kpmurphy/github/rebayes/misc/rotated_digits_matlab.mat')
XTrain, YTrain, XValidation, YValidation = dat['XTrain'], dat['YTrain'], dat['XValidation'], dat['YValidation']
print(XTrain.shape, YTrain.shape, XValidation.shape, YValidation.shape)
print(YTrain[:10])