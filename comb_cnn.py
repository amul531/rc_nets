#this works!!!!
# 6-24-2016: Rewrite draft of CNN with numpy for incremental testing.
# -------------------------------------------------------------------
# Notes:
# - Array indexing: correct convention is (Depth, Height, Width)
# - Padding and stride not yet taken into account
# - Using Python 2.7
# 
# To do:
# - Apply element-wise activation function after convolving. ReLU/tanh
# - ReLu/tanh/sigmoid activation functions after fully-connected layers
# - Softmax vs. logistic regression
# - Backpropagation in FC architecture?
# -------------------------------------------------------------------
#kernels-filters/weights
import numpy as np
import nnCls as nn

def generateRandomKernels(inputDepth, numKernels, kernelSize):
	return [np.random.rand(inputDepth, kernelSize, kernelSize) for i in range(numKernels)]



def convLayer(inputVolume, kernels):
	print 'Putting input through convolutional layer.'

	kernelSize = np.shape(kernels[0])[2]

	inputHeight = np.shape(inputVolume)[1]
	inputWidth = np.shape(inputVolume)[2]

	outputHeight = inputHeight - kernelSize + 1
	outputWidth = inputWidth - kernelSize + 1
	
	if kernelSize > outputHeight or kernelSize > outputWidth:
		raise ValueError('Kernel size is larger than output volume height or width.')

	print 'kernel size:', kernelSize
	print 'output volume height:', outputHeight
	print 'output volume width:', outputWidth

	outputVolume = np.zeros((len(kernels), outputHeight, outputWidth))
	
	print 'expected output volume shape:', np.shape(outputVolume)

	for kernelIndex in range(len(kernels)):
		for i in range(outputHeight):
			for j in range(outputWidth):
				featureMap = np.sum(inputVolume[:, i:i+kernelSize, j:j+kernelSize] * kernels[kernelIndex])
				outputVolume[kernelIndex, i, j] = featureMap


	return np.tanh(outputVolume)




def poolLayer(inputVolume, poolFactor):
	print 'Putting input through pooling layer.'

	inputDepth = np.shape(inputVolume)[0]
	inputHeight = np.shape(inputVolume)[1]
	inputWidth = np.shape(inputVolume)[2]

	if not ((inputHeight % poolFactor == 0) and (inputWidth % poolFactor == 0)):
		raise ValueError('Pool factor does not neatly divide the input volume height and weight.')

	outputHeight = inputHeight / poolFactor
	outputWidth = inputWidth / poolFactor

	maxIndexMatrix = np.zeros((inputDepth, outputHeight, outputWidth))
	outputVolume = np.zeros((inputDepth, outputHeight, outputWidth))

	print 'expected output volume shape', np.shape(outputVolume)

	for layerIndex in range(inputDepth):
		for i in range(outputHeight):
			for j in range(outputWidth):
				poolWindow = inputVolume[
								layerIndex,
								i*poolFactor : i*poolFactor+poolFactor,
								j*poolFactor : j*poolFactor+poolFactor]
				maxVal = np.amax(poolWindow)
				outputVolume[layerIndex, i, j] = maxVal

	return outputVolume


def generateRandomWeights(numInputNeurons, numOutputNeurons):
	return np.random.rand(numInputNeurons, numOutputNeurons)


def fcLayer(X, W, activation):
	#Note: make sure to concatenate another neuron, for bias

	print 'Putting input through FC layer.'
	
	Z = np.dot(X, W)

	if activation=='tanh':
		return np.tanh(Z)
	elif activation=='zeromax':
		return np.maximum(0, Z)
	else:
		raise ValueError('Invalid activation function option.')


if __name__ == "__main__":
	#Layer 0: Random input volume of dimension 3x64x64
	inputVolume = np.random.rand(3, 28, 28)


	#Layer 1: Convolutional layer with 10 filters of dimension 3x3
	kernels1 = generateRandomKernels(inputDepth=np.shape(inputVolume)[0], numKernels=3, kernelSize=3)
	outputVolume_conv1 = convLayer(inputVolume=inputVolume, kernels=kernels1)
	print 'resulting output volume shape:', np.shape(outputVolume_conv1)

	#Layer 2: Pooling layer with pool factor of 2
	outputVolume_pool1 = poolLayer(inputVolume=outputVolume_conv1, poolFactor=2)
	print 'resulting output volume shape:', np.shape(outputVolume_pool1)
	poolFlat = outputVolume_pool1.reshape((1,outputVolume_pool1.shape[0]*outputVolume_pool1.shape[1]*outputVolume_pool1.shape[2]))
	print 'poolFlattened shape:', poolFlat.shape
    
	print "feedforward/nn class fns-train"
	cnn_test = nn.nn(poolFlat, np.array([[1]]),5)

	print "feedforward/nn class fns-train"
	cnn_test.train_nn(poolFlat,np.array([[1]]),0.25,501)
	cnn_test.conf_matrix(poolFlat,np.array([[1]])) 
