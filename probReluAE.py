import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

class ConvLayer(object):
	"""
		:param input
			shape = (batchsize, channels, row, col)
		:param filter_shape
			shape = (filters, channels, row, col)
	"""
	def __init__(self, rng, input, image_shape, prob, filter_shape, conv_stride):
		assert image_shape[1] == filter_shape[1]
		self.image_shape = image_shape
		self.input = input
		self.prob = prob
		self.filter_shape = filter_shape
		self.conv_stride = conv_stride

		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   4)
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		
		self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
		self.Q = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(image_shape[1], filter_shape[0], filter_shape[2], filter_shape[3])),
            dtype=theano.config.floatX),
                               borrow=True)
		self.params = [self.W, self.Q]

	def feedforward(self):
		conv_out = conv.conv2d(input=self.input, filters=self.W,
                filter_shape=self.filter_shape, subsample=self.conv_stride, border_mode='valid')
		
		mask = np.asarray(
			rng.uniform(
				low=0, 
				high=1, 
				size=(self.image_shape[0], self.image_shape[1], 
					self.image_shape[2]-self.filter_shape[2]+1, self.image_shape[3]-self.filter_shape[3]+1)
				),
			    dtype=theano.config.floatX)
		mask = (mask > 1 - self.prob)
		self.output = conv_out * mask
	

if __name__ == '__main__':
	rng = np.random.RandomState(23455)
	img = theano.shared(np.asarray(rng.uniform(low=0, high=1, size=[100, 1024]), dtype=theano.config.floatX), borrow=True)
	index = T.lscalar()
	x = T.matrix('x')
	batchsize = 5
	L1 = ConvLayer(
		rng,
		input=x.reshape((batchsize, 1, 32, 32)),
		image_shape=[batchsize, 1, 32, 32],
		prob=0.4,
		filter_shape=[32, 1, 5, 5],
		conv_stride=[1, 1]
	)

	lr = 0.00001

	L1.feedforward()
	recon = conv.conv2d(
		   input=L1.output,
		   filters=L1.Q, 
		   subsample=L1.conv_stride, 
		   border_mode='full'
	)
	#assert np.shape(recon)==np.shape(self.input)
	cost = 0.5*T.mean(T.sum(T.square(recon - L1.input)))
	grads = T.grad(cost, L1.params)
	updates = []
	for param_i, grad_i in zip(L1.params, grads):
		updates.append((param_i, param_i - lr * grad_i))

	model = theano.function([index], cost, updates=updates,
		    givens={
		    	x: img[batchsize * index : batchsize * (index + 1)]})
	cost = []
	for it in xrange(20):
		for i in xrange(19):
			cost.append(model(i))





		