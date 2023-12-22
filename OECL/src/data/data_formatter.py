class MultiTransforms(object):
	def __init__(self, transform):
		self.trn1 = transform
		self.trn2 = transform

	def __call__(self, sample):
		x1 = self.trn1(sample)
		x2 = self.trn2(sample)

		return x1, x2
