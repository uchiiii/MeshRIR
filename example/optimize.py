import numpy as np
import scipy.special as special
import scipy.spatial.distance as distfuncs


class MKL:
	"""
	MKL(Multiple Kernel Learning)
	"""

	def __init__(self, subkernels, x, y, lamb=1e-1):
		self.MAXITR = 10000
		self.lamb = lamb

		self.n = x.shape[0]
		self.x = x
		self.y = y

		self.subkernels = subkernels
		self.d = len(self.subkernels)
		self.K = np.zeros((self.d, self.n, self.n), dtype=np.complex128)
		for i in range(self.d):
			self.K[i, :, :] = self.subkernels[i](x, x)

		if self.n != y.shape[0]:
			raise Exception('x.shape[0] should be the same as y.shape[0]')

	def __get_K(self, q):
		Kcur = np.zeros((self.n, self.n), dtype=np.complex128)
		for i in range(self.d):
			Kcur += q[i] * self.K[i,:,:]

		return Kcur

	def __get_Q(self, K, alpha):
		return ((K @ alpha - self.y).conj() @ (K @ alpha - self.y) + self.lamb * alpha.conj() @ K @ alpha).real

	def __get_armijo_step_size(self, q, Dr, partial_J, step0, J_cur, c1=0.5, t=0.5):
		step = step0
		m = Dr.conj() @ partial_J

		while True:
			K_next = self.__get_K(q + step * Dr)
			alpha = np.linalg.inv(K_next + self.lamb * np.eye(self.n, dtype=np.complex128)) @ self.y
			J_new = self.__get_Q(K_next, alpha)

			if J_new <= J_cur + c1 * step * m:
				return step
			else:
				step = step * t
			
		return step / 2.0

	def optimize(self, algo="l2"):
		if algo == "l1":
			return self.mkl_l1()
		elif algo == "l2":
			return self.mkl_l2()
		else:
			raise Exception('got invalid algorithm {}'.format(algo))

	def mkl_l1(self):
		epsilon = 1e-5

		q = np.zeros(self.d)
		q.fill(1.0/self.d)

		GAMMA_MAX = 1e5

		for iter in range(50):
			K_cur = self.__get_K(q)

			alpha = np.linalg.inv(K_cur + self.lamb * np.eye(self.n, dtype=np.complex128)) @ self.y
			J = self.__get_Q(K_cur, alpha)

			parial_J = np.zeros(self.d, dtype=np.complex128)
			for i in range(self.d):
				parial_J[i] = -alpha.conj() @ self.K[i] @ alpha
			parial_J = parial_J.real

			dJmin = np.min(parial_J[q > 0.0])
			dJmax = np.max(parial_J[q > 0.0])
			if np.abs(dJmin-dJmax) < epsilon and np.sum(parial_J[q == 0.0] < dJmax) == 0:
				break

			mu = np.argmax(q)

			Dr = np.zeros(self.d, dtype=np.float128)
			for i in range(self.d):
				if i != mu:
					if np.abs(q[i]) < epsilon and parial_J[i] - parial_J[mu] > 0.0:
						Dr[i] = 0.0
						q[i] = 0.0
					elif q[i] > 0.0:
						Dr[i] = - parial_J[i] + parial_J[mu]

			for i in range(self.d):
				if i != mu and q[i] > 0.0:
					Dr[mu] -= Dr[i]

			J_bar = 0.0
			q_bar = np.copy(q)
			Dr_bar = np.copy(Dr)
			alpha_bar = np.copy(alpha)

			gamma_cur = 0.0
			while J_bar + epsilon < J:
				q = np.copy(q_bar)
				Dr = np.copy(Dr_bar)

				K_cur = self.__get_K(q)
				alpha = np.linalg.inv(K_cur + self.lamb * np.eye(self.n, dtype=np.complex128)) @ self.y
				J = self.__get_Q(K_cur, alpha)

				mu = np.argmax(q)
				nu = 1

				gamma_max = GAMMA_MAX

				for i in range(self.d):
					if Dr[i] < 0.0 and -q[i] / Dr[i] < gamma_max:
						gamma_max = -q[i] / Dr[i]
						nu = i

				q_bar = q + gamma_max * Dr
				mu = 0
				if nu != 1:
					mu = 1
				else:
					mu = 2

				for i in range(self.d):
					if i == nu:
						continue
					if q_bar[i] > q_bar[mu]:
						mu = i

				Dr_bar[mu] = Dr[mu] + Dr[nu]
				Dr_bar[nu] = 0.0
				q_bar[nu] = 0.0

				Knext = self.__get_K(q)
				alpha_bar = np.linalg.inv(Knext + self.lamb * np.eye(self.n, dtype=np.complex128)) @ self.y
				J_bar = self.__get_Q(Knext, alpha_bar)

				gamma_cur = gamma_max

			if gamma_cur > 0:
				step = self.__get_armijo_step_size(q, Dr, parial_J, gamma_cur, J)
			else:
				step = 0.0

			q += step * Dr

		K_cur = self.__get_K(q)
		self.K_est = K_cur
		self.invK = np.linalg.inv(K_cur + self.lamb * np.eye(self.n, dtype=np.complex128))
		self.alpha = self.invK @ self.y
		self.q = q
		return q

	def mkl_l2(self):
		epsilon = 1e-5
		ratio = 0.5

		q0 = np.zeros(self.d)
		q1 = np.zeros(self.d)
		q1.fill(1.0/self.d)

		K0 = self.__get_K(q1)
		alpha1 = np.linalg.inv(K0+self.lamb*np.eye(self.n, dtype=np.complex128)) @ self.y
		alpha2 = np.zeros(self.n, dtype=np.complex128)

		n_itr = 0

		while np.linalg.norm(alpha1 - alpha2) > epsilon and n_itr < self.MAXITR:
			alpha2 = alpha1
			n_itr += 1

			v = np.zeros(self.d)
			for i in range(self.d):
				v[i] = np.real(alpha2 @ self.K[i,:,:] @ alpha2)

			q1 = q0 + 1.0 * v / np.linalg.norm(v)

			K_cur = np.zeros((self.n, self.n), dtype=np.complex128)
			for i in range(self.d):
				K_cur += q1[i] * self.K[i,:,:]

			alpha1 = ratio * alpha2 + (1-ratio) * (np.linalg.inv(K_cur + self.lamb * np.eye(self.n, dtype=np.complex128)) @ self.y)

		self.alpha = alpha1
		K_est = self.__get_K(q1)

		self.K_est = K_est
		self.invK = np.linalg.inv(K_est + self.lamb * np.eye(self.n, dtype=np.complex128))
		self.q = q1
		return q1
	
	def __get_main_kernel(self, x_est):
		n_est = x_est.shape[0]
		ker = np.zeros((n_est, self.n), dtype=np.complex128)
		for i in range(self.d):
			ker += self.q[i] * self.subkernels[i](x_est, self.x)
		return ker

	def get_estimated_filter(self, x_est):
		kappa = self.__get_main_kernel(x_est)
		return kappa @ self.invK # whose shape is (n_est, n)

	def predict(self, x_est):
		kappa = self.__get_main_kernel(x_est)
		return kappa @ self.alpha

def normal_kernel(k, beta, eta, x, y):
	distMat = distfuncs.cdist(x, y)
	return special.spherical_jn(0, k * distMat)

def kernel(k, beta, eta):
	return lambda x, y: normal_kernel(k, beta, eta, x, y)

if __name__=="__main__":
	# Sound speed (m/s)
	c = 347.3

	freq = 500.0
	k = 2 * np.pi * freq / c

	subkernels = []
	d_beta = 10
	d_eta = 10
	betas = np.arange(0.0, 9.0, 1.0)
	etas = np.arange(-np.pi, np.pi, 2 * np.pi / d_eta)

	numMic = 4

	for beta in betas:
		for eta in etas:
			subkernels.append(kernel(k, beta, eta))

	posMic = np.array([[0, 0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

	m = MKL(subkernels, posMic, np.ones(numMic))
	q = m.optimize(algo="l1")
	m.predict(np.ones((2,3)))

	distMat = distfuncs.cdist(posMic, posMic)
	K = special.spherical_jn(0, k * distMat)
	Kinv = np.linalg.inv(K + m.lamb * np.eye(numMic))

	np.testing.assert_array_almost_equal(m.K_est.real, K)

	np.testing.assert_array_almost_equal(m.invK, Kinv, decimal=4)
