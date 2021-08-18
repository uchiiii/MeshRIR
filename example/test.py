import unittest
import numpy as np
import scipy.special as special
import scipy.signal as signal

def sinc(x):
	if x == 0.0:
		return 1.0
	else:
		return np.sin(x) / x

class FunctionTest(unittest.TestCase):
	def setUp(self):
		self.epsilon = 1e-4
		pass

	def tearDown(self):
		pass

	def test_spherical_real(self):
		for x in np.arange(0.0, 10.0, 0.5):
			self.assertAlmostEqual(sinc(x), special.spherical_jn(0, x))

	def test_spherical_imag(self):
		for _ in range(10):
			x = np.random.random() + np.random.random() * 1j
			self.assertAlmostEqual(sinc(x), special.spherical_jn(0, x))

		
	def test_example(self):
		self.assertEqual(True, True)

	def test_fft(self):
		# x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
		# y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
		x = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0])
		y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

		len_x = x.shape[0]
		len_y = y.shape[0]
		length = pow(2, int(np.ceil(np.lib.scimath.log2(len_x+len_y-1))))
		np.testing.assert_array_almost_equal(
				x, np.fft.ifft(np.fft.fft(x)))
		np.testing.assert_array_almost_equal(
				np.convolve(x, y, mode='valid'), 
				signal.fftconvolve(x, y, mode='valid'))
		print(signal.fftconvolve(x, y, mode='valid'))
		print(signal.fftconvolve(x, y, mode='full').shape)
		# np.testing.assert_array_almost_equal(
		# 		np.fft.irfft(np.fft.rfft(x, n=len_x+len_y-1) * np.fft.rfft(y, n=len_x+len_y-1)),
		# 		signal.fftconvolve(x, y, mode='full'))
		# np.testing.assert_array_almost_equal(
		# 		np.fft.irfft(np.fft.rfft(x, n=len_x) * np.fft.rfft(y, n=len_x)),
		# 		signal.fftconvolve(x, y, mode='valid'))
		np.testing.assert_array_almost_equal(
				np.convolve(x, y, mode='full'), 
				np.fft.ifft(np.fft.fft(x, n=length) * np.fft.fft(y, n=length))[:len_x+len_y-1].real)		
		np.testing.assert_array_almost_equal(
				np.convolve(x, y, mode='full'), 
				np.fft.irfft(np.fft.rfft(x, n=length) * np.fft.rfft(y, n=length))[:len_x+len_y-1].real)		
		np.testing.assert_array_almost_equal(
				np.convolve(x, y, mode='full'), 
				np.fft.ifft(np.fft.fft(x, n=2*length) * np.fft.fft(y, n=2*length))[:len_x+len_y-1].real)
		np.testing.assert_array_almost_equal(
				np.convolve(x, y, mode='full'), 
				np.fft.ifft(np.fft.fft(x, n=length) * np.fft.fft(y, n=length))[:len_x+len_y-1].real)		
		diff = ((len_x + len_y - 1) - len_x) // 2
		np.testing.assert_array_almost_equal(
				np.convolve(x, y, mode='same'), 
				np.fft.ifft(np.fft.fft(x, n=length) * np.fft.fft(y, n=length))[diff:diff+len_x].real)		

	def test_rfft(self):
		x = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0])
		kiTF = np.fft.rfft(x)
		print(kiTF.shape)
		print(np.fft.rfftfreq(x.shape[0], d=1/100))
		print(np.fft.fftfreq(x.shape[0], d=1/100))
		# _kiTF = np.concatenate((kiTF[:-1], kiTF[-1:0:-1].conj()))
		_kiTF = np.concatenate((kiTF[:], kiTF[-2:0:-1].conj()))
		_sig1 = np.fft.ifft(_kiTF, axis=0).real
		_sig2 = np.fft.irfft(kiTF, axis=0)
		np.testing.assert_array_almost_equal(_sig1, _sig2)
	
	def test_fftfreq(self):
		f_100 = 100.0
		fftlen = 20
		np.testing.assert_array_almost_equal(
				np.arange(0,fftlen/2+1)/fftlen*f_100,
				np.fft.rfftfreq(fftlen, d=1/f_100))

	def test_stft(self):
		x = np.random.rand(1024)

		freq = 8000.0
		_fs, _ts, spec_x = signal.stft(x, fs=freq, window='hamming', nperseg=256)
		np.testing.assert_array_almost_equal(
			x,
			signal.istft(spec_x, fs=freq, window='hamming')[1]
		)


if __name__ == "__main__":
	unittest.main()