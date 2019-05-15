import pylab
from matplotlib.pyplot import imread
import numpy as np
from scipy import fftpack
from skimage import io

im = io.imread('../Dataset/mit_noise_periodic.jpg', as_gray=True)
pylab.figure(figsize=(15,10))
# im = np.mean(aa, axis=2) / 255
print(im.shape)
pylab.subplot(1,2,1), pylab.imshow(im, cmap='gray'), pylab.axis('off')
pylab.title('Original Image')
F1 = fftpack.fft2((im).astype(float))
F2 = fftpack.fftshift( F1 )
pylab.subplot(1,2,2), pylab.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=pylab.cm.gray)
pylab.xticks(np.arange(0, im.shape[1], 25))
pylab.yticks(np.arange(0, im.shape[0], 25))
pylab.title('Noisy Image Spectrum')
pylab.tight_layout()
pylab.show()

# F2[170:176,:220] = F2[170:176,230:] = 0 # eliminate the frequenci
# im1 = fftpack.ifft2(fftpack.ifftshift( F2 )).real
# pylab.axis('off'), pylab.imshow(im1, cmap='gray'), pylab.show()