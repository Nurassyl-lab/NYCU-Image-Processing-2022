# In[1]
'import libraries'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, fftshift, ifftshift, ifft2
import matplotlib
import pandas as pd

# In[2]
'load images'

fruits = np.array(Image.open('fruit.tif'))
kid = np.array(Image.open('kid.tif'))

# In[3]    
'define some functions'

def centered_fft2D(im):
    fft_im = fft2(im)
    center_im = fftshift(fft_im)
    return center_im

def plot_im(im, save = False, name = '', cm = 'gray'):
    if save:
        matplotlib.use('Agg')
    else:
        matplotlib.use('qt5agg')
        
    fig = plt.figure(dpi = 150)
    fig.set_size_inches(im.shape[0]/150, im.shape[1]/150, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im, cmap = cm)
    
    if save:
        plt.savefig(name+'.png', dpi = 150)
        plt.close()
        matplotlib.use('qt5agg')
    return

def G_LPF(im, cutoff = 100):#default cutoff is 100, but I'm using 200
    H = np.zeros(im.shape, dtype = np.float32)
    for u in range(im.shape[0]):
        for v in range(im.shape[1]):
            D = np.sqrt((u-im.shape[0]/2)**2 + (v-im.shape[1]/2)**2)
            H[u][v] = np.exp(-D**2/(2*cutoff**2))
    return H

# In[4]
'Task (b) Fourier Magnitude Spectrum'

fruits_hat = centered_fft2D(fruits)
kid_hat = centered_fft2D(kid)

plot_im(np.log(1 + np.abs(fruits_hat)), save = True, name = 'fruits_fft_mag')
plot_im(np.log(1 + np.abs(kid_hat)), save = True, name = 'kid_fft_mag')

# In[5]
'Task (c) and (d) Gaussian LPF'

padded_fruits =  np.pad(fruits, ((0,600),(0,600)))
padded_kid =  np.pad(kid, ((0,600),(0,600)))
 
pad_fruits_fft = centered_fft2D(padded_fruits)
pad_kid_fft = centered_fft2D(padded_kid)

g_lpf = G_LPF(pad_fruits_fft, cutoff = 200)

plot_im(np.abs(g_lpf), save = True, name = 'mag_resp_lpf')

shifted_fruit_lpf_out = pad_fruits_fft * g_lpf
shifted_kid_lpf_out = pad_kid_fft * g_lpf

fruit_lpf_out_padded = ifft2(ifftshift(shifted_fruit_lpf_out))
kid_lpf_out_padded = ifft2(ifftshift(shifted_kid_lpf_out))

fruit_lpf_out = fruit_lpf_out_padded[0:600, 0:600]
kid_lpf_out = kid_lpf_out_padded[0:600, 0:600]

plot_im(np.abs(fruit_lpf_out), save = True, name = 'fruits_lpf_out')
plot_im(np.abs(kid_lpf_out), save = True, name = 'kid_lpf_out')

# In[6]
'Task (c) and (d) Gaussian HPF'

g_hpf = 1 -  g_lpf

plot_im(np.abs(g_hpf), save = True, name = 'mag_resp_hpf')

shifted_fruit_hpf_out = g_hpf * pad_fruits_fft
shifted_kid_hpf_out = g_hpf * pad_kid_fft

fruit_hpf_out_padded = ifft2(ifftshift(shifted_fruit_hpf_out))
kid_hpf_out_padded = ifft2(ifftshift(shifted_kid_hpf_out))

fruit_hpf_out = fruit_hpf_out_padded[0:600, 0:600]
kid_hpf_out = kid_hpf_out_padded[0:600, 0:600]

plot_im(np.abs(fruit_hpf_out), save=True, name='fruits_hpf_out')
plot_im(np.abs(kid_hpf_out), save=True, name='kid_hpf_out')

# In[7]
'Task (e) top 25 DFT(u,v) pairs from Task (b)'

dic = {'kid':['' for i in range(25)], 'fruit':['' for i in range(25)]}
df = pd.DataFrame.from_dict(dic)
df.index.name = 'index'

fruits_left_fr_reg = np.abs(fruits_hat[0:600, 0:300])
kid_left_fr_reg = np.abs(kid_hat[0:600, 0:300])

sort_fruits = sorted(fruits_left_fr_reg.flatten())[-25:]
sort_kid = sorted(kid_left_fr_reg.flatten())[-25:]

count_row_fruits = 0
count_row_kid = 0

'searching from top left corner'
for u in range(fruits_left_fr_reg.shape[0]):
    for v in range(fruits_left_fr_reg.shape[1]):
        if fruits_left_fr_reg[u,v] in sort_fruits:
            df.loc[count_row_fruits, 'fruit'] = str(u) + ', ' +  str(v)
            count_row_fruits += 1
            
        if kid_left_fr_reg[u,v] in sort_kid:
            df.loc[count_row_kid, 'kid'] = str(u) + ', ' +  str(v)
            count_row_kid += 1

df.to_csv('table.csv', index=True)