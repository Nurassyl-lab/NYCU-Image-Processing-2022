import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
import scipy.ndimage.filters
import matplotlib as mpl
import matplotlib
import pandas as pd
matplotlib.use('Agg')

def pswise_contrast_stretch(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
contr_stretch = np.vectorize(pswise_contrast_stretch)

def Histogram(x):
    a , b = x.shape[0], x.shape[1]
    dic = {float(i):0 for i in range(256)}
    for row in x:
        for pixel in row:
            dic[round(pixel)] += 1
    
    hist = np.array(list(dic.values())) / (a*b)
    return hist

def plot_hist(folder, name, hist):
    plt.figure()
    plt.title(name)
    plt.bar(np.arange(0,256), hist)
    plt.savefig(folder+name+'.png')
    plt.close()

def save_image(data, cm, fn):#save image without white borders ;)
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(3, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, cmap=cm)
    plt.savefig(fn, dpi = 200) 
    plt.close()
    
def project1(picture, name, df):
    #a)original image
    im = np.array(Image.open(picture))
    im = contr_stretch(im, np.min(im), 0, np.max(im), 255)
    save_image(im, 'gray', name+'/'+name+'_a.png')
    
    #b)Laplacian
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9, -1], 
                       [-1, -1, -1]])
    
    laplacian = scipy.ndimage.filters.convolve(im, kernel)
    save_image(contr_stretch(laplacian, np.min(laplacian), 0, np.max(laplacian), 255), 'gray', name+'/'+name+'_b.png')
    
    #c)Laplacian Sharpened
    sharp_im = im + laplacian
    save_image(contr_stretch(sharp_im, np.min(sharp_im), 0, np.max(sharp_im), 255), 'gray', name+'/'+name+'_c.png')
    
    #d)Sobel 
    kernel_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gx = scipy.ndimage.filters.convolve(im, kernel_x)
    Gy = scipy.ndimage.filters.convolve(im, kernel_y)
    Gx_Gy = np.abs(Gx) + np.abs(Gy)
    save_image(contr_stretch(Gx_Gy, np.min(Gx_Gy), 0, np.max(Gx_Gy), 255), 'gray', name+'/'+name+'_d.png')
    
    #e)smoothing
    kernel_smthg = np.ones((5,5))/25
    smoothed = scipy.ndimage.filters.convolve(Gx_Gy, kernel_smthg)
    save_image(contr_stretch(smoothed, np.min(smoothed), 0, np.max(smoothed), 255), 'gray', name+'/'+name+'_e.png')
    
    #f)extracted_feature
    ext_feat = smoothed * laplacian
    save_image(contr_stretch(ext_feat, np.min(ext_feat), 0, np.max(ext_feat), 255), 'gray', name+'/'+name+'_f.png')
    
    #g)a+f
    # g = im + ext_feat
    g = im + contr_stretch(ext_feat, np.min(ext_feat), 0, np.max(ext_feat), 255)
    # g = im + ext_feat
    g = contr_stretch(g, np.min(g), 0, np.max(g), 255)
    save_image(contr_stretch(g, np.min(g), 0, np.max(g), 255), 'gray', name+'/'+name+'_g.png')
    
    #h) power law transform
 
    y = 1.3
    c = 1/5.4

    
    #here g is r
    h = c * (g**(y))
    save_image(h, 'gray', name+'/'+name+'_h.png')
    
    orig = Histogram(im)
    out = Histogram(h)
    
    if name == 'kid':
        for i in range(256):
            df.loc[i, df.columns[1]] = orig[i]
            df.loc[i, df.columns[2]] = out[i]
    else:
        for i in range(256):
            df.loc[i, df.columns[3]] = orig[i]
            df.loc[i, df.columns[4]] = out[i]
    
#input your image
df = pd.read_excel('Histograms.xlsx')
project1('kid blurred-noisy.tif', 'kid', df)
project1('fruit blurred-noisy.tif', 'fruit', df)
df.to_excel("Histograms.xlsx")

plot_hist('kid/', 'hist_kid_orig', np.array(df.loc[:, df.columns[1]]))
plot_hist('kid/', 'hist_kid_out', np.array(df.loc[:, df.columns[2]]))

plot_hist('fruit/', 'hist_fruits_orig', np.array(df.loc[:, df.columns[3]]))
plot_hist('fruit/', 'hist_fruits_out', np.array(df.loc[:, df.columns[4]]))