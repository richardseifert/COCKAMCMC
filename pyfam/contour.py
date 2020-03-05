import numpy as np
import matplotlib.pyplot as plt

#adding in the contour script
def multidigitize(x, y, bins_x,bins_y):
    d_x = np.digitize(x.flat, bins_x)
    d_y = np.digitize(y.flat, bins_y)
    return d_x, d_y

def linlogspace(xmin,xmax,n):
    return np.logspace(np.log10(xmin),np.log10(xmax),n)

#here's the contour actual values
def contour(x, y,
            bins=10,
            threshold=5,
            marker='.',
            marker_color=None,
            ncontours=5,
            fill=False,
            mesh=False,
            contourspacing=linlogspace,
            mesh_alpha=0.5,
            norm=None,
            axis=None,
            cmap=None,
            **kwargs):
    if axis is None:
        axis = plt.gca()

    
    if hasattr(bins, 'ndim') and bins.ndim == 2:
        nbins_x, nbins_y = bins.shape[0]-1, bins.shape[1]-1
    else:
        try:
            nbins_x = nbins_y = len(bins)-1
        except TypeError:
            nbins_x = nbins_y = bins

    ok = np.isfinite(x)*np.isfinite(y)

    H, b_x, b_y = np.histogram2d(x[ok], y[ok], bins = bins)
    
    d_x, d_y = multidigitize(x[ok], y[ok], b_x, b_y)
    
    plottable = np.ones([nbins_x+2, nbins_y+2], dtype = 'bool')
    plottable_hist = plottable[1:-1, 1:-1]
    assert H.shape == plottable_hist.shape
    
    plottable_hist[H > threshold] = False
    
    H[plottable_hist] = 0
    
    toplot = plottable[d_x, d_y]
    
    c_x = (b_x[1:]+b_x[:-1])/2
    c_y = (b_y[1:]+b_y[:-1])/2
    levels = contourspacing(threshold-0.5, H.max(), ncontours)
    
    if cmap is None:
        cmap = plt.cm.get_cmap()
        cmap.set_under((0,0,0,0))
        cmap.set_bad((0,0,0,0))
    
    if fill:
        con = axis.contourf(c_x, c_y, H.T, levels= levels, norm = norm, cmap = cmap,  **kwargs)
    else: 
        con = axis.contour(c_x, c_y, H.T,levels=levels,norm=norm,cmap=cmap,**kwargs) 
    if mesh: 
        mesh = axis.pcolormesh(b_x, b_y, H.T, **kwargs)
        mesh.set_alpha(mesh_alpha)
        #Is there a way to add lines w the contour levels?
    
    if 'linestyle' in kwargs:
        kwargs.pop('linestyle')
        
    return c_x, c_y, H, x[ok][toplot], y[ok][toplot]


if __name__ == "__main__":
    N = 5000000
    x = np.random.normal(0, 1, N)
    y = x+np.random.normal(0, 1, N)


    fig, ax = plt.subplots()
    #ax.scatter(x, y)

    contour(x, y, bins = 100, fill = True, ncontours = 7, threshold = 50, axis = ax)
