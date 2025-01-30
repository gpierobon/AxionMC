
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

try:
    import yt
except ImportError:
    pass

from params import *

def cbar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
         minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")
    return cbar

def single_plot(xlab='',ylab='',\
                 lw=1.5,lfs=25,tfs=18,size_x=13,size_y=8,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)

    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)
    #ax.tick_params(which='major',direction='in',width=0.8,length=8,right=True,top=True,pad=7)
    #ax.tick_params(which='minor',direction='in',width=0.8,length=8,right=True,top=True)
    if Grid: ax.grid()
    return fig,ax

def projection_plot():
    print("Plotting projectons in aout/plots ...")
    #ga.check_folder('plots',snap_base)
    delta += 1

    if Nfiles > 1:
        for i in range(Nfiles):
            fig,ax = single_plot(size_x=13,size_y=12)
            im = ax.imshow(np.log10(np.mean(delta[i],axis=2)),cmap='cmr.%s'%mtype,vmax=3.5,vmin=-1,origin='lower',extent=[-L/2,L/2,-L/2,L/2])
            cbar(im)
            ax.set_xlabel(r'$x~({\rm pc}/h)$'); ax.set_ylabel(r'$y~({\rm pc}/h)$')
            ax.set_title(r'$z=%d$'%z[i])
            fig.savefig('aout/plots/'+str(snap_base)+'/z%d.pdf'%i,bbox_inches='tight')
    else:
        fig,ax = single_plot(size_x=13,size_y=12)
        im = ax.imshow(np.log10(np.mean(delta,axis=2)),cmap='cmr.%s'%mtype,vmax=3.5,vmin=-1,origin='lower',extent=[-L/2,L/2,-L/2,L/2])
        cbar(im)
        ax.set_xlabel(r'$x~({\rm pc}/h)$'); ax.set_ylabel(r'$y~({\rm pc}/h)$')
        ax.set_title(r'$z=%d$'%z)
        fig.savefig('aout/plots/'+str(snap_base)+'/z%d.pdf'%stime,bbox_inches='tight')


def vol_render(de,res,bmin,bmax,snap_base,j):
    data = dict(density = (de, "g/cm**3"))
    ds = yt.load_uniform_grid(data, de.shape, length_unit="pc")
    sc = yt.create_scene(ds, field=("density"))
    sc.camera.resolution = (res, res)
    sc.camera.focus = ds.arr([0.3, 0.3, 0.3], "unitary")
    source = sc[0]
    source.tfh.set_bounds((bmin, bmax))
    sc.camera.position = ds.arr([0, 0, 0], "unitary")
    sc.render()
    sc.save(f'aout/render/'+str(snap_base)+f'/shell_{j}.png', sigma_clip=4)

