import os,sys
import errno,glob
import h5py as h5
import numpy as np
import pandas as pd
import scipy
import tqdm
from scipy import stats,interpolate

def check_folder(foldname,snap_base):
    try:
        os.mkdir('aout/'+str(foldname)+'/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def set_thres(vthres,vthres2,fil_flag):
    filam      = fil_flag
    if fil_flag:
        threshold  = -0.05       # Defaulted to almost average density   
        thresh2    = -vthres2    # Input
        print("Finding filaments volume and density as function of redshift for values %.2f < delta < -0.05 ..."%(-vthres2))
    else:
        threshold  = -vthres     # Input 
        thresh2    = -1.0        # Defaulted to zero energy  
        print("Finding voids volume and density as function of redshift for values delta < %.2f ..."%(-vthres))
    return vthres, vthres2, fil_flag

def get_info(f):
    """
    Prints info for a given snapshot
    """
    filename = f+'.hdf5'
    fi = h5.File(filename, 'r')
    print('File: ',fi, '\n')
    print("Redshift: %.1f"%fi['Header'].attrs['Redshift'])
    print("NumParts: %d^3"%int(np.ceil(fi['Header'].attrs['NumPart_Total'][1]**(1/3))))
    print("Box Size: %.5f pc"%fi['Header'].attrs['BoxSize'])


def find_radius(dist, glen, nbin, thres=0.9):
    rbins = np.geomspace(np.sort(dist)[1],np.max(dist),nbin)
    r_r, r_bin = np.histogram(dist,bins=rbins)

    for i in range(nbin-1,0,-1):
        frac = np.sum(r_r[:i])/glen
        if frac <=thres:
            return r_bin[i], frac


def get_ids(pref, time):
    head, pp = load_particles(pref+'/snap_%.3d'%time, verbose=False)
    xpos, ypos, zpos = pp[0][:,0], pp[0][:,1], pp[0][:,2]
    ids = pp[3]
    df = pd.DataFrame(data = {'x': xpos, 'y': ypos, 'z': zpos}, index=ids)
    hdata = h5.File(pref+'/fof_subhalo_tab_%.3d.hdf5'%time, 'r')
    glen = np.array(hdata['Group/GroupLen'])
    goff = np.array(hdata['Group/GroupOffsetType'])[:,1]
    #print("Getting halo ids at %d ..."%time)
    halo_ids = {}
    for j in range(len(glen)):
        halo_id = j
        particle_ids = ids[goff[j]:goff[j]+glen[j]]
        halo_ids[halo_id] = particle_ids

    halo_ids = {key: set(value) for key, value in halo_ids.items()}
    return df, halo_ids

def get_radii(base, file, thres=0.9):
    if os.path.exists(base+'/subhalo_rads_%.3d.txt'%file):
        print("Read from cache!")
        return np.loadtxt(base+'/subhalo_rads_%.3d.txt'%file)
    head, pp = load_particles(base+'/snap_%.3d'%file, verbose=False)
    hfof, halos = load_halos(base+'/fof_subhalo_tab_%.3d'%file, fof=False,verbose=False)
    data = h5.File(base+'/fof_subhalo_tab_%.3d.hdf5'%file, 'r')
    xpos, ypos, zpos = pp[0][:,0], pp[0][:,1], pp[0][:,2]
    ids = pp[3]
    df = pd.DataFrame(data = {'x': xpos, 'y': ypos, 'z': zpos}, index=ids)
    glen = np.array(data['Subhalo/SubhaloLen'])
    goff = np.array(data['Subhalo/SubhaloOffsetType'])[:,1]
    red = head['Redshift']
    rads = []
    for i in range(len(halos[2])):
        halo0_ids = ids[goff[i]:goff[i]+glen[i]]
        halo_ppos = df.loc[halo0_ids].to_numpy()
        halo_pos = data['Subhalo/SubhaloPos'][i]
        dist = np.sqrt((halo_ppos[:,0]-halo_pos[0])**2+(halo_ppos[:,1]-halo_pos[1])**2+(halo_ppos[:,2]-halo_pos[2])**2)
        rad, frac = find_radius(dist, glen[i], 200, thres=thres)
        cc = data['Subhalo/SubhaloHalfmassRad'][i]/rad*(1+red)
        if cc < 0.1:
            rads.append(data['Subhalo/SubhaloHalfmassRad'][i])
        else:
            rads.append(rad/(1+red))
    return np.array(rads)

def load_isolated(f,):
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'
    print('Extracting halo data from %s'%filename)

    fh = h5.File(filename, 'r')
    head = dict(fh['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)

    pos   = []
    mass  = []
    rad   = []
    size  = []
    out   = []

    nsubs = np.array(fh['Group/GroupNsubs'])
    iso_mask = np.array([np.sum(nsubs[:i[0]]) for i in np.argwhere(nsubs==1)])

    ratio = 100*(len(np.argwhere(nsubs==1))/len(np.array(fh['Subhalo/SubhaloMass'])))
    print("%.1f%% of halos are isolated and loaded"%ratio)

    pos   += [np.array(fh['Subhalo/SubhaloCM'])[iso_mask]]
    mass  += [np.array(fh['Subhalo/SubhaloMass'])[iso_mask]]
    rad   += [np.array(fh['Subhalo/SubhaloHalfmassRad'])[iso_mask]]
    size  += [np.array(fh['Subhalo/SubhaloLen'])[iso_mask]]


    out  += [np.concatenate(pos,axis=1)]
    out  += [np.concatenate(mass)]
    out  += [np.concatenate(rad)]
    out  += [np.concatenate(size)]

    return head, out, iso_mask

def load_merged(f):
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'
    print('Extracting halo data from %s'%filename)

    fh = h5.File(filename, 'r')
    head = dict(fh['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)

    pos   = []
    mass  = []
    rad   = []
    size  = []
    sats  = []
    out   = []

    nsubs = np.array(fh['Group/GroupNsubs'])
    mer_mask = np.argwhere(nsubs>1)

    pos   += [np.array(fh['Group/GroupPos'])[mer_mask]]
    mass  += [np.array(fh['Group/GroupMass'])[mer_mask]]
    rad   += [a*np.array(fh['Group/Group_R_Crit200'])[mer_mask]]
    size  += [np.array(fh['Group/GroupLen'])[mer_mask]]
    sats  += [np.array(fh['Group/GroupNsubs'])[mer_mask]]

    out  += [np.concatenate(pos,axis=1)]
    out  += [np.concatenate(mass)]
    out  += [np.concatenate(rad)]
    out  += [np.concatenate(size)]
    out  += [np.concatenate(sats)]

    return head, out, mer_mask



def load_halos(f,fof=True,radius='R200',isolated=False,additional=False,verbose=True):
    """
    Loads fof data for a given fof tab
    Used for profiles of MC halos or general properties
    """
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'

    if verbose:
        if fof:
            print('Extracting FOF data from %s'%filename)
        else:
            print('Extracting SubHalo data from %s'%filename)

    fi = h5.File(filename, 'r')
    head = dict(fi['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)
    pos   = []
    vel   = []
    mass  = []
    rad   = []
    size  = []
    out   = []

    if fof:
        if isolated == True:
            nsubs = np.array(fi['Group/GroupNsubs'])
            iso_ind = np.where(nsubs==1)

            pos_arr = np.array(fi['Group/GroupPos'])
            vel_arr = np.array(fi['Group/GroupVel'])*np.sqrt(a)
            mass_arr = np.array(fi['Group/GroupMass'])
            rad_arr = np.array(fi['Group/Group_R_Crit200'])
            size_arr = np.array(fi['Group/GroupLen'])

            pos += [np.array(pos_arr[iso_ind])]
            vel += [np.array(vel_arr[iso_ind])]
            mass += [np.array(mass_arr[iso_ind])]
            rad += [np.array(rad_arr[iso_ind])]
            size += [np.array(size_arr[iso_ind])]

        else:
            pos  += [np.array(fi['Group/GroupPos'])]
            vel  += [np.array(fi['Group/GroupVel'])*np.sqrt(a)]
            mass += [np.array(fi['Group/GroupMass'])]
            if radius == 'R200':
                rad += [np.array(fi['Group/Group_R_Crit200'])]
            elif radius == 'R500':
                rad += [np.array(fi['Group/Group_R_Crit500'])]
            elif radius == 'RMean':
                rad += [np.array(fi['Group/Group_R_Mean200'])]
            elif radius == 'TopHat':
                rad += [np.array(fi['Group/Group_R_TopHat200'])]
            else:
                raise ValueError('Selected radius is unknown')

            size += [np.array(fi['Group/GroupLen'])]

    else:
        pos   += [np.array(fi['Subhalo/SubhaloCM'])]
        vel   += [np.array(fi['Subhalo/SubhaloVel'])*np.sqrt(a)]
        mass  += [np.array(fi['Subhalo/SubhaloMass'])]
        rad   += [np.array(fi['Subhalo/SubhaloHalfmassRad'])]
        size  += [np.array(fi['Subhalo/SubhaloLen'])]

        if additional:
            vdisp = []
            spin  = []
            vdisp += [np.array(fi['Subhalo/SubhaloVelDisp'])]
            spin  += [np.array(fi['Subhalo/SubhaloSpin'])]


    out  += [np.concatenate(pos,axis=1)]
    out  += [np.concatenate(vel,axis=1)]
    out  += [np.concatenate(mass)]
    out  += [np.concatenate(rad)]
    out  += [np.concatenate(size)]

    if additional:
        out += [np.concatenate(vdisp)]
        out += [np.concatenate(spin)]

    if verbose:
        if additional:
            print('At z=%d, %d halos loaded: data is stored in header,pos,vel,mass,radius,size,vdisp,spin'%(z,len(pos[0])))
        else:
            print('At z=%d, %d halos loaded: data is stored in header,pos,vel,mass,radius,size'%(z,len(pos[0])))

    return head,out

def iso_ratio(f):
    """
    """
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'

    fi = h5.File(filename, 'r')
    z = fi['Header'].attrs['Redshift']
    nsubs = np.array(fi['Group/GroupNsubs'])
    iso = np.array(np.where(nsubs == 1)).size
    isoratio = iso/nsubs.size
    return z,isoratio

def get_isolated_masses(f):
    """
    """
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'

    fi = h5.File(filename, 'r')
    nsubs_arr = np.array(fi['Group/GroupNsubs'])
    gr_mass_arr = np.array(fi['Group/GroupMass'])
    iso_ind = np.where(nsubs_arr==1)
    iso_arr = np.array(gr_mass_arr[iso_ind])
    return iso_arr

def get_merged_masses(f,nsubs_min=10):
    """
    """
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'

    fi = h5.File(filename, 'r')
    nsubs_arr = np.array(fi['Group/GroupNsubs'])
    gr_mass_arr = np.array(fi['Group/GroupMass'])
    merged_ind = np.where(nsubs_arr>nsubs_min)
    merged_arr = np.array(gr_mass_arr[merged_ind])
    return merged_arr


def select_halos(f,Nhalos=100,verbose=True):
    """
    Loads halo data for a given fof tab
    Selects largest subhalos inside groups
    """
    if '.hdf5' in f:
        filename = str(f)
    else:
        filename = str(f)+'.hdf5'

    if verbose:
            print('Selecting halos from %s'%filename)

    fi = h5.File(filename, 'r')
    head = dict(fi['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)
    ngroups = head['Ngroups_Total']
    nsubs = head['Nsubhalos_Total']
    if verbose:
        print("Found %d groups and %d subhalos"%(ngroups,nsubs))

    '''
    nsubs = # Determine if group has a subhalo
    3mass = np.array(fi['Group/GroupMass'])
    sort_ind  = np.where(mass> 1e-12)
    mass = mass[sort_ind]
    return mass
    #pos   = []
    #vel   = []
    #mass  = []
    #rad   = []
    #size  = []
    #out   = []

    #fi[]

    #if fof:
        #pos  += [np.array(fi['Group/GroupPos'])]
        #vel  += [np.array(fi['Group/GroupVel'])*np.sqrt(a)]
    #    mass += [np.array(fi['Group/GroupMass'])]
    #    rad += [np.array(fi['Group/Group_R_Mean200'])]
    '''

def load_particles(f,verbose=True):
    """
    Loads particle data for a given snapshot
    Usage:
    pos,vel,mass,ID = load_particles('/path/to/snap',verbose=True)

    """
    filename = str(f)+'.hdf5'

    if verbose:
        print('Extracting particle data from %s'%filename)

    fi = h5.File(filename, 'r')
    head = dict(fi['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)
    mtab = head['MassTable']
    nparts = head['NumPart_Total'][1]

    pos  = np.zeros((3,np.sum(nparts)),dtype=np.float32)
    vel  = np.zeros((3,np.sum(nparts)),dtype=np.float32)
    mass = np.zeros(np.sum(nparts),dtype=np.float32)
    ID   = np.zeros(np.sum(nparts),dtype=np.uint32)
    out  = []

    pos = np.array(fi['PartType1/Coordinates'])
    vel = np.array(fi['PartType1/Velocities'])*np.sqrt(a)

    if mtab[1] == 0.:
        mass = np.array(fi['PartType1/Masses'])
    else:
        mass = np.full(nparts,mtab[1])

    ID = np.array(fi['PartType1/ParticleIDs'])

    out += [pos]
    out += [vel]
    out += [mass]
    out += [ID]
    if verbose:
        print('%d particles loaded: data is stored in header,pos,vel,mass,ID'%nparts)

    return head,out

def boundf(dire,gridsize,totmass,skip=2,catalog='fof',masstype='samemass',isolated=False):

    fof_tab = []
    for filename in glob.iglob(dire+'/fof_subhalo_tab_*', recursive=True):
            fof_tab.append(filename)
    fof_tab = sorted(fof_tab)
    ff = [h5.File(fof_tab[i],'r') for i in range(len(fof_tab))]
    z =  [ff[i]['Header'].attrs['Redshift'] for i in range(len(fof_tab))]
    if masstype == 'samemass':
        print('Bound fraction from z=%d to z=%d'%(z[0],z[-1]))
    else:
        print('Bound fraction from z=%d to z=%d'%(z[skip],z[-1]))
    if masstype == 'diffmass':
        zli = []
        hli = []
        for i in range(skip,len(fof_tab)):
            if catalog == 'fof':
                h, out = load_halos(fof_tab[i],fof=True,isolated=isolated,radius='R200',additional=False,verbose=False)
            else:
                h, out = load_halos(fof_tab[i],fof=False,radius='R200',additional=False,verbose=False)
            halomass = np.sum(out[2])
            hli.append(halomass/totmass)
            zli.append(z[i])
        hli = np.array(hli)
        z   = np.array(zli)
        bound = np.column_stack([z,hli])

    elif masstype == 'samemass':
        if catalog == 'fof':
            halo  = np.array([np.sum(np.array(ff[i]['Group/GroupLen']))/gridsize**3 for i in range(len(fof_tab))])
            bound = np.column_stack([z,halo])
        elif catalog == 'subfind':
            halo = np.array([np.sum(np.array(ff[i]['Subhalo/SubhaloLen']))/gridsize**3 for i in range(len(fof_tab))])
            bound = np.column_stack([z,halo])
        else:
            raise ValueError("Unknown option: select fof or subfind")

    else:
        raise ValueError("Unknown option: select samemass or diffmasss")

    return bound

def dens_profile(x,mass,L,rmin,rad,nbins=200):
    '''
    Computes profiles
    '''
    nparts = x.shape[0]
    x[x >= 0.5*L] -= L
    x[x < -0.5*L] -= L

    r    = np.sqrt(np.sum(x.T**2,axis=0))
    bins = np.geomspace(rmin,rad,nbins)
    if bins[1]<bins[0]:
        bins = bins[::-1]

    bvol = 4./3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    hist_mass,hbins = np.histogram(r,bins=bins,weights=mass)

    r_out = 0.5*(bins[1:]+bins[:-1])
    rho_out = hist_mass/bvol

    return r_out/rad, rho_out

#def HMF(massarr,minv,maxv,num):
#    y,x,_ = stats.binned_statistic(massarr,massarr,statistic='count', bins=np.logspace(minv, maxv, num=num))
#    return x,y

def HMF(massarr,num=70):
    minv = np.min(massarr)
    maxv = np.max(massarr)
    y,x,_ = stats.binned_statistic(massarr,massarr,statistic='count', bins=np.geomspace(minv, maxv, num=num))
    return x,y

def fit_pl(r,rho):
    pl_profile = lambda x, rho0, r0, alpha: rho0*(x/r0)**alpha
    minimization = lambda x: np.sqrt(np.sum(np.abs(np.log10(pl_profile(r,x[0],x[1],x[2])/rho))**2))
    x0 = [1e3,1e-6,-3]
    method = 'Nelder-Mead'
    fit = scipy.optimize.minimize(minimization,x0,method=method)
    return fit.x[0], fit.x[1], fit.x[2]

def fit_nfw(r,rho): # my own fitting routine, seems to give more reliable results than curve_fit

    nfw_profile = lambda x, rho0, rs: rho0/( (x/rs) * (1 + x/rs)**2)
    minimization = lambda x: np.sqrt(np.sum(np.abs(np.log10(nfw_profile(r,x[0],x[1])/rho))**2))
    x0 = [1e3, 5e-7]
    method = 'Nelder-Mead' #'L-BFGS-B'
    #xbnd = [[1e-5,1], [5e-8,5e-4]]
    #fit = scipy.optimize.minimize(minimization,x0,method=method,bounds=xbnd) #use this for L-BFGS-B method
    fit = scipy.optimize.minimize(minimization,x0,method=method)
    return fit.x[0], fit.x[1]

def inte(f):
    x = f[:,0]; y = f[:,1]
    finterp = interpolate.InterpolatedUnivariateSpline(x, y, k=1)
    xx = np.linspace(x[0], x[-1], 5*len(x))
    qq = [finterp.integral(0, t) for t in xx]
    return qq[-1]

def alpha(f):
    x = f[:,0]; y = f[:,1]*x**4*4*np.pi
    finterp = interpolate.InterpolatedUnivariateSpline(x, y, k=1)
    xx = np.linspace(x[0], x[-1], 5*len(x))
    qq = [finterp.integral(0, t) for t in xx]
    return qq[-1]

def beta(f):
    x = f[:,0]; y = f[:,1]/x**2
    finterp = interpolate.InterpolatedUnivariateSpline(x, y, k=1)
    xx = np.linspace(x[0], x[-1], 5*len(x))
    qq = [finterp.integral(0, t) for t in xx]
    return np.sqrt(qq[-1])


#def vol_render(de,res,bmin,bmax,snap_base,j):
#    data = dict(density = (de, "g/cm**3"))
#    ds = yt.load_uniform_grid(data, de.shape, length_unit="pc")
#    sc = yt.create_scene(ds, field=("density"))
#    sc.camera.resolution = (res, res)
#    sc.camera.focus = ds.arr([0.3, 0.3, 0.3], "unitary")
#    source = sc[0]
#    source.tfh.set_bounds((bmin, bmax))
#    sc.camera.position = ds.arr([0, 0, 0], "unitary")
#    sc.render()
#    sc.save(f'aout/render/'+str(snap_base)+f'/shell_{j}.png', sigma_clip=4)

