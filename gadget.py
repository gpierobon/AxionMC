import os,sys
import errno,glob
import h5py as h5
import numpy as np
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

def boundf(dire,gridsize,totmass,skip=2,catalog='fof',masstype='samemass'):
     
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
                h, out = load_halos(fof_tab[i],fof=True,radius='R200',additional=False,verbose=False) 
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

def dens_profile(x,mass,L,rmin,rad,nbins=50):
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

def HMF(massarr,minv,maxv,num):
    y,x,_ = stats.binned_statistic(massarr,massarr,statistic='count', bins=np.logspace(minv, maxv, num=num))
    return x,y


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
    return np.sqrt(qq[-1])

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

