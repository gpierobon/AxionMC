import os,sys
import errno,glob
import h5py as h5
import numpy as np
from scipy import stats,interpolate,integrate

sys.path.insert(0,"/home/561/gpp547/AxionMC")
import gadget as ga

def save_mc_data(snap_base,stime,nhalos,fof,outpath):
    print("Saving Minicluster data ...")
    
    snap = str(snap_base)+'/snap_%.3d'%stime
    f_fof  = str(snap_base)+'/fof_subhalo_tab_%.3d'%stime
    
    with h5.File(f_fof+'.hdf5','r') as fof_file:
        soft_length = fof_file['Parameters'].attrs['SofteningComovingClass1']
        redshift    = fof_file['Header'].attrs['Redshift']
        boxsize    = fof_file['Header'].attrs['BoxSize']
    additional  = False

    head, pp    = ga.load_particles(snap,verbose=False) # pp holds pos,vel,mass,ID
    hfof, halos = ga.load_halos(f_fof,fof=fof,radius='R200',additional=additional,verbose=True) # halos holds pos,vel,mass,rad,size(+veldisp,spin)
    
    pos       = pp[0] 
    mass      = pp[2]
    halopos   = halos[0]
    halomass  = halos[2]
    rad       = halos[3]
    num_halos = nhalos 
    
    mask      = np.argsort(halomass)[::-1]
    halopos   = halopos[mask,:]
    halomass  = halomass[mask]
    rad       = rad[mask]
    rmin      = 2*soft_length
    
    if fof == True:
        f = h5.File(str(outpath)+'/fof_z_'+str(stime)+'.hdf5', 'w')
    else:
        f = h5.File(str(outpath)+'/sub_z_'+str(stime)+'.hdf5', 'w')

    for i in range(num_halos):
        print('Analysing halo %d/%d ... '%(i+1,num_halos))
        
        f_halo  = f.create_group('Halo_'+str(i))
        ff_halo = f_halo.create_group('Header')
        ff_halo.attrs['Redshift'] = redshift
        ff_halo.attrs['Softening'] = soft_length
        ff_halo.attrs['Radius'] = rad[i]
        ff_halo.attrs['BoxSize'] = boxsize
        ff_halo.attrs['Rmin'] = rmin
        ff_halo.attrs['Mass'] = halomass[i]
        ff_halo.attrs['x'] = halopos[i:i+1,0]
        ff_halo.attrs['y'] = halopos[i:i+1,1]
        ff_halo.attrs['z'] = halopos[i:i+1,2]

        x,y  = ga.dens_profile(pos-halopos[i:i+1,:],mass,boxsize,rmin,rad[i]) 
        x = np.array(x); y = np.array(y)
        y_phys = y*(1/(1+redshift))**3
        f_halo.create_dataset('r', data=x)
        f_halo.create_dataset('rho', data=y)
        f_halo.create_dataset('rho_phys', data=y_phys)
        
        fi   = np.column_stack((x,y_phys))
        alph = ga.alpha(fi)/(halomass[i]*rad[i]**2)
        ff_halo.attrs['Alpha'] = alph

    f.close()



