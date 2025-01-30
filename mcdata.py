import os,sys
import errno,glob
import h5py as h5
import numpy as np
import time
from scipy import stats,interpolate,integrate

sys.path.insert(0,"/home/561/gpp547/AxionMC")
import gadget as ga

def save_profile_data(snap_base,stime,outpath,nhalos=10,minp=True,minparts=100,newrad=True,rthres=0.9):
    print("Saving Minicluster data ...")

    snap = str(snap_base)+'/snap_%.3d'%stime
    f_fof  = str(snap_base)+'/fof_subhalo_tab_%.3d'%stime

    with h5.File(f_fof+'.hdf5','r') as fof_file:
        soft_length = fof_file['Parameters'].attrs['SofteningComovingClass1']
        redshift = fof_file['Header'].attrs['Redshift']
        boxsize = fof_file['Header'].attrs['BoxSize']
        hubble = fof_file['Parameters'].attrs['HubbleParam']
    additional  = False

    print(2*soft_length/(1+redshift))

    head, pp    = ga.load_particles(snap,verbose=True) # pp holds pos,vel,mass,ID
    hfof, halos = ga.load_halos(f_fof,fof=False,radius='R200',additional=additional,verbose=True) # halos holds pos,vel,mass,rad,size(+veldisp,spin)

    pos       = pp[0]
    mass      = pp[2]
    halopos   = halos[0]
    halomass  = halos[2]
    if newrad:
        print("New radii computing ...")
        rad   = ga.get_radii(snap_base, stime, thres=rthres)
        print("done")
    else:
        rad   = halos[3]

    ## Filter the halos by particle mass, then order 
    if minp == True:
        pmask = np.where(halos[4]>minparts)
        halopos = halopos[pmask]
        halomass = halomass[pmask]
        rad = rad[pmask]
        mmask = np.argsort(halomass)[::-1]
        halopos   = halopos[mmask]
        halomass  = halomass[mmask]
        rad       = rad[mmask]
        num_halos = len(rad)
        print(num_halos)
    else:
        mask      = np.argsort(halomass)[::-1]
        halopos   = halopos[mask,:]
        halomass  = halomass[mask]
        rad       = rad[mask]
        num_halos = nhalos
        print(num_halos)

    print(rad[:10])
    print(halomass[:10])

    rmin      = 2*soft_length/(1+redshift)

    f = h5.File(str(outpath)+'/sub_z_'+str(stime)+'.hdf5', 'w')

    f_head = f.create_group('Header')
    f_head.attrs['Redshift'] = redshift
    f_head.attrs['Softening'] = soft_length
    f_head.attrs['BoxSize'] = boxsize


    for i in range(num_halos):
        print('Analysing halo %d/%d ... '%(i+1,num_halos))
        start= time.time()

        f_halo  = f.create_group('Halo_'+str(i))
        f_halo.attrs['Radius'] = rad[i]
        f_halo.attrs['Rcut'] = rmin
        f_halo.attrs['Mass'] = halomass[i]
        f_halo.attrs['x'] = halopos[i:i+1,0]
        f_halo.attrs['y'] = halopos[i:i+1,1]
        f_halo.attrs['z'] = halopos[i:i+1,2]

        x,y  = ga.dens_profile(pos-halopos[i:i+1,:],mass,boxsize,rmin,rad[i])
        x = np.array(x); y = np.array(y)
        y_phys = y*(1+redshift)**3*hubble**3

        f_halo.create_dataset('r', data=x)
        f_halo.create_dataset('rho', data=y)
        f_halo.create_dataset('rho_phys', data=y_phys)

        pl_params = ga.fit_pl(x, y)
        f_halo.create_dataset('pl_fit', data=pl_params)

        nfw_params = ga.fit_nfw(x, y)
        f_halo.create_dataset('nfw_fit', data=nfw_params)
        print("Took %d seconds"%(time.time()-start))

        #fi   = np.column_stack((x*rad[i],y_phys))
        #print("R,M: ",rad[i],halomass[i])
        #print(x*rad[i])
        #fi   = np.column_stack((x,y))
        #alph = ga.alpha(fi)
        #alph = ga.alpha(fi)/(halomass[i])
        #alph = ga.alpha(fi)/(halomass[i]*rad[i]**2)
        #f_halo.attrs['Alpha'] = alph
        #print("Alpha =",alph)

    f.close()



