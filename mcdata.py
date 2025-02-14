import os,sys
import errno,glob
import h5py as h5
import numpy as np
import time
from scipy import stats,interpolate,integrate
import scipy.ndimage as sn

sys.path.insert(0,"/home/561/gpp547/AxionMC")
import gadget as ga

def save_profile_data(snap_base,stime,outpath,fof=False,nhalos=10,minp=True,minparts=100,newrad=True,rthres=0.9):
    print("Saving Minicluster data ...")

    snap = str(snap_base)+'/snap_%.3d'%stime
    f_fof  = str(snap_base)+'/fof_subhalo_tab_%.3d'%stime
    disc_count = 0

    with h5.File(f_fof+'.hdf5','r') as fof_file:
        soft_length = fof_file['Parameters'].attrs['SofteningComovingClass1']
        redshift = fof_file['Header'].attrs['Redshift']
        boxsize = fof_file['Header'].attrs['BoxSize']
        hubble = fof_file['Parameters'].attrs['HubbleParam']

    head, pp    = ga.load_particles(snap,verbose=True) # pp holds pos,vel,mass,ID
    hfof, halos = ga.load_halos(f_fof,fof=fof,radius='R200',verbose=True) # halos holds pos,vel,mass,rad,size(+veldisp,spin)

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


    # Filter the halos by particle mass, then order 
    if minp == True:
        pmask = np.where(halos[4]>minparts)
        halopos = halopos[pmask]
        halomass = halomass[pmask]
        trad = rad[pmask]
        mmask = np.argsort(halomass)[::-1]
        halopos   = halopos[mmask]
        halomass  = halomass[mmask]
        nrad      = trad[mmask]
        num_halos = len(nrad)
    else:
        mask      = np.argsort(halomass)[::-1]
        halopos   = halopos[mask,:]
        halomass  = halomass[mask]
        rad       = rad[mask]
        num_halos = nhalos

    rr   = nrad*(1+redshift)
    rmin = 2*soft_length

    f = h5.File(str(outpath)+'/sub_z_'+str(stime)+'.hdf5', 'w')

    f_head = f.create_group('Header')
    f_head.attrs['Redshift'] = redshift
    f_head.attrs['Softening'] = soft_length
    f_head.attrs['BoxSize'] = boxsize

    for i in range(num_halos):
        start= time.time()

        x,y  = ga.dens_profile(pos-halopos[i:i+1,:],mass,boxsize,rmin,rr[i], nbins=200)
        x = np.array(x)
        y = np.array(y)*(1+redshift)**3

        srho = sn.gaussian_filter(y, sigma=2)
        filt = np.array(np.where(y==0.)).size

        if np.sum(np.diff(srho) >= 0)/len(y)>0.05 or filt > 0:
            disc_count +=1

        else:
            f_halo  = f.create_group('Halo_'+str(i))
            f_halo.attrs['Radius'] = nrad[i]
            f_halo.attrs['Mass'] = halomass[i]
            f_halo.attrs['x'] = halopos[i:i+1,0]
            f_halo.attrs['y'] = halopos[i:i+1,1]
            f_halo.attrs['z'] = halopos[i:i+1,2]
            f_halo.create_dataset('r', data=x)
            f_halo.create_dataset('rho', data=y)

            pl_params = ga.fit_pl(x*nrad[i], y)
            f_halo.create_dataset('pl_fit', data=pl_params)

            nfw_params = ga.fit_nfw(x*nrad[i], y)
            f_halo.create_dataset('nfw_fit', data=nfw_params)

        print("Halo %d/%d took %d seconds"%(i+1,num_halos,time.time()-start))

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
    print("Filtered %d halo profiles with substructure!"%(disc_count))



def save_isolated(snap_base, stime, outpath, minparts=200):

    snap = str(snap_base)+'/snap_%.3d'%stime
    f_fof  = str(snap_base)+'/fof_subhalo_tab_%.3d'%stime

    with h5.File(f_fof+'.hdf5','r') as fof_file:
        soft_length = fof_file['Parameters'].attrs['SofteningComovingClass1']
        redshift = fof_file['Header'].attrs['Redshift']
        boxsize = fof_file['Header'].attrs['BoxSize']
        hubble = fof_file['Parameters'].attrs['HubbleParam']

    head, pp         = ga.load_particles(snap,verbose=True) # pp holds pos,vel,mass,ID
    hfof, halos, iso = ga.load_isolated(f_fof) # halos holds pos,mass,rad,size

    pos       = pp[0]
    mass      = pp[2]
    halopos   = halos[0]
    halomass  = halos[1]
    size      = halos[3]

    rad   = ga.get_radii(snap_base, stime)
    rad   = rad[iso]

    pmask = np.where(halos[3]>minparts)
    halopos = halopos[pmask]
    halomass = halomass[pmask]
    trad = rad[pmask]
    size = size[pmask]
    mmask = np.argsort(halomass)[::-1]
    halopos   = halopos[mmask]
    halomass  = halomass[mmask]
    nrad      = trad[mmask]
    size      = size[mmask]
    num_halos = len(nrad)

    rr   = nrad*(1+redshift)
    rmin = 2*soft_length

    f = h5.File(str(outpath)+'/iso_z_'+str(stime)+'.hdf5', 'w')

    f_head = f.create_group('Header')
    f_head.attrs['Redshift'] = redshift
    f_head.attrs['Softening'] = soft_length
    f_head.attrs['BoxSize'] = boxsize

    print("Running %d halos"%num_halos)

    for i in range(num_halos):
       start= time.time()

       x,y  = ga.dens_profile(pos-halopos[i:i+1,:],mass,boxsize,rmin,rr[i], nbins=200)
       x = np.array(x)
       y = np.array(y)*(1+redshift)**3

       srho = sn.gaussian_filter(y, sigma=2)
       filt = np.array(np.where(y==0.)).size

       #if np.sum(np.diff(srho) >= 0)/len(y)>0.05 or filt > 0:
       #    disc_count +=1

       #else:
       f_halo  = f.create_group('Halo_'+str(i))
       f_halo.attrs['Radius'] = nrad[i]
       f_halo.attrs['Mass'] = halomass[i]
       f_halo.attrs['Size'] = size[i]
       f_halo.attrs['x'] = halopos[i:i+1,0]
       f_halo.attrs['y'] = halopos[i:i+1,1]
       f_halo.attrs['z'] = halopos[i:i+1,2]
       f_halo.create_dataset('r', data=x)
       f_halo.create_dataset('rho', data=y)

       pl_params = ga.fit_pl(x*nrad[i], y)
       f_halo.create_dataset('pl_fit', data=pl_params)

       nfw_params = ga.fit_nfw(x*nrad[i], y)
       f_halo.create_dataset('nfw_fit', data=nfw_params)

       print("Halo %d/%d took %d seconds"%(i+1,num_halos,time.time()-start))



