import os,sys,errno
import numpy as np
import h5py as h5

# Needs Pylians3 (https://github.com/franciscovillaescusa/Pylians3) and yt 
try:
    import MAS_library as MASL
    import Pk_library as PKL
    import void_library as VL
    import yt
except ImportError:
    pass

import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
 
sys.path.insert(0,"/home/561/gp5547/jaxionsdir_v1.1/jaxions/scripts")
sys.path.insert(0,"/Users/z5278074/pierobon/jaxions/scripts")
sys.path.insert(0, './') 

from params import *
import gadget as ga 
from pyaxions import jaxions as pa

if readmode == True:
    if Nfiles > 1:
        f     = [h5.File(read_base+'_%.3d.hdf5'%i,'r') for i in readlist]
        z     = [f[i]['density'].attrs['redshift'] for i in range(len(readlist))]
        delta = [f[i]['density'] for i in range(len(readlist))]
    else:
        f     = h5.File(read_base+'_%.3d.hdf5'%stime,'r')
        z     = f['density'].attrs['redshift']
        delta = f['density']

    print("Read files %s "%f)
else:
    if Nfiles > 1:
        snapshots = [str(snap_base)+'/snap'+'_%.3d'%i for i in range(Nfiles)]
        f = [h5.File(str(snap)+'.hdf5','r') for snap in snapshots]
        z = [f[i]['Header'].attrs['Redshift'] for i in range(Nfiles)]
        L = f[0]['Header'].attrs['BoxSize']
    else:
        snapshot  = str(snap_base)+'/snap'+'_%.3d'%stime
        f = h5.File(str(snapshot)+'.hdf5','r')
        z = f['Header'].attrs['Redshift']
        L = f['Header'].attrs['BoxSize']

BoxSize   = L
grid      = gridsize                 
ptypes    = [1]
MAS       = 'CIC'
verbose   = True
threads   = 4

if readmode == False:
    # CIC creation of delta via MASL library
    if Nfiles > 1:
        delta = [MASL.density_field_gadget(snap, ptypes, grid, MAS, do_RSD=False, axis=0, verbose=verbose) for snap in snapshots]
    else:
        delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD=False, axis=0, verbose=verbose) 
    
   delta /= np.mean(delta, dtype=np.float64) 
   delta -= 1.0

if gloablp == True:
    if Nfiles > 1: raise Exception("Not implemented yet")
    else:
        head, pp = ga.load_particles(snapshot)
        masses = pp[2]
        avdens = np.sum(masses)/L**3
        print("Total mass is %e"%(np.sum(masses)))
        print("Average energy density at z=%d is %e Msun/pc^3"%(z,avdens))
     # TO CONTINUE


##########################################################################
# Postprocessing options (see params.py file)
##########################################################################

if atype == 0:

    print("Plotting projectons in aout/plots ...")
    ga.check_folder('plots',snap_base)
    delta += 1

    if Nfiles > 1:
        for i in range(Nfiles):
            fig,ax = pa.plot.single(size_x=13,size_y=12)
            im = ax.imshow(np.log10(np.mean(delta[i],axis=2)),cmap='cmr.%s'%mtype,vmax=3.5,vmin=-1,origin='lower',extent=[-L/2,L/2,-L/2,L/2]); pa.plot.cbar(im)
            ax.set_xlabel(r'$x~({\rm pc}/h)$'); ax.set_ylabel(r'$y~({\rm pc}/h)$')
            ax.set_title(r'$z=%d$'%z[i])
            fig.savefig('aout/plots/'+str(snap_base)+'/z%d.pdf'%i,bbox_inches='tight')
    else:
        fig,ax = pa.plot.single(size_x=13,size_y=12)
        im = ax.imshow(np.log10(np.mean(delta,axis=2)),cmap='cmr.%s'%mtype,vmax=3.5,vmin=-1,origin='lower',extent=[-L/2,L/2,-L/2,L/2]); pa.plot.cbar(im)
        ax.set_xlabel(r'$x~({\rm pc}/h)$'); ax.set_ylabel(r'$y~({\rm pc}/h)$')
        ax.set_title(r'$z=%d$'%z)
        fig.savefig('aout/plots/'+str(snap_base)+'/z%d.pdf'%stime,bbox_inches='tight')
        
##########################################################################
if atype == 1:
    
    print('Saving volume rendering in aout/render ...')
    ga.check_folder('render',snap_base)

    res  = camres
    bmin = 1e-6
    bmax = 1e3
    
    if Nfiles > 1:
        for i in range(Nfiles):
            ga.vol_render(delta[i],res,bmin,bmax,snap_base,i)
    else:
        ga.vol_render(delta,res,bmin,bmax,snap_base,stime)

##########################################################################
if atype == 2:

    print("Saving power spectrum data in aout/spectrum ...")
    ga.check_folder('spectrum',snap_base)
    axis = 0 

    if Nfiles > 1:
        Pk  = [PKL.Pk(de, BoxSize, axis, MAS, threads, verbose) for de in delta]
        k   = [Pk[i].k3D for i in range(Nfiles)]
        Pk0 = [Pk[i].Pk[:,0] for i in range(Nfiles)]
        
        for i in range(Nfiles):
            np.savetxt('aout/spectrum/'+str(snap_base)+'/ps_%.3d.txt'%i,np.column_stack((k[i],Pk0[i],Pk0[i]*k[i]**3/(2*np.pi**2))))
    else:
        Pk  = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)
        k   = Pk.k3D
        Pk0 = Pk.Pk[:,0]
        np.savetxt('aout/spectrum/'+str(snap_base)+'/ps_%.3d.txt'%stime,np.column_stack((k,Pk0,Pk0*k**3/(2*np.pi**2))))

##########################################################################
if atype == 3:
    print("Saving delta grids in aout/delta ...")
    ga.check_folder('delta',snap_base)
    
    if Nfiles > 1:
        for i in range(Nfiles):
            f = h5.File('aout/delta/'+str(snap_base)+f'/delta_{i}.hdf5', "w")
            dset = f.create_dataset("density",data=delta[i])
            f.close()
    else:
        f = h5.File('aout/delta/'+str(snap_base)+f'/delta_{stime}.hdf5', "w")
        dset = f.create_dataset("density",data=delta)
        f.close()

##########################################################################
if atype == 4:
    print("Saving density distribution data")
    ga.check_folder('dist',snap_base)
    
    if Nfiles == 1: raise Exception('For the delta distribution select multiple files')

    delta += 1

    inc = 1
    N = 512
     
    for i in range(Nfiles):
        h,db = np.histogram(np.log10(delta[i].flatten()),nbins,range=[-6,6])
        dc = (db[1:]+db[0:-1])/2
        
        if dflag:
            print("Taking gradient distribution in the %s direction "%dire)
            if dire == 'N':                                         # North direction (0,0,+1) axis 2          
                diff = np.roll(delta[i], inc,axis=2)-delta[i]
            elif dire == 'S':                                       # South direction (0,0,-1) axis 2
                diff = np.roll(delta[i],-inc,axis=2)-delta[i]
            if dire == 'E':                                         # East direction  (+1,0,0) axis 0
                diff = np.roll(delta[i], inc,axis=0)-delta[i]
            elif dire == 'W':                                       # West direction  (-1,0,0) axis 0
                diff = np.roll(delta[i],-inc,axis=0)-delta[i]
            if dire == 'F':                                         # Front direction (0,+1,0) axis 1
                diff = np.roll(delta[i], inc,axis=1)-delta[i]
            elif dire == 'B':                                       # Back direction  (0,-1,0) axis 1
                diff = np.roll(delta[i],-inc,axis=1)-delta[i]
            elif dire == 'R':                                       # Random direction  
                delt = np.array(delta[i]).flatten()
                dr, diff = ga.density_gradient(delt,N)
                dr, diff = ga.to_numpy(dr,diff)
            elif dire == 'A':                                       # Averaged over six directions
                d1 = np.roll(delta[i], inc,axis=2)-delta[i]
                d2 = np.roll(delta[i],-inc,axis=2)-delta[i]
                d3 = np.roll(delta[i], inc,axis=0)-delta[i]
                d4 = np.roll(delta[i],-inc,axis=0)-delta[i]
                d5 = np.roll(delta[i], inc,axis=1)-delta[i]
                d6 = np.roll(delta[i],-inc,axis=1)-delta[i]
                diff = (d1+d2+d3+d4+d5+d6)/6

            h2,db2 = np.histogram(np.log10(diff.flatten()),nbins,range=[-6,6])
            dc2 = (db[1:]+db[0:-1])/2
            np.savetxt('aout/dist/'+str(snap_base)+'/dist_'+str(i)+'_'+str(dire)+'.txt',np.column_stack((10**dc,h,10**dc2,h2)))
            if dire == 'R':
                np.savetxt('aout/dist/'+str(snap_base)+'/scatter_'+str(i)+'.txt',np.column_stack((dr,diff)))
        else: 
            np.savetxt('aout/dist/'+str(snap_base)+'/dist_'+str(i)+'.txt',np.column_stack((10**dc,h)))
        print("done!")

##########################################################################
if atype == 5:
    print('Saving density distribution data for void/filaments centers')
    ga.check_folder('dist',snap_base)
    if Nfiles > 1: raise Exception('Void density distribution is only available for Nfiles=1')
    
    delta += 1; inc = 1 
    threshold, thresh2, filam = ga.set_thres(vthres,vthres2,fil_flag=False)

    DeltaGrid  = L/512
    rad_list   = [rlist]
    Radii      = DeltaGrid*np.array(rad_list, dtype=np.float32)
    threads1   = 16; threads2   = 4
    V = VL.void_finder(delta, BoxSize, threshold, thresh2, Radii, threads1, threads2, void_field=False, filaments=filam)
    void_pos    = V.void_pos
    
    cmx  = [int(np.round(void_pos[i,0]/L*512)) for i in range(len(void_pos))]
    cmy  = [int(np.round(void_pos[i,1]/L*512)) for i in range(len(void_pos))]
    cmz  = [int(np.round(void_pos[i,2]/L*512)) for i in range(len(void_pos))]
    dist = [delta[cmx[i],cmy[i],cmz[i]]        for i in range(len(void_pos))]
    dist = np.array(dist)
    
    if filam: np.savetxt('aout/dist/'+str(snap_base)+'/fildist_r%d_z%d.txt'%(rad_list[0],stime),dist)
    else:     np.savetxt('aout/dist/'+str(snap_base)+'/vdist_r%d_z%d.txt'%(rad_list[0],stime),dist)
    

##########################################################################
if atype == 6:
    print("Compute void size function ...")
    ga.check_folder('void',snap_base)
    
    if Nfiles > 1: raise Exception('Void size function is only available for Nfiles=1')
    
    DeltaGrid  = L/512
    threshold  = -vthres; thresh2    = -vthres2
    Radii      = DeltaGrid*np.arange(5,67+1,2, dtype=np.float32)
    threads1   = 16; threads2   = 4

    V = VL.void_finder(delta, BoxSize, threshold, thresh2, Radii, threads1, threads2, void_field=False, filaments=False)
    void_pos    = V.void_pos    
    void_radius = V.void_radius 
    VSF_R       = V.Rbins   
    VSF         = V.void_vsf

    print("Saving void finding data in aout/void ...")
    np.savetxt('aout/void/'+str(snap_base)+'/vsf_thr_'+str(vthres)+'_z%d.txt'%stime,np.column_stack((VSF_R,VSF)))

##########################################################################
if atype == 7:
    ga.check_folder('void',snap_base)
    
    if Nfiles> 1:

        filam      = fil_flag
        filam      = False
        if filam == True:
            threshold  = -0.05       # Defaulted to almost average density   
            thresh2    = -vthres2    # Input
            print("Finding filaments volume and density as function of redshift for values %.2f < delta < -0.05 ..."%(-vthres2))
        else:
            threshold  = -vthres     # Input 
            thresh2    = -1.0        # Defaulted to zero energy  
            print("Finding voids volume and density as function of redshift for values delta < %.2f ..."%(-vthres))
        
        DeltaGrid  = L/512
        Radii      = DeltaGrid*np.arange(5,50+1,5, dtype=np.float32)
        threads1   = 16
        threads2   = 4
        void_field = True
        
        if filam == True:
            fv = open('aout/void/'+str(snap_base)+'/ffil_thr_'+str(vthres2)+'_dens.txt','w+') 
        else:
            fv = open('aout/void/'+str(snap_base)+'/fvoid_thr_'+str(vthres)+'_dens.txt','w+') 
        print("Saving void finding data in aout/void ...")
        
        for i in range(Nfiles):
            V = VL.void_finder(delta[i], BoxSize, threshold, thresh2, Radii, threads1, threads2, void_field=void_field, filaments=filam)
            void_pos    = V.void_pos    
            void_radius = V.void_radius 
            VSF_R       = V.Rbins       
            VSF         = V.void_vsf
            if void_field:
                vfield = V.in_void
                vfraction = np.sum(vfield)/(512**3)
            check       = VL.void_safety_check(np.float32(delta[i]), np.float32(void_pos), np.float32(void_radius), np.float32(BoxSize))
            ov          = check.mean_overdensity
            print("\n\nPartial result %f %f %f \n\n"%(z[i],vfraction,1+np.mean(ov)))
            fv.write("%f %f %f\n"%(z[i],vfraction,1+np.mean(ov)))
        fv.close()
    else:

        filam      = fil_flag
        filam      = False
        if filam == True:
            threshold  = -0.05       # Defaulted to almost average density   
            thresh2    = -vthres2    # Input
            print("Finding filaments volume and density as function of redshift for values %.2f < delta < -0.05 ..."%(-vthres2))
        else:
            threshold  = -vthres     # Input 
            thresh2    = -1.0        # Defaulted to zero energy  
            print("Finding voids volume and density as function of redshift for values delta < %.2f ..."%(-vthres))
        
        DeltaGrid  = L/512
        Radii      = DeltaGrid*np.arange(5,50+1,5, dtype=np.float32)
        threads1   = 16
        threads2   = 4
        void_field = True
        
        if filam == True:
            fv = open('aout/void/'+str(snap_base)+'/ffil_thr_'+str(vthres2)+'_dens.txt','w+') 
        else:
            fv = open('aout/void/'+str(snap_base)+'/fvoid_thr_'+str(vthres)+'_dens_z_%d.txt'%stime,'w+') 
        print("Saving void finding data in aout/void ...")
        
        
        V = VL.void_finder(delta, BoxSize, threshold, thresh2, Radii, threads1, threads2, void_field=void_field, filaments=filam)
        void_pos    = V.void_pos    
        void_radius = V.void_radius 
        VSF_R       = V.Rbins       
        VSF         = V.void_vsf
        if void_field:
            vfield = V.in_void
            vfraction = np.sum(vfield)/(512**3)
        check       = VL.void_safety_check(np.float32(delta), np.float32(void_pos), np.float32(void_radius), np.float32(BoxSize))
        ov          = check.mean_overdensity
        print("\n\nPartial result %f %f %f \n\n"%(z,vfraction,1+np.mean(ov)))
        fv.write("%f %f %f\n"%(z,vfraction,1+np.mean(ov)))
        fv.close()


##########################################################################
if atype == 8:
    print("Computing density profiles of MC halos ...")
    ga.check_folder('profile',snap_base)
    
    if Nfiles > 1:
        raise Exception('Density profile estimation is only available for one file, set Nfiles=1')

    snap = str(snap_base)+'/snap_00'+str(stime)
    f_fof  = str(snap_base)+'/fof_subhalo_tab_00'+str(stime)
    
    with h5.File(f_fof+'.hdf5','r') as fof_file:
        soft_length = fof_file['Parameters'].attrs['SofteningComovingClass1']  
    additional  = False

    head, pp    = ga.load_particles(snap,verbose=False) # pp holds pos,vel,mass,ID
    hfof, halos = ga.load_halos(f_fof,fof=fof_flag,radius=rad_type,additional=additional,verbose=True) # halos holds pos,vel,mass,rad,size(+veldisp,spin)
    
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
    rmin      = r_min*soft_length

    for i in range(num_halos):
        print('Computing profile %d/%d ... '%(i+1,num_halos))
        x,y  = ga.dens_profile(pos-halopos[i:i+1,:],mass,BoxSize,rmin,rad[i]) 
        x = np.array(x); y = np.array(y)
        if fof_flag:
            np.savetxt('aout/profile/'+str(snap_base)+'/f_'+str(stime)+'_gr_'+str(i)+'.txt',np.column_stack([x, y]))
	else:
            np.savetxt('aout/profile/'+str(snap_base)+'/f_'+str(stime)+'_sub_'+str(i)+'.txt',np.column_stack([x, y]))
          
##########################################################################
if atype == 9:
    print("Computing density profiles of voids ...")
    ga.check_folder('vprofile',snap_base)
    ga.check_folder('vprofile/raw',snap_base)

    if Nfiles > 1:
        raise Exception('Density profile estimation for voids is only available for one file, set Nfiles=1')
     
    num_voids  = nhalos
    threshold  = -vthres 
    thresh2    = -vthres2    
    DeltaGrid  = L/512
    rad_list   = [rlist]
    Radii      = DeltaGrid*np.array(rad_list, dtype=np.float32)
    threads1   = 16
    threads2   = 4
    void_field = False
    filam      = False
    
    V = VL.void_finder(delta, BoxSize, threshold, thresh2, Radii, threads1, threads2, void_field=void_field, filaments=filam)
    void_pos    = V.void_pos    
    void_radius = V.void_radius

    snap = str(snap_base)+'/snap_%.3d'%stime
    head, pp    = ga.load_particles(snap,verbose=True) 
    rmin = BoxSize/512
    rad  = 20*void_radius 
    pos  = pp[0]
    mass = pp[2]
    
    rho_list = []
    #nvoids   = min(num_voids,len(rad))
    nvoids   = len(rad) # All possible voids 

    for i in range(nvoids):
        print('Computing void profile %d/%d ... '%(i+1,nvoids))
        x,y  = ga.dens_profile(pos-void_pos[i:i+1,:],mass,BoxSize,rmin,rad[i],nbins=200) 
        x = np.array(x); y = np.array(y)
        #np.savetxt('aout/vprofile/raw/'+str(snap_base)+'/void_z_'+str(stime)+'_thr_'+str(vthres)+'_r_'+str(rlist)+'_'+str(i)+'.txt',np.column_stack([x, y]))
        rho_list += [y]

    y_mean = np.mean(rho_list,axis=0)
    y_err  = np.std(rho_list,axis=0)/np.sqrt(nvoids)

    np.savetxt('aout/vprofile/'+str(snap_base)+'/vprof_z_'+str(stime)+'_thr_'+str(vthres)+'_r_'+str(rlist)+'.txt',np.column_stack([x, y_mean, y_err]))

##########################################################################
if atype == 10:
    print("Density variation plot")
    ga.check_folder('dist',snap_base)
    
    if Nfiles > 1: raise Exception('Select only one snapshot! Set Nfiles=1')
    N = 512

    max_step = 15
    Ntraj    = 5000

    dr = np.arange(max_step+1)
    inc  = 0
    axis = 0

    rlist = np.random.randint(0,N**3+1, size=Ntraj)
    li = []
    count = 0
    for i in rlist:
        dire = np.random.randint(0,5)
        if dire == 0:
            axis = 2
        elif dire == 1:
            axis = 2
        elif dire == 4:
            axis = 1
        elif dire == 5:
            axis = 1

        if dire == 0 or dire == 2 or dire == 4:
            field = [np.roll(delta,j,axis=axis) for j in np.arange(max_step+1)]
        if dire == 1 or dire == 3 or dire == 5:
            field = [np.roll(delta,-j,axis=axis) for j in np.arange(max_step+1)]

        field = [field[j].flatten() for j in np.arange(max_step+1)]
        line = [field[j][i] for j in np.arange(max_step+1)]
        li.append(line)
        count += 1

    lines = np.array(li)


    fig,ax = pa.plot.single('$\Delta r$/(0.4 mpc)',r'Density variation, $\sigma_\delta(\Delta r)$',size_y=10,lfs=35)
    
    def line_background(lw,col):
        return [pe.Stroke(linewidth=lw, foreground=col), pe.Normal()]
    
    mathpazo = True
    
    if mathpazo:
        plt.rcParams.update({
            "text.usetex":True,
            "font.family":"serif",
            "font.serif":["Palatino"],
        })

    nbins = 60

    stdrho50 = np.zeros((max_step))
    stdrho84 = np.zeros((max_step))
    stdrho16 = np.zeros((max_step))
    stdrho05 = np.zeros((max_step))
    stdrho95 = np.zeros((max_step))

    for j in range(1,max_step):
        #rho_std = 100*np.std(lines[:,0:j+1],axis=1) # alternative definition
        rho_std = 100*(np.amax(lines[:,0:j+1],axis=1)-np.amin(lines[:,0:j+1],axis=1))
        
        stdrho50[j] = np.percentile(rho_std,50)
        stdrho84[j] = np.percentile(rho_std,84)
        stdrho16[j] = np.percentile(rho_std,16)
        stdrho05[j] = np.percentile(rho_std,5)
        stdrho95[j] = np.percentile(rho_std,95)

        h,drho_bins = np.histogram(np.log10(rho_std),bins=nbins,range=[-2,3])
        plt.pcolormesh(np.array([0.25,1.5/2])+dr[j],10.0**drho_bins[0:-1],np.column_stack((h,h)),cmap=cm.RdPu,zorder=-10)
        plt.pcolormesh(np.array([0.25,1.5/2])+dr[j],10.0**drho_bins[0:-1],np.column_stack((h,h)),cmap=cm.RdPu,zorder=-10)
        plt.pcolormesh(np.array([0.25,1.5/2])+dr[j],10.0**drho_bins[0:-1],np.column_stack((h,h)),cmap=cm.RdPu,zorder=-10)

    plt.plot(dr[1:-1]+0.5,stdrho50[1:],'o-',color='w',lw=3,path_effects=line_background(4,'k'))
    plt.plot(dr[1:-1]+0.5,stdrho84[1:],'--',color='w',lw=3,path_effects=line_background(4,'k'))
    plt.plot(dr[1:-1]+0.5,stdrho16[1:],'--',color='w',lw=3,path_effects=line_background(4,'k'))
    # plt.plot(dr[1:-1]+0.5,stdrho95[1:],'--',color='w',lw=2,path_effects=line_background(3,'k'))
    # plt.plot(dr[1:-1]+0.5,stdrho05[1:],'--',color='w',lw=2,path_effects=line_background(3,'k'))

    plt.yscale('log')

    plt.ylim([3e-2,8e2])
    plt.xlim(left=1)

    s2year = 1/(365*24*3600)
    km2pc = 3.24078e-14
    dx = 0.4*1e-3
    v = 246.0

    x_min,x_max = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim([x_min*dx/(v*km2pc/s2year),x_max*dx/(v*km2pc/s2year)])
    ax2.set_xlabel('Observation time [years]',labelpad=10,fontsize=35)        
    ax2.tick_params(which='major',direction='out',width=2.5,length=13,pad=10)
    ax2.tick_params(which='minor',direction='out',width=1,length=10)

    ax.set_yticks([0.1,1,10,100])
    ax.set_yticklabels(['0.1\%','1\%','10\%','100\%'])

    fig.savefig('aout/varplot16.png',bbox_inches='tight',dpi=300)
        
print("done!")


