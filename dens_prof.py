import numpy as np
import sys
import h5py as h5
import g3read as g
import g3matcha as matcha
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic


def find_radius(dist_arr, h_glen, nbin): # routine that determines radius of halo defined by containing 90% of all particles of halo

    rbins = np.logspace(np.log10(1.0/0.7),np.log10(np.max(dist_arr)),nbin)
    r_r, r_bin = np.histogram(dist_arr,bins=rbins)
    
    for i in range(nbin-1,0,-1):
        frac = np.sum(r_r[:i])/h_glen
        if frac <=0.9:
            return r_bin[i], frac

def fit_nfw(r,rho): # my own fitting routine, seems to give more reliable results than curve_fit

    nfw_profile = lambda x, rho0, rs: rho0/( (x/rs) * (1 + x/rs)**2)
    minimization = lambda x: np.sqrt(np.sum(np.abs(np.log10(nfw_profile(r,x[0],x[1])/rho))**2))
    x0 = [1e3, 5e-7]
    method = 'Nelder-Mead' #'L-BFGS-B'
    #xbnd = [[1e-5,1], [5e-8,5e-4]]
    #fit = scipy.optimize.minimize(minimization,x0,method=method,bounds=xbnd) #use this for L-BFGS-B method
    fit = scipy.optimize.minimize(minimization,x0,method=method)
    return fit.x[0], fit.x[1]

def fit_pl(r,rho):

    pl_profile = lambda x, rho0, r0, alpha: rho0*(x/r0)**alpha
    minimization = lambda x: np.sqrt(np.sum(np.abs(np.log10(pl_profile(r,x[0],x[1],x[2])/rho))**2))
    x0 = [1e3,1e-6,-3]
    method = 'Nelder-Mead'
    fit = scipy.optimize.minimize(minimization,x0,method=method)
    return fit.x[0], fit.x[1], fit.x[2]

def compute_densityprofile(dist_arr, radii, nbin, redshift, h, pmass):

    rbins = np.logspace(np.log10(1.0/h),np.log10(radii),nbin)
    r_r, r_bin = np.histogram(dist_arr,bins=rbins)
    vol = 4.0*np.pi/3.0*(r_bin[1:]**3-r_bin[:-1]**3)*(lunit/((1+redshift)*h))**3 # redshift included, i.e. physical volume
    rho_profile = pmass*r_r/vol
    cbins = (r_bin[:-1]+r_bin[1:])/2.0 * lunit/h/(1+redshift) # centered bins in physical units

    index_soft = np.where(cbins >= 4*lunit/h/(1+redshift)) # consider only particles above 4*AU/h (in physical units)
    return cbins[index_soft], rho_profile[index_soft]


# input params

sim = '1024'
idir = '020'
simdir = ['020']
path = '/home/uni09/cosmo/beggeme/minicluster'

munit = 1.0e-15 #in solar masses
lunit = 4.848e-6 #in pc corresponding to 1AU

snapbase = path+'/'+sim+'/snapdir_'+idir+'/snap_'+idir
groupbase = path+'/'+sim+'/groups_'+idir+'/sub_'+idir

f = g.GadgetFile(snapbase + '.0')
h = f.header.HubbleParam
redshift = f.header.redshift
boxsize = f.header.BoxSize*lunit/h # comoving box size in pc
partmass = f.header.mass[1]*munit/h

print('Boxsize in comoving pc: ',boxsize)
print('Redshift: ',redshift)

# read in all halos; consider only halos with substructures (likely MCHs)

mvir = g.read_new(groupbase, 'MMEA', 0, multiple_files=True, is_snap=False)
rvir = g.read_new(groupbase, 'RMEA', 0, multiple_files=True, is_snap=False)
glen = g.read_new(groupbase, 'GLEN', 0, multiple_files=True, is_snap=False)
nsub = g.read_new(groupbase, 'NSUB', 0, multiple_files=True, is_snap=False)
fsub = g.read_new(groupbase, 'FSUB', 0, multiple_files=True, is_snap=False)
slen = g.read_new(groupbase, 'SLEN', 1, multiple_files=True, is_snap=False)
smst = g.read_new(groupbase, 'SMST', 1, multiple_files=True, is_snap=False)

#index_mch = np.where( ( (mvir != 0) & (nsub > 1) ) )
index_mch = np.where( ( (mvir >= 10000*partmass*h/munit) & (nsub > 1) ) ) #10000*partmass for 1024 data
Mvir_mch = np.array(mvir[index_mch])*munit/h # in solar mass
Rvir_mch = np.array(rvir[index_mch])*lunit/h/(1+redshift) # in physical pc
glen_mch = glen[index_mch]

# read in all particle positions and their IDs; create pandas dataframe

ppos_all = g.read_new(snapbase, 'POS ', 1, multiple_files=True, is_snap=True)
pids_all = g.read_new(snapbase, 'ID  ', 1, multiple_files=True, is_snap=True)
df = pd.DataFrame(data = {'x': ppos_all[:,0], 'y': ppos_all[:,1], 'z': ppos_all[:,2]}, index=pids_all)

nbins = 200 # number of bins for density profile

f = h5.File('./output/'+sim+'/density_profiles/densprofile_z'+str(round(redshift)) + '.hdf5', 'w')

for ih in index_mch[0][:3]: # loop through each MCH; get particles contained in it; compute density profile
    print('halo number:',ih)

    f_halo = f.create_group('mch_'+str(ih))

    for halo in matcha.yield_haloes(groupbase, ihalo_start=ih, ihalo_end=ih, with_ids=True, blocks=('GLEN', 'MMEA', 'RMEA', 'GPOS','MTOP','RTOP', 'MCRI', 'RCRI','NSUB')):
        halo_pids = halo['ids']
        halo_glen = halo['GLEN']
        halo_pos = halo['GPOS']
        halo_mmea = halo['MMEA']*munit/h
        halo_mtop = halo['MTOP']*munit/h
        halo_mcri = halo['MCRI']*munit/h
        halo_rmea = halo['RMEA']*lunit/h/(1+redshift)
        halo_rtop = halo['RTOP']*lunit/h/(1+redshift)
        halo_rcri = halo['RCRI']*lunit/h/(1+redshift)
        halo_nsub = halo['NSUB']

    halo_ppos = df.loc[halo_pids].to_numpy() # get particles contained in halo
    dist = np.sqrt((halo_ppos[:,0] - halo_pos[0])**2 + (halo_ppos[:,1] - halo_pos[1])**2 + (halo_ppos[:,2] - halo_pos[2])**2)
    rad, frac = find_radius(dist, halo_glen, nbins)
    r_profile, rho_profile = compute_densityprofile(dist, rad, nbins, redshift, h, partmass)
    
    nfw_params = fit_nfw(r_profile, rho_profile)
    pl_params = fit_pl(r_profile, rho_profile)

    f_halo.create_dataset('r', data=r_profile)
    f_halo.create_dataset('rho', data=rho_profile)
    f_halo.create_dataset('nfw_fit', data=nfw_params)
    f_halo.create_dataset('pl_fit', data=pl_params)
    f_halo.attrs['RMEA'] = halo_rmea
    f_halo.attrs['RTOP'] = halo_rtop
    f_halo.attrs['RCRI'] = halo_rcri
    f_halo.attrs['MMEA'] = halo_mmea
    f_halo.attrs['MTOP'] = halo_mtop
    f_halo.attrs['MCRI'] = halo_mcri
    f_halo.attrs['NSUB'] = halo_nsub
    f_halo.attrs['GLEN'] = halo_glen

f.close()
