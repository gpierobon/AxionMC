
nogrid    = False                    # Don't read/write grids  
readmode  = False                    # Read grids instead of calculating 
snap_base = 'size1024/L6/outhalo'    # Output simulation folder 
read_base = 'size1024/L24delta/density_1024'  
readlist  = [16,18]                  # Grids to open
atype = 0                            # 0 = projection plot
				     # 1 = volume render plot	
				     # 2 = power spectrum
				     # 3 = delta grid save in hdf5 
				     # 4 = distribution of 1+delta in $NB bins and $DIFF is selected
				     # 5 = distribution of 1+delta for void/filament centers
				     # 6 = void size function at given time $ST and threshold $THR
				     # 7 = void fraction and energy as a function of time and threshold $THR
				     # 8 = density profiles of heaviest $NH objects at time $ST
				     # 9 = density profiles of first $NH voids at time $ST and threshold/radius $THR,VRAD
				     # 10 = density variation MonteCarlo simulation 

Nfiles   = 1                         # Number of files to process
stime    = 1                         # Snapshot selected
gridsize = 512                       # Grid size
globalp  = True                      # Print global properties   

mtype    = 'pride'                   # Map type, only if OPT=0. Maps from cmasher ONLY are accepted 
camres   = 2048                      # Camera resolution in volume render

nbins    = 80                        #  Number of bins for given distribution
dflag    = False                     # If true adds distribution of gradient density 
dire     = 'N'                       # Direction for gradient distribution  


vrad     = 10                        # Void radius for density distribution (OPT=5) or profile (OPT=9)
fil_flag = False                     # Check for filaments instead of voids
vthres   = 0.7                       # Void threshold for smoothing, only for OPT=5,6
vthres2  = 0.4                       # Lower threshold for filaments
 
fof_flag = True                      # FOF data or Subhalo data for profiles 
rad_type = 'R200'                    # Radius in profile: 'R200','R500','RMean','TopHat'
r_min    = 1                         # Rmin for profile in units of the softening length
nhalos   = 100                       # Number of halos/voids to profile

