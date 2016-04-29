#
# List of basic (default) commands to run the BALFF, calculate luminosity density (epsilon), 
# luminosity function normalization (phi*) and generate some diagnostic plots. The latter steps
# are somewhat specific to the z~8 case presented in Schmidt et al. (2014) and provide script
# ‘script templates' for further analysis.
#
# === MCMC VALUES ===
Niter=100 # 50000
Nburn=10  # 10000
contamfracval=0.42

# === INPUT FILES ===
dataarrayfile=./balff_data/objects_info.fits
lookupfile=./balff_data/objects_info_lookuptable_3x3tab_parallelrun_allfields.npz
basename=./balff_output/objects_info_lookuptable_3x3tab_parallelrun_allfields_pymcchains_contamfrac0p42

mcmcchainfile=${basename}.pickle
phistarfile=${basename}_PhiStar.fits
epsilonstring="'${basename}_epsilon_Mminint*.fits'"

# === RUN MCMC CODE ===
./balff_run.py $dataarrayfile --verbose --contamfrac $contamfracval --selectionfct 1  --Niter $Niter --Nburn $Nburn --Nthin 1 --samprange -2.0 2.0 -2.0 2.0 3.0 30.0 --chainstart -0.98 -0.119 12.0 --errdist normal --lookuptable $lookupfile --LFredshift 7.8

# === ESTIMATE PHISTAR ===
./balff_estimatePhiStar.py $mcmcchainfile $dataarrayfile --zrange 7.5 8.5 --verbose

# === ESTIMATE EPSILON ===
for ii in -10 -11 -12 -13 -14 -15 -16 -17 -17.7 -18; do ./balff_estimateEpsilon.py $mcmcchainfile $dataarrayfile --verbose --Mminintval $ii ; done

# === PLOT QCF, HIST AND MULTID ===
./balff_plot_epsilonQCF.py $epsilonstring $phistarfile $contamfracval --verbose --kLstarchains $mcmcchainfile --createMultiD

# === PLOT LF AND ALPHA VS MSTAR ===
./balff_plot.py $dataarrayfile $mcmcchainfile -2 2 -2 2 --verbose --sigval 5 --bradley12 --lookuptable $lookupfile
