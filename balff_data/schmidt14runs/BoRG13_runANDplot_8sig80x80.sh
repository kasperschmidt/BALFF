# === VALUES ===
Niter=50000
Nburn=10000
contamfracval=0.42
contamfracvalphistar=0.0
phistarval=-3.51 #+ 0.36 - 0.52
plotdir=plots/

# === FILENAMES ===
dataarrayfile=dataarray_BoRG13sample130920_8sig.fits
lookupfile=dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields.npz 
mcmcchainfile=mcmcchains/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p42_50kV3full.pickle
phistarfile=mcmcchains/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p42_50kV3full_PhiStarContamfrac0p0.fits
epsilonstring="'mcmcchains/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p42_50kV3full_epsilon_Mminint*.fits'"

# === RUN MCMC CODE ===
./mpd_pymc_v3_runWithRAM.py $dataarrayfile --verbose --contamfrac $contamfracval --selectionfunction 1  --Niter $Niter --Nburn $Nburn --Nthin 1 --samprange -2.0 2.0 -2.0 2.0 3.0 30.0 --klval -0.98 0.76 1e12 --chainstart -0.98 -0.119 12.0 --lookuptable $lookupfile

# === ESTIMATE PHISTAR ===
./mpd_pymc_v3_estimatePhiStar.py $mcmcchainfile $dataarrayfile $contamfracvalphistar --verbose

# === ESTIMATE EPSILON ===
for ii in -10 -11 -12 -13 -14 -15 -16 -17 -17.7 -18; do ./mpd_pymc_v3_estimateEpsilon.py $mcmcchainfile $dataarrayfile --verbose --Mminintval $ii ; done

# === PLOT QCF, HIST AND MULTID ===
./mpd_pymc_v3_plot_epsilonQCF.py $epsilonstring $phistarfile $contamfracval --verbose --kLstarchains $mcmcchainfile --createMultiD

# === PLOT LF AND ALPHA VS MSTAR ===
./mpd_pymc_v3_runWithRAM_plot.py $dataarrayfile $mcmcchainfile -2 2 -2 2 --verbose --outputdir $plotdir --lookuptable $lookupfile  --sigval 8 --bradley12 --phistarval $phistarval

# === OPEN PLOTS ===
open $plotdir/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p0_the*png $plotdir/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p0*.pdf mcmcchains/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p0*.pdf mcmcchains/dataarray_BoRG13sample130920_8sig_lookuptable_80x80tab_allfields_pymcchains_contamfrac0p0*.png 
