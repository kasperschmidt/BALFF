
# Bayesian Algorithm for Luminosity Function Fitting (BALFF)

README for the Bayesian Algorithm for Luminosity Function Fitting (BALFF) presented
in Schmidt et al. (2014) and extended to include magnification bias in Mason et al. (2015) 

If you find this code useful please cite

- Schmidt et al. (2014), ApJ, 786:57, http://adsabs.harvard.edu/abs/2014ApJ...786...57S
- Mason et al. (2015), ApJ, 805:79, http://adsabs.harvard.edu/abs/2015ApJ...805...79M

## Table of Content

- [Description](#description)
- [Script Overview](#script-overview)
- [Dependencies and Requirements](#dependencies-and-requirements)
  - [Standard Packages](#standard-packages)
  - [Special Packages](#special-packages)
- [Running BALFF](#running-balff)
  - [Default Run](#default-run)
  - [Accounting for Magnification Bias](#accounting-for-magnification-bias)
- [Main Keywords in BALFF](#main-keywords-in-balff)
- [References](#references)

## Description

The Bayesian Algorithm for Luminosity Function Fitting (BALFF) presented in [Schmidt et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...786...57S) is an algorithm that fits the (UV) luminosity function to a sample of galaxies (e.g., photometrically selected Lyman Break Galaxies, LBGs) selected from a set of observations.
BALFF uses the Schechter Function shown to be a good approximation for the underlying distribution at low and high redshift, as its luminosity function model.
The Bayesian formalism which BALFF is build on, avoids binning (and thus smearing) of object samples, includes a likelihood based on the formally correct binomial distribution as opposed to the often-used approximate Poisson distribution, and models the photometric uncertainties of each object in the sample directly, making full use of the full information providing more rigorous results.
In [Mason et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...805...79M) the default BALFF setup, was extended to account for the magnification bias of galaxies in the input sample from both strong and weak gravitational lensing by foreground sources.

Luminosity functions obtained with BALFF (or any other fitting method) can be used to discuss the physics of reionization as also presented in [Schmidt et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...786...57S). Scripts accompanying the core of the BALFF tool provides means for such a discussion and comparison. The fraction of neutral hydrogren at the targeted redshift is estimated assuming theoretically motivated priors on the clumping factor and the photon escape fraction.

Note that in the current BALFF framework the p(z) prior on the redshift of the individual objects (LBG candidates) is not explicitly taken into account.

For detailed information on the Bayesian framework and the formal derivation and description of the terms in the Bayesian expressions, please refer to [Schmidt et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...786...57S) and [Mason et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...805...79M).

## Script Overview

The following gives and overview of the scripts provided with the BALFF code

- `balff_run_commands.sh`
  - Shell script with default commands to run all necessary steps in a full BALFF analysis.
- `balff_run.py`
  - Wrapper setting up the `pymc` MCMC sample used to obtain the full BALFF luminosity function estimate and uncertainties.
- `balff_mpd.py`
  - The main class used to load the input, setting up the framework, and calculating the marginal posterior distribution at the heart of the BALFF luminosity function fitting framework.
- `balff_estimatePhiStar.py`
  - Estimating the luminosity function normalization (phi\*) for the luminosity function determined for the input data by BALFF.
- `balff_estimateEpsilon.py`
  - Estimating the luminosity density (epsilon) for a given integration limit, corresponding to the luminosity function determined for the input data by BALFF.
- `balff_plot_epsilonQCf.py`
  - Estimating the neutral hydrogen fraction of the IGM assuming default values (distributions) for the escape fraction, clumping factor, and conversion between luminosity density and ionizing photons. Diagnostic plots are generated.
- `balff_plot.py`
  - Generating diagnostic plots of the samples and luminosity function resulting from running BALFF on an input data array.
- `balff_createDataArray.py` 
  - Generating a data array on the fits format expected by BALFF from provided object names, field names, luminosities, and limiting luminosities.
- `balff_createDataArray_sim.py`
  - Generating a data array for a simulated sample of data. I.e., data with a known input Schechter Function distribution. Useful for testing and de-bugging.
- `balff_createLookupTable.py` 
  - Generating a look-up tables of tabulated values on the binary `.npz` format expected by BALFF. This look-up table can be used to speed up the BALFF run via interpolation of values, rather than running the full integration for each MCMC step.
- `balff_createSelectionFunctionFiles.py`
  - Storing selection and completeness functions in the `.npz` file format expected by BALFF. Can also be used to plot and estimate values of these. If the completeness C(m) is included in the selection function S(m,z), completeness function files of constant 1 can be generated for BALFF input.
- `balff_utilities.py`
  - Various utilities used in the BALFF code.

## Dependencies and Requirements

The code is written in python and uses a wide range of default packages included in standard installations of python. A few special packages as outlined below, needs to be installed on top of that to get BALFF running.

### Standard Packages

The following standard packages are imported in on or more of the BALFF scripts: 
`os`,
`sys`,
`pdb`,
`time`,
`glob`,
`numpy`,
`pylab`,
`scipy`,
`types`,
`mpmath`,
`pyfits`,
`commands`,
`argparse`,
`datetime`,
`cosmocalc`,
`matplotlib`,
`mpl_toolkits`,
`multiprocessing`,

### Special Packages

- `pymc`: The default MCMC sampler is `pymc` which needs to be installed. To run `pymc` a `FORTRAN` and `C` compiler needs to be available. E.g., `gfortran` and `gcc` both available at http://hpc.sourceforge.net. Alternatively, compilers can be installed on MacOSX via the `Xcode` app. The `pymc`  instructions can be found at https://pymc-devs.github.io/pymc/INSTALL.html. 
- `pymc_steps`: A robust adaptive metropils algorithm (RAM) for stepping in parameters space in the `pymc` chains. Available at https://github.com/brandonckelly/pymc_steps
- `astropysics`: Suite of astronomy related utilities. Available here at https://pythonhosted.org/Astropysics/
- `fast_kde`: Used for plotting `balff_plot_epsilonQCf.confcontours()`. Provided in BALFF GitHub repository.


## Running BALFF

In this section the default run of BALFF ([Schmidt et al. 2014](http://adsabs.harvard.edu/abs/2014ApJ...786...57S)) and a run of BALFF accounting for the magnification bias of sources ([Mason et al. 2015](http://adsabs.harvard.edu/abs/2015ApJ...805...79M)) are described. 

To run BALFF and generate the diagnostic plots and output files, a set of data inputs are required as described below. BALFF assumes that data are stored in the directory `./balff_data/`. The data output and plots generated by running BALFF will be stored in `./balff_output` and `./balff_plots`. Both of these directories will be created if they do not exist. 

### Default Run

The default run of BALFF corresponds to running the code described by [Schmidt et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...786...57S). This run requires the following data input (the names used below only refer to the default set of files provide in the BALFF GitHub directory):

- `fields_info.txt`: This files contain a minimal set of information needed for BALFF. The columns of the file are:
  - `fieldname`: String contain the unique name of the field
  - `filter`: Filter which the field was observed in (corresponding to the fileter used to assemble the luminosity function)
  - `maglim5sigma`: 5sigma limiting magnitude (including Galactic extinction) of the field
  - `Av`: A_V value for the field
  - `fieldarea`: Field area in arc minutes
  - `magmedfield`: Median observed magnitude error over field
- `objects_info.fits`: This binary fits table contain information on the individual objects that a luminosity functions should be fitted to. In the case of the BoRG sample analyzed by [Schmidt et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...786...57S)) and [Mason et al. 2015](http://adsabs.harvard.edu/abs/2015ApJ...805...79M) these are photometrically selected LBGs. The fits format expected by BALFF can be generated using `balff_createDataArray.py`. The columns in the binary table are:
  - `OBJNAME`: String containing the unique name of the object
  - `FIELD`: Name of field (from `fields_info.txt`) the object is located in
  - `LOBJ`: The luminosity of the object in units of []1e-44 erg/s]. I.e., this includes the redshift information when converting the apparent magnitude to absolute magnitude and then to luminosity (These conversion can be done using the tools in `balff_utilities.py`)
  - `LOBJERR`: The error on the object's luminosity
  - `LFIELDLIM`: The Nsigma limiting luminosity of the field the object was found in (corresponds to maglimNsigma from `fields_info.txt`)
  - `LFIELDLIMERR`: The error on the limiting luminosity of the field- `selectionfunctions/Szm*SN*.npz`: Files containing selection function for each individual field. The selection function is assumed to be dependent on both the apparent magnitude and the redshift, i.e., S(z,m). By default S(z,m) does not include the completeness function. This is provided with `C*SN*.npz`. However, S(z,m) does include the completeness function of constant 1 can be provided. Known selection functions can be converted into the `.npz` format expected by BALFF with `balff_createSelectionFunctionFiles`.- `selectionfunctions/C*SN*.npz`: Files containing the completeness function for each individual field. The completeness function is assume to only depend on the apparent magnitude, i.e., C(m). If the completeness measure is include in the selection function S(z,m) a constant function C(m) = 1 can be provided.- `objects_info_lookuptable_*.npz`[optional]:  The `*.npz` look-up table corresponds to the data sample in `objects_info.fits` and provides a gridded surface of the time consuming intergrals in the MCMC estimate of the luminosity function described by `balff_mpd.py`. When providing a look-up table to the BALFF scripts, the integral values are obtained by interpolating on this surface. This speeds up the estimate of the individual integrals in each MCMC step. The agreement of the actual integral and the interpolation on the look-up table surface, is obviously strongly dependent on the resolution of the grid the look-up table is calculated on. A look-up table can be generated with `balff_createLookupTable.py`.- `BoRG13_z8_5and8sigmaLF.txt`[optional]: Tabulated version of the BoRG13 5sigma and 8sigma luminosity functions presented by [Schmidt et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...786...57S). These are only used for plotting purposed in `balff_plot.py`.

The actual set of commands to run are summarized/exemplified in the shell script `balff_run_commands.sh` which can be used to execute a full BALFF run including plotting with simply doing `./balff_run_commands.sh` on the `bash` command line. 

A boiled-down minimal version of the `bash` commands to run BALFF and generate diagnostic plots are:

```
./balff_run.py $dataarrayfile --errdist normal --lookuptable $lookupfile --verbose

./balff_estimatePhiStar.py $mcmcchainfile $dataarrayfile --verbose

./balff_estimateEpsilon.py $mcmcchainfile $dataarrayfile --Mminintval -17.7 --verbose

./balff_plot_epsilonQCF.py $epsilonstring $phistarfile $contamfracval --kLstarchains $mcmcchainfile --verbose

./balff_plot.py $dataarrayfile $mcmcchainfile -2 2 -2 2 --verbose
```

### Accounting for Magnification Bias

Running BALFF accounting for the weak and strong lensing magnification bias of the luminosity function as described by [Mason et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...805...79M) is done by simply using `--errdist magbias` instead of the default `--errdist normal` when executing the `./balff_run.py` command.
The `magbias` run of BALFF requires the same data input as the default run described above. Additionally, the `magbias` run requires that the following file is available in `./balff_data/`:

- `fields_magbiaspdf.txt`: This file describes the probability distribution of the magnifications p(mu) for each of the fields listed in `fields_info.txt`. The p(mu)s are assumed parametrized as a sum of 1, 2, or 3 Gaussians. The file contains the following columns:
  - `Field`: Name of the field the p(mu) corresponds to
  - `mean`: Field mean magnification
  - `p1`: Weight of first Gaussian component of p(mu). Note that `p1`+`p2`+`p3`=1
  - `mean1`: Mean magnification of first Gaussian component of p(mu)
  - `std1`: Standard deviation of first Gaussian component of p(mu)
  - `p2`: Weight of second Gaussian component of p(mu). Note that `p1`+`p2`+`p3`=1
  - `mean2`: Mean magnification of second Gaussian component of p(mu)
  - `std2`: Standard deviation of second Gaussian component of p(mu)
  - `p3`: Weight of third Gaussian component of p(mu). Note that `p1`+`p2`+`p3`=1
  - `mean3`: Mean magnification of third Gaussian component of p(mu)
  - `std3`: Standard deviation of third Gaussian component of p(mu)

The individual p(mu) for `fields_magbiaspdf.txt` can be generated by the lens model of your choice. See [Mason et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...805...79M) for an example of estimating strong and intermediate lensing p(mu)s from the photometry of foreground sources. Weak lensing p(mu)s can be estimated by the [Pangloss code](https://github.com/drphilmarshall/Pangloss), though weak lensing makes a negligible contribution to magnification bias at z<~8.

If a field experiences no lensing, its p(mu) in `fields_magbiaspdf.txt` can be set to:

```
fieldname   1.0   1.0 1.0 0.0   0.0 1.0 0.0   0.0 1.0 0.0
```
which correspond to a single delta function at 1.0.

If a source is likely strongly lensed, the field hosting that source must be split in two: 1) a ~few arc second cut-out area around the source which has the strong lensing p(mu) parameters and 2) the remainder of the field, which experiences no strong lensing and therefore will have field mean magnification of 1.0. The new areas of these sub-fields should be included in `field_info.txt`. These areas are the observed areas and should not be demagnified by mu, as that is accounted for in the posterior distribution estimate in `balff_mpd.py`.

As and example, if a field (fieldX) includes a strongly lensed source (objX1), the field has to be split in two by cutting out a region around objX1 which has the strong lensing p(mu), and `field_info.txt`, `objects_info.fits`, and `fields_magbiaspdf.txt` should be modified from 

```
field_info.txt:  
fieldX filter maglim5sigma Av fieldareaX magmedfieldX

objects_info.fits:  
objX1   fieldX  LOBJX1 LOBJERRX1 LFIELDLIM LFIELDLIMERR
objX2   fieldX  LOBJX2 LOBJERRX2 LFIELDLIM LFIELDLIMERR
objX3   fieldX  LOBJX3 LOBJERRX3 LFIELDLIM LFIELDLIMERR

fields_magbiaspdf.txt:  
fieldX   meanX     pX1 meanX1 stdX1   pX2 meanX2 stdX2
```
to something like

```
field_info.txt:  
fieldX_nostrong filter maglim5sigma Av fieldareaX-fieldareaX_strong magmedfieldX_nostrong
fieldX_strong   filter maglim5sigma Av fieldareaX_strong            magmedfieldX_strong

objects_info.fits:  
objX1   fieldX_strong    LOBJX1 LOBJERRX1 LFIELDLIM LFIELDLIMERR
objX2   fieldX_nostrong  LOBJX2 LOBJERRX2 LFIELDLIM LFIELDLIMERR
objX3   fieldX_nostrong  LOBJX3 LOBJERRX3 LFIELDLIM LFIELDLIMERR

fields_magbiaspdf.txt:  
fieldX_nostrong meanX_nostrong pX_nostrong1 meanX_nostrong1 stdX_nostrong1 pX_nostrong2 meanX_nostrong2 stdX_nostrong2
fieldX_strong   meanX_strong   pX_strong1   meanX_strong1   stdX_strong1   pX_strong2   meanX_strong2   stdX_strong2
```
Here `fieldareaX_strong` is the area of the cut-out around objX1. And then meanX_nostrong is always 1 as this field is experiencing no strong lensing.

For more details please refer to [Mason et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...805...79M).

## Main Keywords in BALFF 

Below a few of the main keywords available in `balff_run.py` and `balff_mpd.py` are described. For more details on the individual routines and scripts, please refer to the headers of these and their subroutines.

### balff_mpd.py

```
datafitstable     : Binary Fits table containing the data used to calculate the mpd
                    The class will turn this into a numpy array with dimensions N x M where N is
                    the number of objects in the selection (one for each row) and M is the number
                    of parameters. The following M parameters (columns) are expected in the fits
                    table (at least):
                        OBJNAME       Name of object
                        FIELD         Field the object was found in
                        LOBJ          Luminosity of object [10^44 erg/s]
                        LOBJERR       Uncertainty on lumintoisiy LOBJ
                        LFIELDLIM     X sigma limiting luminosity of FIELD [10^44 erg/s]
                        LFIELDLIMERR  Uncertainty on luminosity limit LFIELDLIM
                    Furthermore the fits table header should contain the keyword
                        LLIMSIFG      Number of sigmas (X) the limiting luminosity LFIELDLIM corresponds to
                    The fits table can be created with either balff_createDataArray.py or
                    balff_createDataArray_sim.py
lookuptable         The name of the lookup table to use. Default is None, i.e., all calculations are
                    done while running, which is very time-consuming.
                    A look-up table can be created with balff_createLookupTable.py
selectionfunction : The selection function to use in calculations. Choices are:
                        0 = Step function (balff_mpd.stepfct) - DEFAULT
                        1 = Tabulated (real) selection functions (balff_mpd.tabselfct)
                    (ignored if lookup table is provided as selection function is only
                    used when calculating intSelintFtilde).
errdist           : The error distribution to use in model. Chose between
                    'normal'       A normal (gauss) distribution of errors (DEFAULT)
                    'lognormal'    A log normal distribution of errors
                    'normalmag'    A log normal in L turned into normal in mag
                    'magbias'      Use gauss error distribution taking magnification bias into account
                                   where p(mu) is a sum of up to 3 gaussian distributions
LFredshift        : The redshift at which the luminosity function is determined. Used for distance
                    modulus when turning 5-sigma limiting magnitude into absolute luminosity and
                    selection function integrations.
```

### balff_run.py

```
--contamfrac     : Contamination fraction to apply to sample in datafile. Default value is 0.42
--Niter          : Number of MCMC iterations.             DEFAULT = 10000
--Nburn          : Number of MCMC burn-in iterations.     DEFAULT = 1000
--Nthin          : Factor to thin MCMC output with.       DEFAULT = 10
--step           : Stepping in MCMC chain to perform.
                   Chose between:
                        Robust Adaptive Metropolis Hastings (DEFAULT)
                        'metropolis'
                   To add more see pymc.StepMethodRegistry
--samprange      : Setting allowed values to sample from as
                   kmin kmax log10Lstarmin log10Lstarmax logNmin logNmax
                   The default ranges are:
                         kmin           = -4.0
                         kmax           =  4.0
                         logLstarmin    = -3.0
                         logLstarmax    =  3.0
                         logNmin        =  0.0
                         logNmax        = 50.0
--errdist        : The error distribution to use in balff_mpd model. Chose between
                      'normal'         A normal (gauss) distribution of errors (DEFAULT)
                      'lognormal'      A log normal distribution of errors
                      'normalmag'      A log normal in L turned into normal in mag
                      'magbias'        Use error distribution taking magnification bias into account
                                       where p(mu) is a sum of up to 3 gaussian distributions which
                                       should be provided for each field in ./balff_data/fields_magbiaspdf.txt
```


## References 

- Schmidt et al. (2014), ApJ, 786:57, http://adsabs.harvard.edu/abs/2014ApJ...786...57S
- Mason et al. (2015), ApJ, 805:79, http://adsabs.harvard.edu/abs/2015ApJ...805...79M
- pymc_steps: https://github.com/brandonckelly/pymc_steps
- Pangloss: https://github.com/drphilmarshall/Pangloss

