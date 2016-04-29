#!/usr/bin/env python2.7
#+
#----------------------------
#   NAME
#----------------------------
# balff_estimate_PhiStar.py
#----------------------------
#   PURPOSE/DESCRIPTION
#----------------------------
# Estimataing phi* based on a balff_mpd run, i.e., the output from running balff_mpd.py
#
# This is done by drawing Nhighz from a binomial distribution from the Ntot=Nhighz+Ncontam distribution
# for a given assumed contamination fraction. Note that the contamination fraction is included
# in the posterior and the phi* is therefore independent of contamination fraction
#
# The distribution of Nhighz is used to calculate the distribution of phi* for all pairs of (k,L*,Nhighz)
# mcmc draws.
#----------------------------
#   INPUTS:
#----------------------------
# mcmcpickle       : pickle file containing mcmc chains created by running balff_run.py
# dataarray        : datarray the mcmc sampler in balff_run.py was run with.
#----------------------------
#   OPTIONAL INPUTS:
#----------------------------
# --lookuptable    : Dictionary of look-up tables used for data array when creating balff_mpd class
# --zrange         : Redshift range to estimate the comoving cosmological volume of the ideal
#                    full-sky survey within
# --verbose        : Toggle verbosity
# --help           : Printing help menu
#----------------------------
#   EXAMPLES/USAGE
#----------------------------
# see the shell script balff_run_commands.sh
#----------------------------
#   MODULES
#----------------------------
import balff_utilities as util
import argparse
import sys
import numpy as np
import pymc
import balff_mpd as mpd
import scipy.integrate
import scipy.stats
from cosmocalc import cosmocalc
import pyfits
#-------------------------------------------------------------------------------------------------------------
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("mcmcpickle", type=str, help="File containing pickle file with MCMC chains.")
parser.add_argument("dataarray", type=str, help="Data array MCMC sampler was run with.")
# ---- optional arguments ----
parser.add_argument("--lookuptable", type=str, help="Dictionary of look-up tables to use.")
parser.add_argument("--zrange", type=float, nargs=2, help="Redshift range to estiamte volume within [zmin,zmax]")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose comments")
args = parser.parse_args()
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- START OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------
# Literature values
labelsB12   = ['Bradley et al. (2012)','Oesch et al. (2012)','Bouwens et al. (2011b)',
               'Lorenzoni et al. (2011)','Trenti et al. (2011)','McLure et al. (2010)',
               'Bouwens et al. (2010a)','Schenker et al. (2013)','McLure et al. (2013)']
Nliterature = len(labelsB12)
linestyles  = ['b-','g-','y-','m-','b--','r--','y--','c-','r-']
lphistarB12 = np.array([-3.37,-3.17,-3.23,-3.0,-3.4,-3.46,-2.96,-3.5,-3.35]) # log \phi_{*}$ (Mpc$^{-3}$)
MstarB12    = np.array([-20.26,-19.80,-20.10,-19.5,-20.2,-20.04,-19.5,-20.44,-20.12])
mappstarB12 = util.magabs2app(MstarB12,'zdummy','RAdummy','DECdummy',Av='Avdummy',band='Jbradley2012',cos='WMAP7BAOH0')
alphaB12    = np.array([-1.98,-2.06,-1.91,-1.7,-2.0,-1.71,-1.74,-1.94,-2.02])
kB12        = alphaB12+1.0
LstarB12    = util.Mabs2L(MstarB12,MUVsun=5.48)
epsilonB12  = np.array([25.50,25.58,25.65,25.23,25.45,25.17,25.18,25.46,25.46]) #calculated + literature vals
#-------------------------------------------------------------------------------------------------------------
def schechter(L,alpha,Lstar,phistar):
    '''
    Calculating the Schechter function value.
        Phi(L)dL = phi* x (L/L*)^alpha x exp(L/L*) d(L/L*)
    Units returned are:
        [phistar] = Mpc^-3
    '''
    power   = (L/Lstar)**alpha
    expon   = np.exp(-(L/Lstar))
    schval  = phistar / Lstar * power * expon
    return schval
#-------------------------------------------------------------------------------------------------------------
def volume(zmin,zmax,cos=[0.3,70]):
    '''
    Calculating dv/dz for a given redshift range and cosmology
    '''
    dz    = zmax-zmin
    vmin  = cosmocalc(zmin,H0=cos[1],WM=cos[0])['VCM_Gpc3']*1.0e9 # volume in Mpc^3
    vmax  = cosmocalc(zmax,H0=cos[1],WM=cos[0])['VCM_Gpc3']*1.0e9 # volume in Mpc^3
    dv    = vmax-vmin
    return dv/dz,vmax
#-------------------------------------------------------------------------------------------------------------
# Calculating volume in dz
if args.zrange:
    zmin  = args.zrange[0]
else:
    zmin  = 7.5

if args.zrange:
    zmax  = args.zrange[1]
else:
    zmax  = 8.5

if args.verbose: print ' - Estimating comoving cosmological volume of the ideal full-sky survey for z = ['+\
                       str(zmin)+','+str(zmax)+']'
Dz        = zmax-zmin
dVdz,Vmax = volume(zmin,zmax)
Vol       = dVdz * Dz
#-------------------------------------------------------------------------------------------------------------
# Loading balff MCMC info
if args.verbose: print ' - Loading MCMC chains and stats outputted by balff_run.py'
mcmcdb    = pymc.database.pickle.load(args.mcmcpickle)
kvaldraw  = mcmcdb.trace('theta')[:,0]
logLstar  = mcmcdb.trace('theta')[:,1]
logNdraw  = mcmcdb.trace('theta')[:,2]

Lstardraw = 10**logLstar
Nmcmc     = len(kvaldraw)
if args.verbose: print ' - Found ',Nmcmc,' MCMC draws in chains'

#-------------------------------------------------------------------------------------------------------------
# Loading data array
if args.verbose: print ' - Loading data array'
if args.lookuptable:
    lutabval  = args.lookuptable
else:
    lutabval  = None
mpdclass = mpd.balff_mpd(args.dataarray,lookuptable=lutabval,verbose=0)
Nobj     = mpdclass.Nobj
fieldent = np.unique(mpdclass.data['FIELD'],return_index=True)
fields   = fieldent[0]
Nfield   = len(fields)
LJlim    = mpdclass.data['LFIELDLIM'][fieldent[1]]
if args.verbose: print ' - Found ',Nobj,' objects spread over ',Nfield,' fields'
#-------------------------------------------------------------------------------------------------------------
if 'simulatedsamples' in args.mcmcpickle:
    ftot = args.contamfracsim
else:
    fborg = 0.0
    fhudf = 0.0
    fcont = np.zeros(Nobj)
    for ii in xrange(Nobj):
        if mpdclass.data['FIELD'][ii][0:4] == 'borg':
            fcont[ii] = fborg
        elif mpdclass.data['FIELD'][ii][0:3] == 'UDF':
            fcont[ii] = fhudf # assuming no contamination in UDF/ERS fields
        elif mpdclass.data['FIELD'][ii][0:3] == 'ERS':
            fcont[ii] = fhudf # assuming no contamination in UDF/ERS fields
        else:
            print 'Field ',mpdclass.data['FIELD'][ii],' not recognized (assuming 0.0 contamination)'

    Nborg = len(fcont[(fcont != 0.0) & (fcont < 1.0)])
    Nhudf = len(fcont[fcont == 0.0])

    if Nhudf != (Nobj-Nborg): sys.exit('ERROR the number of HUDF and BoRG objects does not sum to Nobj --> Aborting')
    ftot = 1.0 - (np.float(Nhudf)/np.float(Nobj)*(1.0-fhudf) + np.float(Nborg)/np.float(Nobj)*(1.0-fborg))

#-------------------------------------------------------------------------------------------------------------
# Get distribution of Nhighz from Nmcmc
Nhighz = 10.0**logNdraw

#-------------------------------------------------------------------------------------------------------------
# estimate phi* distribution
phistarest = np.zeros(Nmcmc)
logLmin    = -4
logLmax    = np.inf
if args.verbose: print ' - Estimating phi* from Nhighz distribution'
for ii in xrange(Nmcmc):
    alpha       = kvaldraw[ii]-1.0
    Lstar       = Lstardraw[ii]
    intsch, err = scipy.integrate.quad(lambda L: schechter(L,alpha,Lstar,1.0),10**logLmin,np.inf)
    phistarest[ii]  = Nhighz[ii] / Vol / intsch

    infostr = '   Phistar value no. '+str("%.8d" % (ii+1))+'    (out of '+str("%.8d" % Nmcmc)+')'
    sys.stdout.write("%s\r" % infostr)
    sys.stdout.flush()

#-------------------------------------------------------------------------------------------------------------
# Estimate Ntrue from Phi* literature values
Ntrue = np.zeros(Nliterature)
for ii in xrange(Nliterature):
    alpha       = alphaB12[ii]
    Lstar       = LstarB12[ii]
    phistar     = 10**lphistarB12[ii]
    intsch, err = scipy.integrate.quad(lambda L: schechter(L,alpha,Lstar,phistar),10**logLmin,np.inf)
    Ntrue[ii]   = Vol * intsch

#-------------------------------------------------------------------------------------------------------------
# Saving phi* values
fitsname  = args.mcmcpickle.replace('.pickle','_PhiStar.fits')
if args.verbose: print '\n - Writing k, L* and phi* values to fits table:\n   '+fitsname

col1  = pyfits.Column(name='K' , format='D', array=kvaldraw)
col2  = pyfits.Column(name='LSTAR', format='D', array=Lstardraw)
col3  = pyfits.Column(name='PHISTAR', format='D', array=phistarest)
cols  = pyfits.ColDefs([col1, col2, col3])
tbhdu = pyfits.new_table(cols)          # creating table header

# writing hdrkeys:   '---KEY--',                  '----------------MAX LENGTH COMMENT-------------'
tbhdu.header.append(('NFIELD  ' ,Nfield           ,'Number of fields used in estimate'),end=True)
tbhdu.header.append(('NMCMC   ' ,Nmcmc            ,'Number of MCMC draws'),end=True)

hdu      = pyfits.PrimaryHDU()             # creating primary (minimal) header
thdulist = pyfits.HDUList([hdu, tbhdu])    # combine primary and table header to hdulist
thdulist.writeto(fitsname,clobber=True)  # write fits file (clobber=True overwrites excisting file)  
#-------------------------------------------------------------------------------------------------------------
datfitsPS = pyfits.open(fitsname)
fitstabPS = datfitsPS[1].data
psval     = np.log10(fitstabPS['PHISTAR'])
medianps  = np.median(psval)
psvalm    = np.sort(psval)[int(0.16*len(psval))]
psvalp    = np.sort(psval)[int(0.84*len(psval))]
dpsm      = medianps - psvalm
dpsp      = psvalp - medianps

if args.verbose:
    print '\n - The obtained value of phi* is (median +/- 68% conf):'
    print '   log(phi*)      = ',str("%.2f" % medianps),'+',str("%.2f" % dpsp),'-',str("%.2f" % dpsm)
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- END OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------



