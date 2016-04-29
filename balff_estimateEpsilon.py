#!/usr/bin/env python2.7
#+
#----------------------------
#   NAME
#----------------------------
# balff_estimateEpsilon.py
#----------------------------
#   PURPOSE/DESCRIPTION
#----------------------------
# Estimataing the luminosity density epsilon (normalized by phi*) based on a balff_mpd run,
# i.e., the output from running balff_run.py
#----------------------------
#   INPUTS:
#----------------------------
# mcmcpickle       : pickle file containing mcmc chains created by running balff_mpd.py
# dataarray        : data array the mcmc sampler in balff_mpd.py was run with.
#----------------------------
#   OPTIONAL INPUTS:
#----------------------------
# --Mminintval     : Lower integration limit when estimating epsilon [absolute mag].
#                    Will be turned into L/[1e44erg/s] in actual integration using MUVsun=5.48
#                    Default value = 0.0718 corresponding to M = -17.7 with MUVsun=5.48
# --lookuptable    : Dictionary of look-up tables used for data array. Used when creating the balff_mpd class
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
import time
import pyfits
#-------------------------------------------------------------------------------------------------------------
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("mcmcpickle", type=str, help="File containing pickle file with MCMC chains.")
parser.add_argument("dataarray", type=str, help="Data array MCMC sampler was run with.")
# ---- optional arguments ----
parser.add_argument("--Mminintval", type=float, help="The lower integration limit (in absolue mag)")
parser.add_argument("--lookuptable", type=str, help="Dictionary of look-up tables to use.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose comments")
args = parser.parse_args()
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- START OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------
# Loading MCMC info
if args.verbose: print ' - Loading MCMC chains and stats'
mcmcdb    = pymc.database.pickle.load(args.mcmcpickle)
kvaldraw  = mcmcdb.trace('theta')[:,0]
logLstar  = mcmcdb.trace('theta')[:,1]
logNdraw  = mcmcdb.trace('theta')[:,2]
#import mpd.modifytrace as mt; kvaldraw, logLstar, logNdraw = mt.modifytrace(kvaldraw,logLstar,logNdraw)

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
LJlimmin = np.min(mpdclass.data['LFIELDLIM'])
if args.verbose: print ' - Found ',Nobj,' objects spread over ',Nfield,' fields'
#-------------------------------------------------------------------------------------------------------------
def schechter(L,alpha,Lstar,phistar):
    '''
    Calculating the Schechter function value. 
        Phi(L)dL = phi* x (L/L*) x exp(L/L*) d(L/L*)
    Units returned are:
        [phistar] = Mpc^-3
    '''
    power   = (L/Lstar)**alpha
    expon   = np.exp(-(L/Lstar))
    schval  = phistar / Lstar * power * expon
    return schval
#-------------------------------------------------------------------------------------------------------------
def integrateLFL(Lmin,Lmax,Lstar,alpha,phistar,intmethod='trapz'):
    """
    Function integrating (L x LF) over a specified Luminoisity range to
    obtain the luminoisity density.
    In Bouwens et al. (2012) the luminiisity densities in figure 20 are
    calculated by integrating down to Mabs = -17.7 which corresponds to
        butil.Mabs2L(-17.7,MUVsun=5.48) = 0.0718 * 1e44 erg/s
    """
    logLval = np.linspace(np.log10(Lmin),np.log10(Lmax),1000)
    Lval    = 10**logLval

    if intmethod == 'quad':
        intvalquad, error = scipy.integrate.quad(lambda ll: ll*schechter(ll,alpha,Lstar,phistar),Lmin,Lmax)
        if intvalquad != 0:
            intvaltrapz  = scipy.integrate.trapz(Lval*schechter(Lval,alpha,Lstar,phistar)*(Lval*np.log(10)),logLval)
            intval       = intvaltrapz
            print intval,intvalquad,intvaltrapz
        else:
            intval  = intvalquad
    elif intmethod == 'trapz':
        intvaltrapz      = scipy.integrate.trapz(Lval*schechter(Lval,alpha,Lstar,phistar)*(Lval*np.log(10)),logLval)
        intval           = intvaltrapz
    else:
        sys.exit('selected integration method ('+intmethod+') is not valid --> ABORTING')

    return intval

#-------------------------------------------------------------------------------------------------------------
# Luminoisity densities from the literature
if args.verbose: print ' - Calculate z~8 log10(epsilon) from literature M* and phi* for comparison:'
labellit   = ['Schmidt et al. (2014) 5sig','Schmidt et al. (2014) 8sig','Bradley et al. (2012)',
              'Oesch et al. (2012)','Bouwens et al. (2011)',
              'Lorenzoni et al. (2011)','Trenti et al. (2011)','McLure et al. (2010)',
              'Bouwens et al. (2010)','Schenker et al. (2013)','McLure et al. (2013)']
lphistarlit = np.array([-3.24,-3.51,-3.37,-3.17,-3.23,-3.0,-3.4,-3.46,-2.96,-3.5,-3.35]) # log \phi_{*}$ (Mpc$^{-3}$)
Mstarlit    = np.array([-20.15,-20.40,-20.26,-19.80,-20.10,-19.5,-20.2,-20.04,-19.5,-20.44,-20.12])
alphalit    = np.array([-1.87,-2.08,-1.98,-2.06,-1.91,-1.7,-2.0,-1.71,-1.74,-1.94,-2.02])
Lstarlit    = util.Mabs2L(Mstarlit,MUVsun=5.48)
Nlit        = len(alphalit)
wave        = 1600.e-10 #m
freq        = 2.99e8/wave
liteps      = np.zeros(Nlit) # literature values of epsilon
intmeth     = 'trapz' # 'quad'

for ii in xrange(Nlit):
    oneintval  = integrateLFL(0.1,1000,Lstarlit[ii],alphalit[ii],10**lphistarlit[ii],intmethod=intmeth)
    liteps[ii] = np.log10(oneintval * 1e44 / freq) # luminosity density in units erg/s/Hz/Mpc3
    if args.verbose: print '   log10(epsilon / [erg/s/Hz/Mpc3]) from ',labellit[ii],' = ',str("%.2f" % liteps[ii])
#-------------------------------------------------------------------------------------------------------------
# Integrate Nmcmc schechter functions
if args.verbose: print ' - Calculating epsilon/phi* based on the k and L* MCMC chains'
epsilon   = np.zeros(Nmcmc)
t0 = time.time()

if args.Mminintval: # setting the minimum luminoisity to integrate down to
    Lminintval = util.Mabs2L(args.Mminintval,MUVsun=5.48)
else:
    Lminintval = 0.0718 # DEFAULT - corresponds to -17.7

for mm in xrange(Nmcmc):
    twointval    = integrateLFL(Lminintval,1000,Lstardraw[mm],kvaldraw[mm]-1.0,1.0,intmethod=intmeth)
    epsilon[mm]  = np.log10(twointval * 1e44 / freq)

    if (args.verbose == 1):
        infostr = '    Added value '+str("%.8d" % (mm+1))+' ('+str("%.5f" % epsilon[mm])+\
                  ') to epsilon vector           (out of '+str("%.8d" % Nmcmc)+')'
        sys.stdout.write("%s\r" % infostr)
        sys.stdout.flush()

#-------------------------------------------------------------------------------------------------------------
# Saving epsilon values
fitsname = args.mcmcpickle.replace('.pickle','_epsilon.fits')
if args.Mminintval: fitsname = fitsname.replace('.fits','_Mminint'+str(args.Mminintval).replace('.','p')+'.fits')
if args.verbose: print '\nWriting k, L* and log10(epsilon/phi*) values to fits table :',fitsname
    
col1  = pyfits.Column(name='K' , format='D', array=kvaldraw)
col2  = pyfits.Column(name='LSTAR', format='D', array=Lstardraw)
col3  = pyfits.Column(name='EPSILONPHISTAR', format='D', array=epsilon)
cols  = pyfits.ColDefs([col1, col2, col3])
tbhdu = pyfits.new_table(cols)          # creating table header

# writing hdrkeys:   '---KEY--',                  '----------------MAX LENGTH COMMENT-------------'
tbhdu.header.append(('NMCMC   ' ,Nmcmc            ,'Number of MCMC draws'),end=True)

hdu      = pyfits.PrimaryHDU()             # creating primary (minimal) header
thdulist = pyfits.HDUList([hdu, tbhdu])    # combine primary and table header to hdulist
thdulist.writeto(fitsname,clobber=True)    # write fits file (clobber=True overwrites excisting file)
#-------------------------------------------------------------------------------------------------------------
print '\nThe mean of the obtained epsilon/phi* values is :',np.mean(epsilon)
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- END OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------



