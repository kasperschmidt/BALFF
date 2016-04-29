#!/usr/bin/env python2.7
#+
#----------------------------
#   NAME
#----------------------------
# balff_createDataArray_sim.py
#----------------------------
#   PURPOSE/DESCRIPTION
#----------------------------
# Creating a simulated data array needed as input when creating a balff_mpd class object.
#
#----------------------------
#   INPUTS:
#----------------------------
# probdist         : Probability distribution to draw data from. Choices are: 
#                      'gamma'     (NOTE: there is no cutoff on this distribution, i.e. integrals converge)
#                      'schechter'
#                      'normal'
# Ntotal           : Number of objects to create data for. For probdist=Schechter sample goes down to
#                    Lmin = 1e-4 x 1e44 erg/s (corresponding to M=-10)
# Llim             : Luminoisity limit to apply to objects resulting from detection
#                    probability cut (0.1e44 erg/s -> mAB ~ 29.1)
# kval             : Shape parameter k of distribution (k>0) to simulate data for
#                       k = alpha(schecheter) + 1
# Lstar            : Scale parameter Lstar of distribution (Lstar>0) units [1e44 erg/s] to simulate data for
#----------------------------
#   OPTIONAL INPUTS:
#----------------------------
# --nochoice       : set this keyword to ignore the choice to write file to disk (i.e. no interaction required)
# --verbose        : Toggle verbosity
# --help           : Printing help menu
#----------------------------
#   EXAMPLES/USAGE
#----------------------------
# samples with full depth
# bash> ./balff_createDataArray_sim.py 'schechter' 10000 0.01 -0.9 0.5 --verbose
#
#----------------------------
#   MODULES
#----------------------------
import argparse
import sys
import pdb
import balff_createDataArray as bcd
import numpy as np
#-------------------------------------------------------------------------------------------------------------
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("probdist", type=str, help="Probability distribution to draw data from")
parser.add_argument("Ntotal", type=int, help="Total number of objects to draw (from Schechter down to L 1e-4 x 1e44 erg/s)")
parser.add_argument("Llim", type=float, help="luminosity limit to select objects down to.")
parser.add_argument("kval", type=float, help="Shape parameter k of distribution (k = alpha+1 > -1)")
parser.add_argument("Lstar", type=float, help="Scale parameters Lstar of distribution (Lstar > 0) units [1e44 erg/s]")
# ---- optional arguments ----
parser.add_argument("--nochoice", action="store_true", help="Set to ignore choice of saving output to disk")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose comments")
args = parser.parse_args()
#-------------------------------------------------------------------------------------------------------------
def simulate_schechter_distribution(alpha, L_star, L_min, N):
    """
    Generate N samples from a Schechter distribution, which is like a gamma distribution
    but with a negative alpha parameter and a cut off above zero so that it converges.

    Based on algorithm in http://www.math.leidenuniv.nl/~gill/teaching/astro/stanSchechter.pdf

    KBS:-------------------------------------------------------------------------------------
          Code taken from https://gist.github.com/joezuntz/5056136 and modified.
          Schechter distribution with -1 < alpha+1 (k) < -0
        -------------------------------------------------------------------------------------
    """
    output = []
    n = 0

    while n<N:
        Lgam = np.random.gamma(scale=L_star, shape=alpha+2, size=N)  # drawing values from gamma dist with k+1
        Lcut = Lgam[Lgam>L_min]                                      # removing L values from gamma dist > L_min
        ucut = np.random.uniform(size=Lcut.size)                     # random values [0:1]
        Lval = Lcut[ucut<L_min/Lcut]                                 # only keeping L values where ucut < L_min/L
        output.append(Lval)                                          # append these to output array
        n+=Lval.size                                                 # increase counter

    values = np.concatenate(output)[:N]                              # generate output by reformatting
    return values
#-------------------------------------------------------------------------------------------------------------
if args.verbose:
    print ' '
    print ':: '+sys.argv[0]+' :: -- START OF PROGRAM -- '
    print ' '
#-------------------------------------------------------------------------------------------------------------
# Drawing full sample of objects
if args.verbose: print ' - Start drawing full sample from distribution'
if args.probdist == 'gamma': 
    Ldraw_total    = np.random.gamma(args.kval,args.Lstar,args.Ntotal)
elif args.probdist == 'schechter': 
    L_min           = 1.0e-4 # Lower limit on L to enforce convergence;
    Ldraw_total    = simulate_schechter_distribution(args.kval-1.0,args.Lstar,L_min,args.Ntotal)
elif args.probdist == 'normal': 
    Ldraw_total    = np.random.normal(args.Lstar,args.kval,args.Ntotal)
else:
    sys.exit(args.probdist+' is not a valid probability distribution to draw from --> ABORTING')
#-------------------------------------------------------------------------------------------------------------
# add measurement errors
if args.verbose: print ' - Adding measurement errors to drawn sample'
Msun            = [5.61,5.48,4.83,4.42,4.08,3.64,3.32,3.28] # (rest-frame) abs Mag_sun in U,B,V,R,I,J,H,K (www.ucolick.org/~cnaw/sun.html)
Lsun            = 3.839e-11 # 1e44 erg/
measurementerr  = 0.2 # approximate 1sigma measurement error in J
Jmagerr         = np.random.normal(0.0,measurementerr,args.Ntotal) # errors to add
Lerr           = Ldraw_total*np.log(10)/2.5*Jmagerr # Error in luminosity: see notes from LT130408
Ldraw_toterr   = Ldraw_total+Lerr # adding measurement error
Lerrabs        = np.abs(Lerr)
#-------------------------------------------------------------------------------------------------------------
# Select objects above detection limit
if args.verbose: print ' - Applying detection threshold to sample'
MJlim      = -2.5*np.log10(args.Llim/Lsun)+Msun[1]+ 47.14 # apparent mag (47.14 term incl k-correction & reddening)
Llimcheck = 10**( (Msun[1]+47.14-MJlim)/2.5 ) * Lsun
Llimerr   = args.Llim * np.log(10)/2.5*0.2 # uncertainty on detection limit
Ldraw     = Ldraw_toterr[Ldraw_toterr > args.Llim]
Lerrabs   = Lerrabs[Ldraw_toterr > args.Llim]
Nobj       = len(Ldraw) # Counting number of objects above detection limit
#-------------------------------------------------------------------------------------------------------------
if args.nochoice:
    ncstring = '_nochoice'
else: # ask if the results should be saved
    input = raw_input(" - Selected a final sample of Nobj = "+str(Nobj)+
                      " \n   Should I write that sample to disk? (y/n): ")
    if (input == 'y') or (input == 'yes'):
        print "   Okay, good, then I'll continue\n"
    elif (input == 'n') or (input == 'no'):
        sys.exit("   Not satisfied?... okay then I'll abort\n")
    else:
        sys.exit('   "'+input+'" is not a valid answer --> Aborting \n')
#-------------------------------------------------------------------------------------------------------------
# define string with output name
outbase = './balff_data/dataarraySim_pdist'+args.probdist+\
          '_Ntot'+str(args.Ntotal)+\
          '_k'+str(args.kval).replace('.','p')+\
          '_Lstar'+str(args.Lstar).replace('.','p')+\
          '_Llim'+str(args.Llim).replace('.','p')+\
          '_Nobj'+str(Nobj)
if args.nochoice: outbase = outbase+ncstring
outputfile = outbase+'.fits'
#-------------------------------------------------------------------------------------------------------------
plotsamples = 1
if args.nochoice: plotsamples = 0 # ignore plotting when nochoice is set
if plotsamples == 1:
    plotname = outbase+'.pdf'
    import pylab as plt
    Fsize = 10
    plt.rc('text', usetex=True)                      # enabling LaTex rendering of text
    plt.rc('font', family='serif',size=Fsize)           # setting text font
    plt.rc('xtick', labelsize=Fsize)
    plt.rc('ytick', labelsize=Fsize)
    hist = plt.hist(np.log10(Ldraw_total),bins=100,color='0.3',
                    label='Total sample drawn from dist: k='+str(args.kval)+', L*/[1e44erg/s]='+str(args.Lstar))
    plt.hist(np.log10(Ldraw_toterr),bins=hist[1],color='b',alpha=0.5,
             label='Total sample added measurement errors')
    plt.hist(np.log10(Ldraw),bins=hist[1],color='r',alpha=0.5,
             label='Observed sample down to Llim='+str(args.Llim)+' (Mlim$\sim$'+str("%.2f" % MJlim)+')')
    plt.xlabel('log(L/[1e44erg/s])')
    plt.ylabel('\#')
    leg = plt.legend(fancybox=True, loc='upper right',numpoints=1)
    leg.get_frame().set_alpha(0.6)
    if args.verbose: print ' - Writing sample plot to ',plotname
    plt.savefig(plotname)
#-------------------------------------------------------------------------------------------------------------
# Defining/creating data lists
if args.verbose: print ' - Creating dictionary with simulated data '
Nfields   = 1
fieldsize = Nobj/(Nfields+0.0)
number    = 0 # resetting numbering

OBJname   = []
field     = []

for ii in range(Nobj):
    if ii/fieldsize == round(ii/fieldsize):
        number = number+1
        if args.verbose: print ' - Writing data for field number ',number
    OBJname.append('Obj'+format(ii+1, "05d")+'_FIELD'+str(number))
    field.append('FIELD'+str(number))

L        = Ldraw
dL       = Lerrabs
Llim     = [args.Llim for N in range(Nobj)]
dLlim    = [Llimerr for N in range(Nobj)]
#-------------------------------------------------------------------------------------------------------------
# writing output to fits table
bcd.write_fitsfile(OBJname,field,L,dL,Llim,dLlim,1,outputname=outputfile,verbose=True)
#-------------------------------------------------------------------------------------------------------------
if args.verbose:
    print ' '
    print ':: '+sys.argv[0]+' :: -- END OF PROGRAM -- '
    print ' '
#-------------------------------------------------------------------------------------------------------------



