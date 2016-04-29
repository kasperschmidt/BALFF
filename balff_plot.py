#!/usr/bin/env python2.7
#+
#----------------------------
#   NAME
#----------------------------
# balff_plot.py
#----------------------------
#   PURPOSE/DESCRIPTION
#----------------------------
# Diagnostic plotting of output from balff_run.py
#----------------------------
#   INPUTS:
#----------------------------
# datafile         : Fits array containing data
# picklefile       : File containing pickled output from balff_run.py run
# samprange        : Sampling region used when creating data: kmin kmax log10Lstarmin log10Lstarmax
#----------------------------
#   OPTIONAL INPUTS:
#----------------------------
# --lookuptable    : name and path to dictionary containing look-up tables for the fields in 
#                    datafile to use.
# --outputdir      : directory to put plots in. DEFAULT is './balff_plots/'
# --klval          : Coordinates of true/expected k and L* values to show in plots
# --phistarval     : log10(phi*) value if known
# --bradley12      : Set to plot data bins from Bradley et al 2012 on figures of luminosity functions
# --sigval         : Provide the sigma cut on the values in data array. Either 5 or 8 (needed for calculating
#                    volume of binned data)
# --eps            : saving created plots as eps files
# --png            : saving created plots as png files
# --show           : showning plots on screen for manipulation and saving
# --verbose        : set -verbose to get info/messages printed to the screen
# --help           : Printing help menu
#----------------------------
#   EXAMPLES/USAGE
#----------------------------
# see the shell script balff_run_commands.sh
#----------------------------
#   MODULES
#----------------------------
import balff_mpd as mpd
import argparse
import sys
import numpy as np
import pymc
import balff_utilities as butil
import pylab as plt
import pdb
import scipy
from fast_kde import fast_kde
#-------------------------------------------------------------------------------------------------------------
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("datafile", type=str, help="Fits table containing data")
parser.add_argument("picklefile", type=str, help="Pickled data base from balff_run.py")
parser.add_argument("samprange", type=float, nargs=4, help="Samplin ranges used: kmin kmax log10Lstarmin log10Lstarmax")
# ---- optional arguments ----
parser.add_argument("--lookuptable", type=str, help="Dictionary of look-up tables to use.")
parser.add_argument("--outputdir", type=str, help="Output directory to put plots in (DEFAULT is pwd)")
parser.add_argument("--klval", type=float, nargs=2, help="True/expected k and L* value to display on plots")
parser.add_argument("--phistarval", type=float, help="Provide log10(phi*) value if known")
parser.add_argument("--bradley12", action="store_true", help="Set to plot data bins from Bradley et al 2012")
parser.add_argument("--sigval", type=int, help="Sigma cut on sample in data array (5 or 8)")
parser.add_argument("--eps", action="store_true", help="Turn plots into eps files")
parser.add_argument("--png", action="store_true", help="Turn plots into png files")
parser.add_argument("--show", action="store_true", help="Showing plots on screen")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose comments")
args = parser.parse_args()
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- START OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------
# creaing balff_mpd class from data
if args.lookuptable:
    lutabval  = args.lookuptable
else:
    lutabval  = None

mpdclass       = mpd.balff_mpd(args.datafile,lookuptable=lutabval,verbose=0)
data           = mpdclass.data['LOBJ']
Nobj           = mpdclass.Nobj
borgent        = np.where( np.asarray([mpdclass.data['FIELD'][ii][0:4] for ii in range(Nobj)]) == 'borg')
udfent         = np.where( np.asarray([mpdclass.data['FIELD'][ii][0:3] for ii in range(Nobj)]) == 'UDF')
ersent         = np.where( np.asarray([mpdclass.data['FIELD'][ii][0:3] for ii in range(Nobj)]) == 'ERS')
udfersent      = np.append(udfent,ersent)
#-------------------------------------------------------------------------------------------------------------
# Loading mcmc resuts
MM = pymc.database.pickle.load(args.picklefile)
if args.verbose: print ' - Loaded chains from ',args.picklefile
kdrawn         = MM.trace('theta')[:,0] 
logLstardrawn  = MM.trace('theta')[:,1]
logNdraw       = MM.trace('theta')[:,2]

Ndraw          = len(kdrawn)

kmin           = args.samprange[0]
kmax           = args.samprange[1]
logLstarmin    = args.samprange[2]
logLstarmax    = args.samprange[3]
Lstarmin       = 10**(logLstarmax)
Lstarmax       = 10**(logLstarmin)

picklebase     = args.picklefile.split('/')[-1].split('.')[0]
if args.outputdir:
    outdir = args.outputdir
else:
    outdir = './balff_plots/'

#-------------------------------------------------------------------------------------------------------------
# load phistar table if requested
if not args.phistarval:
    if args.verbose: print '\n - NB: Setting log10(phi*) to default value of -3.0  ' \
                           '(otherwise provide it with "--phistarval") \n'
    l10phistar = -3.0 # DEFAULT value
else:
    l10phistar = args.phistarval
phistarMCMC = 10.0**l10phistar
#-------------------------------------------------------------------------------------------------------------
# The binned BoRG LF from Bradley et al. 2012 fig 8 (consistent with tab 7)
BB12_5s_mabs = np.asarray([-21.64307225, -21.14307225, -20.64307225, -20.14307225, -19.64307225])
BB12_5s_logphi = 10**np.asarray([-5.62930191, -4.70267729, -4.21874741, -3.887395, -3.34640162])

BB12_5s_logphi_lo = BB12_5s_logphi-10**(np.asarray([-5.62930191, -4.70267729, -4.21874741, -3.887395, -3.34640162]) -
                         np.asarray([0.7447686, 0.19660763, 0.1369125, 0.16896212, 0.33762804]))
BB12_5s_logphi_hi = 10**(np.asarray([-5.62930191, -4.70267729, -4.21874741, -3.887395, -3.34640162]) +
                         np.asarray([0.51256577, 0.18597924, 0.13303766, 0.1631669, 0.29432524]))-BB12_5s_logphi

#The upper limits is:
BB12_5s_mabs_ul = -22.1430722487
BB12_5s_logphi_ul = 10**-5.83366857823

#And for the BoRG 8-sigma stepwise LF:
BB12_8s_mabs = np.asarray([-21.64307225, -21.14307225, -20.64307225])
BB12_8s_logphi = 10**np.asarray([-5.5178413, -4.61047953, -4.47185492])
BB12_8s_logphi_lo = BB12_8s_logphi-10**(np.asarray([0.74353667, 0.24339243, 0.44896383]) -
                                                  np.asarray([-5.5178413, -4.61047953, -4.47185492]))
BB12_8s_logphi_hi = 10**(np.asarray([0.51404641, 0.22368689, 0.36249946]) +
                                   np.asarray([-5.5178413, -4.61047953, -4.47185492]))-BB12_8s_logphi

BB12_deltamag = 0.25
#-------------------------------------------------------------------------------------------------------------
# UDF bins from tab 3 of Bouwens et al. 2011
BO11_mabs = np.array([-20.74,-20.14,-19.54,-18.94,-18.34,-17.74])
BO11_phi = np.array([0.00011,0.00025,0.00039,0.00103,0.00156,0.00452])
BO11_err_lo = np.array([0.00007,0.00012,0.00017,0.00035,0.00072,0.00207])
BO11_err_hi = np.array([0.00007,0.00012,0.00017,0.00035,0.00072,0.00207])

BO11_deltamag = 0.3
#-------------------------------------------------------------------------------------------------------------
# Create binned data
# UDF/ERS from michele
UDFbinedges = np.array([-25.04, -21.04,-20.44, -19.84, -19.24, -18.64, -18.04, -17.44])
UDFbincen   = np.array([-23.04, -20.74, -20.14, -19.54, -18.94, -18.34, -17.74])
UDFbincenL  = butil.Mabs2L(UDFbincen,MUVsun=5.48)
UDFNobj     = np.array([0, 5, 7, 10, 18, 10, 9])
UDFvolumes  = np.array([100000.0, 75757.6, 46666.7, 42735.0, 29126.2, 10683.8, 3318.58])

UDFmagbin   = 0.6
UDFval      = UDFNobj/UDFvolumes/UDFmagbin  # #Nobj/Mpc3/mag

# UDF calc
UDFMabs     = butil.L2Mabs(mpdclass.data['LOBJ'][udfersent],MUVsun=5.48)
UDFmag      = butil.magabs2app(UDFMabs,8.0,'dummy','dummy',Av=-99,band='Jbradley2012',cos='WMAP7BAOH0')
#UDFmag      = mpdclass.data['JAUTO'][udfersent]
UDFlum      = data[udfersent]
UDFMabs     = UDFmag - 47.04 #butil.L2Mabs(UDFlum,MUVsun=5.48)

UDFNobjcalc = np.zeros(7)
for ii in xrange(7):
    UDFNobjcalc[ii] = len(np.where((UDFMabs >= UDFbinedges[ii]) & (UDFMabs < UDFbinedges[ii+1]))[0])
UDFvalcalc  = UDFNobjcalc/UDFvolumes/UDFmagbin  # #Nobj/Mpc3/mag

# BoRG from michele
BoRGbinedges = np.array([-25.95, -21.95,-21.45, -20.95, -20.45, -19.95])
BoRGbincen   = np.array([-23.95, -21.70, -21.20, -20.70, -20.20])
BoRGbincenL  = butil.Mabs2L(BoRGbincen,MUVsun=5.48)
BoRGNobj     = np.array([0, 2, 7, 13, 11])
BoRGvolumes  = np.array([523001.0, 492971.0, 409328.0, 249372.0, 80199.0])
BoRGmagbin   = 0.5
BoRGval      = BoRGNobj/BoRGvolumes/BoRGmagbin  # #Nobj/Mpc3/mag

# BoRG calc
BoRGlum      = data[borgent]
BoRGMabs     = butil.L2Mabs(BoRGlum,MUVsun=5.48)
BoRGNobjcalc = np.zeros(5)
for ii in xrange(5):
    BoRGNobjcalc[ii] = len(np.where((BoRGMabs > BoRGbinedges[ii]) & (BoRGMabs < BoRGbinedges[ii+1]))[0])
BoRGNfaint   = len(np.where(BoRGMabs > BoRGbinedges[-1])[0])

if args.bradley12:
    BoRGvolumesCalc = BoRGvolumes
else:
    print ' - Data volume estimates not properly enabled (use --bradley12 keyword) --> ABORTING'
    pdb.set_trace()

    if args.verbose: print ' - Calculating volumes for binned data'
    sigmacut = args.sigval
    voldict  = np.load('balff_estimateVolumes_BoRG13volumes.npz') # volumes calc w. balff_estimateVolumes.py
    Mabsvol  = butil.L2Mabs(voldict['Lval'],MUVsun=5.48)
    volfct   = voldict['total_'+str(sigmacut)+'sig']

    Nmagbin = len(BoRGbincen)
    BoRGvolumesCalc = np.zeros(Nmagbin)
    for vv in xrange(Nmagbin):
        ent                 = np.where((Mabsvol > BoRGbinedges[ii]) & (Mabsvol < BoRGbinedges[ii+1]))[0]
        BoRGvolumesCalc[vv] = np.abs(scipy.integrate.trapz(volfct[ent],Mabsvol[ent])) # np.abs 'cause int over mag
        print Mabsvol[ent]

    pdb.set_trace()
        
BoRGvalcalc  = BoRGNobjcalc/BoRGvolumesCalc/BoRGmagbin # Nobj/Mpc3/mag
#-------------------------------------------------------------------------------------------------------------
#                                       PLOTTING
#-------------------------------------------------------------------------------------------------------------
Fsize = 18
plt.rc('text', usetex=True)                      # enabling LaTex rendering of text
plt.rc('font', family='serif',size=Fsize)           # setting text font
plt.rc('xtick', labelsize=Fsize) 
plt.rc('ytick', labelsize=Fsize) 
#-------------------------------------------------------------------------------------------------------------

# creating plot of k-logL* plane with samples
pname2 = outdir+picklebase+'_pymcsampling.pdf'
if args.eps: pname2 = pname2.replace('.pdf','.eps')
if args.png: pname2 = pname2.replace('.pdf','.png')
if args.verbose: print ' - Plotting k-logL* plan with samples to',pname2
plt.clf()

kvaldraw      = np.median(kdrawn)
logLstardraw  = np.median(logLstardrawn)

plt.plot(kdrawn,logLstardrawn,'r.',label='Sampled: k='+str("%.2f" % kvaldraw)+' and log10(L*)='+str("%.2f" % logLstardraw),alpha=0.05,markeredgecolor='r')
if args.klval: plt.plot(args.klval[0],np.log10(args.klval[1]),'ko',zorder=5,label='True: k='+str("%.2f" % args.klval[0])+' and log10(L*)='+str("%.2f" % np.log10(args.klval[1])))
plt.xlabel('k')
plt.ylabel('log10(L*)')
plt.plot([kmin,kmin],[logLstarmin,logLstarmax],'r--',label='Sample region')
plt.plot([kmax,kmax],[logLstarmin,logLstarmax],'r--')
plt.plot([kmin,kmax],[logLstarmin,logLstarmin],'r--')
plt.plot([kmin,kmax],[logLstarmax,logLstarmax],'r--')
plt.xlim(kmin-0.2,kmax+0.2)
plt.ylim(logLstarmin-0.2,logLstarmax+0.2)

leg = plt.legend(fancybox=True, loc='upper right',numpoints=1)
leg.get_frame().set_alpha(0.6)

plt.savefig(pname2)
if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
# creating plot of alpha-M* plane with samples
pname3 = outdir+picklebase+'_alphaMstar.pdf'
if args.eps: pname3 = pname3.replace('.pdf','.eps')
if args.png: pname3 = pname3.replace('.pdf','.png')
if args.verbose: print ' - Plotting alpha-M* plan with samples to',pname3
plt.clf()

alphadraw   = kdrawn-1.0
alphaval    = np.median(alphadraw)

alphavalm   = np.sort(alphadraw)[int(0.16*len(alphadraw))]
alphavalp   = np.sort(alphadraw)[int(0.84*len(alphadraw))]
dalpham     = alphaval - alphavalm
dalphap     = alphavalp - alphaval

Mstardraw   = butil.L2Mabs(10**logLstardrawn,MUVsun=5.48) # Converting L_J into Mabs
Mstarval    = np.median(Mstardraw)

Mstarvalm   = np.sort(Mstardraw)[int(0.16*len(Mstardraw))]
Mstarvalp   = np.sort(Mstardraw)[int(0.84*len(Mstardraw))]
dMstarm     = Mstarval - Mstarvalm
dMstarp     = Mstarvalp - Mstarval

if args.verbose:
    print ' - The obtained values of M* and alpha are (68% conf):'
    print '   M*    = ',str("%.2f" % Mstarval),'+',str("%.2f" % dMstarp),'-',str("%.2f" % dMstarm)
    print '   alpha = ',str("%.2f" % alphaval),'+',str("%.2f" % dalphap),'-',str("%.2f" % dalpham)

amin        = np.min(alphadraw)
amax        = np.max(alphadraw)
Mmin        = np.min(Mstardraw)
Mmax        = np.max(Mstardraw)

plt.plot(Mstardraw,alphadraw,'k.',alpha=0.05,markeredgecolor='k')
plt.plot(10.,10.,'k.',label=r'Posterior draws' ,markeredgecolor='k')

binx = 200
biny = 200
Nval = binx*biny

sig1 = 1-0.68
sig2 = 1-0.95
sig3 = 1-0.99
kde_grid = fast_kde(alphadraw,Mstardraw, gridsize=(binx,biny), weights=None,extents=[amin,amax,Mmin,Mmax])

binarea   = (amax-amin)/binx * (Mmax-Mmin)/biny
kde_int  = kde_grid * binarea # ~integrated value in grid

kde_flat  = np.ravel(kde_int)
sortindex = np.argsort(kde_int,axis=None)[::-1]

gridsigma = np.zeros((binx,biny))

sum = 0.0
for ss in xrange(Nval):
    xx  = np.where(kde_int == kde_flat[sortindex[ss]])
    sum = sum + np.sum(kde_int[xx])
    if (sum < 0.68): gridsigma[xx] = 1.0
    if (sum > 0.68) and (sum < 0.95): gridsigma[xx] = 2.0
    if (sum > 0.95) and (sum < 0.99): gridsigma[xx] = 3.0

plt.contour(gridsigma.transpose(),[1,2,3],extent=[Mmax,Mmin,amin,amax],origin='lower',colors=['r','r','r'],label='contours',zorder=5)
plt.plot(Mstarval,alphaval,'r.',zorder=5,markeredgecolor='r',label=r'[median($M^*$),median($\alpha$)]')
plt.plot([10,11],[10,11],'r-',label='68\% and 95\% confidence',zorder=5)

if args.klval:
    alphain = args.klval[0]-1.0
    Mstarin = butil.L2Mabs(args.klval[1],MUVsun=5.48)
    plt.plot(Mstarin,alphain,'co',zorder=5,label=r'Ref.: $\alpha$='+str("%.2f" % alphain)+' and $M^*$='+str("%.2f" % Mstarin))
    plt.plot(Mstarin,alphain,'co',zorder=5,label=r'Bradley et al. (2012) best-fit')

plt.xlabel(r'$M^*$')
plt.ylabel(r'$\alpha$')

plt.xlim(-23.0,-19.0)
plt.ylim(-2.7,-1.0)

# - - - - - - - - -
plt.annotate('HUDF/ERS only',(-22.0,-1.5),rotation=0,verticalalignment='center',horizontalalignment='center',fontsize=20)

B12range = [-21.5,-19.0,-2.5,-1.0]
#plt.plot([B12range[0],B12range[0]],[B12range[2],B12range[3]],'k:',label='Bradley et al. (2012) window')
#plt.plot([B12range[1],B12range[1]],[B12range[2],B12range[3]],'k:')
#plt.plot([B12range[0],B12range[1]],[B12range[2],B12range[2]],'k:')
#plt.plot([B12range[0],B12range[1]],[B12range[3],B12range[3]],'k:')

leg = plt.legend(fancybox=True, loc='upper left',numpoints=1,prop={'size':12})
leg.get_frame().set_alpha(0.6)

plt.savefig(pname3)
if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
# creating plot of fitted model and data
pname = outdir+picklebase+'_pymcRAMfit.pdf'
if args.eps: pname = pname.replace('.pdf','.eps')
if args.png: pname = pname.replace('.pdf','.png')
if args.verbose: print ' - Plotting MCMC Schechter(<k>,<L*>) and input data to',pname
plt.clf()
Llim     = mpdclass.data['LFIELDLIM'][0] # assuming Llim is the same for all fields
phistar  = 1.0
xval     = np.linspace(Lstarmin,Lstarmax,Nobj)
#fit     = mpdclass.schechter(xval,kvaldraw,logLstardraw,phistar,nonorm=1)

Nvals    = 1000
if kvaldraw > -1: # checking that kvaldraw is inside allowed range for simulate_schechter_distribution
    schval  = butil.simulate_schechter_distribution(kvaldraw-1, 10**logLstardraw, Llim, Nvals, trunmax = 30)
    if len(schval) < Nvals:
        if args.verbose: print '      LOOPFAULT: went into long loop, broke out with only ',len(schval),'drawn values'
    if (schval == 0).all() :
        if args.verbose: print '      LOOPFAULT: went into infinite loop, broke out with  array of zeros'
else:
    schval  = np.linspace(0,10,Nvals)

# --- performing KS test of the two samples ---
from scipy.stats import ks_2samp
ksresult = ks_2samp(schval,data)
# If the K-S statistic (ksresult[0]) is small or the p-value (ksresult[1]) is high,
# we cannot reject the hypothesis that the distributions of the two samples are the same.
# ---------------------------------------------

maxx    = np.max(np.append(schval,data))
width   = 0.1
binval  = np.arange(0,maxx+width,width)
plt.hist(schval,bins=binval,normed=True,color='b',alpha=0.5,label='MCMC Schecter dist.: k='+str("%.2f" % kvaldraw)+' L*='+str("%.2f" % 10**logLstardraw)+' Llim='+str("%.2f" % Llim))
plt.hist(data,bins=binval,normed=True,color='r',alpha=0.5,label='Input data (KS-test: D='+str("%.4f" % ksresult[0])+' p='+str("%.4f" % ksresult[1])+')')
#plt.plot(xval,fit,'g-',label='pymc RAM fit: k='+str(kvaldraw)+' L*='+str(10**logLstardraw))
plt.xlabel('L / [1e44 erg/s]')
plt.ylim(0,)
plt.xlim(0,10)
plt.legend(fancybox=True, loc='upper right')
plt.savefig(pname)

#-------------------------------------------------------------------------------------------------------------
# Plotting resulting schecter function together with literature Schechter functions
pnameLF = outdir+picklebase+'_LFcomparison.pdf'
if args.eps: pnameLF = pnameLF.replace('.pdf','.eps')
if args.png: pnameLF = pnameLF.replace('.pdf','.png')
if args.verbose: print ' - Plotting the MCMC Schechter function with literature Schechters to',pnameLF
plt.clf()
# Data from table 9 in Bradley et al. 2012
labelsB12   = ['Bradley et al. (2012)','Oesch et al. (2012)','Bouwens et al. (2011b)','Lorenzoni et al. (2011)','Trenti et al. (2011)','McLure et al. (2010)','Bouwens et al. (2010a)','Schenker et al. (2013)','McLure et al. (2013)']
linestyles  = ['b-','g-','y-','m-','b--','r--','y--','c-','r-']
lphistarB12 = np.array([-3.37,-3.17,-3.23,-3.0,-3.4,-3.46,-2.96,-3.5,-3.35]) # log \phi_{*}$ (Mpc$^{-3}$)
MstarB12    = np.array([-20.26,-19.80,-20.10,-19.5,-20.2,-20.04,-19.5,-20.44,-20.12])
mappstarB12 = butil.magabs2app(MstarB12,'zdummy','RAdummy','DECdummy',Av='Avdummy',band='Jbradley2012',cos='WMAP7BAOH0')
alphaB12    = np.array([-1.98,-2.06,-1.91,-1.7,-2.0,-1.71,-1.74,-1.94,-2.02])
kB12        = alphaB12+1.0
LstarB12    = butil.Mabs2L(MstarB12,MUVsun=5.48)
#epsilonB12  = np.array([25.50,25.45,25.52,25.33,25.45,25.17,25.38,25.46,25.46]) #calculated
epsilonB12  = np.array([25.50,25.58,25.65,25.23,25.45,25.17,25.18,25.46,25.46]) #calculated + literature vals


Lvalues     = np.linspace(0.03,10,1000)
Mvalues     = butil.L2Mabs(Lvalues,MUVsun=5.48)

lnfact      = np.log(10)/2.5

for ii in range(len(kB12)):
    schval = mpdclass.schechter(Lvalues,kB12[ii],LstarB12[ii],1.0,nonorm=1)
    plt.plot(Lvalues,lnfact*Lvalues*10**lphistarB12[ii]*schval/LstarB12[ii],linestyles[ii],label=labelsB12[ii])
    #plt.plot(Lvalues,10**lphistarB12[ii]/LstarB12[ii]*(Lvalues/LstarB12[ii])**alphaB12[ii]*np.exp(-Lvalues/LstarB12[ii]),linestyles[ii],label=labelsB12[ii])

# MCMC result:
schval      = mpdclass.schechter(Lvalues,kvaldraw,10**logLstardraw,1.0,nonorm=1)
plt.plot(Lvalues,lnfact*Lvalues*phistarMCMC/(10**logLstardraw)*schval,'k-',lw=3,label='MCMC result (log10(phi*)='+str("%.2f" % l10phistar)+')')

phistarav = np.mean(10**lphistarB12)
Lstarav   = np.mean(LstarB12)
schval = mpdclass.schechter(Lvalues,0.0,Lstarav,'dummy',nonorm=1)
#plt.plot(Lvalues,lnfact*Lvalues*phistarav*schval/Lstarav,'k-.',label='k=0.0, L*='+str("%.2f" % Lstarav)+', phi*='+str("%.2e" % phistarav))

schval = mpdclass.schechter(Lvalues,-1.0,Lstarav,'dummy',nonorm=1)
#plt.plot(Lvalues,lnfact*Lvalues*phistarav*schval/Lstarav,'k:',label='k=-1.0, L*='+str("%.2f" % Lstarav)+', phi*='+str("%.2e" % phistarav))

schval = mpdclass.schechter(Lvalues,-2.0,Lstarav,'dummy',nonorm=1)
#plt.plot(Lvalues,lnfact*Lvalues*phistarav*schval/Lstarav,'k--',label='k=-2.0, L*='+str("%.2f" % Lstarav)+', phi*='+str("%.2e" % phistarav))

# UDF/ERS and BoRG data
if args.bradley12:
    plt.plot(UDFbincenL,UDFval,'ro',label='UDF/ERS bins')
    plt.plot(BoRGbincenL,BoRGval,'r^',label='BoRG bins')
else:
    plt.plot(UDFbincenL,UDFvalcalc,'ko',label='UDF/ERS binned data')
    plt.plot(BoRGbincenL,BoRGvalcalc,'k^',label='BoRG binned data')

plt.xlabel('L / [1e44 erg/s])')
plt.ylabel(r'Schechter LF')
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.0376,9.46) # Mabs -17,-23
plt.ylim(1e-7,0.01)

leg = plt.legend(fancybox=True, loc='lower left',numpoints=1,prop={'size':10})
leg.get_frame().set_alpha(0.6)
plt.savefig(pnameLF)

#-------------------------------------------------------------------------------------------------------------
# Plotting resulting schecter function together with literature Schechter functions
pnameLFm = outdir+picklebase+'_LFcomparisonMabs.pdf'
if args.eps: pnameLFm = pnameLFm.replace('.pdf','.eps')
if args.png: pnameLFm = pnameLFm.replace('.pdf','.png')
if args.verbose: print ' - Plotting the MCMC Schechter function (Mabs) with literature Schechters to',pnameLFm
plt.clf()

# Indicate Data Range
UDFfaint  = butil.magapp2abs(26.2,'zdummy','RAdummy','DECdummy',Av='Avdummy',band='Jbradley2012',cos='WMAP7BAOH0')
UDFbright = butil.magapp2abs(29.6,'zdummy','RAdummy','DECdummy',Av='Avdummy',band='Jbradley2012',cos='WMAP7BAOH0')

plt.plot([UDFfaint,UDFbright],[7e-3,7e-3],'k:',lw=3)
plt.annotate('UDF/ERS Data',(UDFfaint-0.2,7e-3),rotation=0,verticalalignment='center',horizontalalignment='left',fontsize=12)

# MCMC result:
schval = mpdclass.schechterMabs(Mvalues,phistarMCMC,Mstarval,kvaldraw)
plt.plot(Mvalues,schval,'k-',lw=3,label='Input BALFF data')
#plt.plot(Mvalues,schval,'k-',lw=3,label='LF from posterior sampling')

schvalpp = mpdclass.schechterMabs(Mvalues,phistarMCMC,Mstarvalp,alphavalp+1.0)
schvalmm = mpdclass.schechterMabs(Mvalues,phistarMCMC,Mstarvalm,alphavalm+1.0)
schvalpm = mpdclass.schechterMabs(Mvalues,phistarMCMC,Mstarvalp,alphavalm+1.0)
schvalmp = mpdclass.schechterMabs(Mvalues,phistarMCMC,Mstarvalm,alphavalp+1.0)

schvalarr = np.array([schvalpp,schvalmm,schvalpm,schvalmp])
schvalmin = [np.min(schvalarr[:,ii]) for ii in xrange(len(Mvalues))]
schvalmax = [np.max(schvalarr[:,ii]) for ii in xrange(len(Mvalues))]

plt.fill_between(Mvalues,schvalmin,schvalmax,alpha=0.20,color='k',zorder=-1)

for ii in range(len(kB12)):
    schval = mpdclass.schechterMabs(Mvalues,10**lphistarB12[ii],MstarB12[ii],kB12[ii])
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# z~6 LF from Bouwens et al. 2014
plot_z6LF = False
if plot_z6LF:
    LFat6 = mpdclass.schechterMabs(Mvalues,0.56e-3,-20.86,-0.81)
    plt.plot(Mvalues,LFat6,'g--',label='$z\sim6$ Bouwens et al. (2014)',lw=2)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# z~8 BoRG13 LF from Schmidt et al. 2014
plot_z8LF = True
if plot_z8LF:
    datfile = './balff_data/BoRG13_z8_5and8sigmaLF.txt'
    B13dat  = np.genfromtxt(datfile,comments='#',dtype=None)
    BoRG13_Mval  = B13dat[:,0]
    BoRG13_LF5s  = B13dat[:,1]
    BoRG13_LF5sp = B13dat[:,2]
    BoRG13_LF5sm = B13dat[:,3]
    BoRG13_LF8s  = B13dat[:,4]
    BoRG13_LF8sp = B13dat[:,5]
    BoRG13_LF8sm = B13dat[:,6]

    plt.plot(BoRG13_Mval,BoRG13_LF5s,'b--',alpha=0.6,lw=2,
             label='BoRG13 5$\sigma$ $z\sim8$ LF \nSchmidt et al. (2014)')
    plt.fill_between(BoRG13_Mval,BoRG13_LF5sm,BoRG13_LF5sp,alpha=0.20,color='b',zorder=-1)

    plt.plot(BoRG13_Mval,BoRG13_LF8s,'g:',alpha=0.6,lw=2,
             label='BoRG13 8$\sigma$ $z\sim8$ LF \nSchmidt et al. (2014)')
    plt.fill_between(BoRG13_Mval,BoRG13_LF8sm,BoRG13_LF8sp,alpha=0.20,color='g',zorder=-1)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Mstarav   = np.mean(MstarB12)
schval = mpdclass.schechterMabs(Mvalues,phistarav,Mstarav,0.0)
#plt.plot(Mvalues,schval,'k-.',label='k=0.0, M*='+str("%.2f" % Mstarav)+', phi*='+str("%.2e" % phistarav))

schval = mpdclass.schechterMabs(Mvalues,phistarav,Mstarav,-1.0)
#plt.plot(Mvalues,schval,'k:',label='k=-1.0, M*='+str("%.2f" % Mstarav)+', phi*='+str("%.2e" % phistarav))

schval = mpdclass.schechterMabs(Mvalues,phistarav,Mstarav,-2.0)
#plt.plot(Mvalues,schval,'k--',label='k=-2.0, M*='+str("%.2f" % Mstarav)+', phi*='+str("%.2e" % phistarav))

# UDF/ERS and BoRG data
if args.bradley12:
    plt.errorbar(BO11_mabs,BO11_phi,
                 yerr=[BO11_err_lo,BO11_err_hi],
                 xerr=BO11_deltamag,fmt='ro',zorder=5,markersize=10,label=r'Bouwens et al. (2011)')

    plt.errorbar(BB12_5s_mabs,BB12_5s_logphi,
                 yerr=[BB12_5s_logphi_lo,BB12_5s_logphi_hi],
                 xerr=BB12_deltamag,fmt='b^',zorder=5,markersize=10,label=r'Bradley et al. (2012) $>5\sigma$')
    plt.errorbar(BB12_5s_mabs_ul,BB12_5s_logphi_ul,xerr=BB12_deltamag,fmt='bv',zorder=5,markersize=10,uplims=True)

    #plt.errorbar(BB12_8s_mabs,BB12_8s_logphi,
    #             yerr=[BB12_8s_logphi_lo,BB12_8s_logphi_hi],
    #             xerr=BB12_deltamag,fmt='gs',zorder=5,markersize=10,label=r'Bradley et al. (2012) $>8\sigma$')
#else:
#    plt.plot(UDFbincen,UDFvalcalc,'ko',label='UDF/ERS binned data')
#    plt.plot(BoRGbincen,BoRGvalcalc,'k^',label='BoRG binned data')

# - - - - - - - - - 
#plt.annotate('HUDF/ERS only',(-21.5,2e-3),rotation=0,verticalalignment='center',horizontalalignment='center',fontsize=18)
plt.annotate('HUDF/ERS + BoRG13 '+str(args.sigval)+'$\sigma$',(-21.5,1e-3),rotation=0,verticalalignment='center',horizontalalignment='center',fontsize=18)
#plt.annotate('Bradley et al. (2012) data',(-21.5,3e-3),rotation=0,verticalalignment='center',horizontalalignment='center',fontsize=18)

plt.xlabel('Mabs UV')
plt.ylabel(r'Schechter LF')
plt.yscale('log')
plt.xlim(-17,-23)
plt.ylim(1e-7,0.01)

leg = plt.legend(fancybox=True, loc='lower left',numpoints=1,prop={'size':12})
leg.get_frame().set_alpha(0.6)

plt.savefig(pnameLFm)
#-------------------------------------------------------------------------------------------------------------
# summarizing output files.
if args.verbose: 
    print '\n - Created the plots:'
    print pname
    print pnameLF
    print pnameLFm
    print pname2
    print pname3
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- END OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------



