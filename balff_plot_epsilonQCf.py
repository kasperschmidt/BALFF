#!/usr/bin/env python2.7
#+
#----------------------------
#   NAME
#----------------------------
# balff_plot_epsilonQCf.py
#----------------------------
#   PURPOSE/DESCRIPTION
#----------------------------
# Creating plots for the Q, C and f values inferred from the epsilon (and phi*) estimates
# where Q is the fraction of ionized H, C is the clumpiness, and f is the ionizing photons
# escape fraction.
#
# The (hardcoded) priors from which C, f and logksi (conversion between UV luminosity density
# and ionizing photons) are drawn are
#   C           Uniform prioir between 1.0 and 6.0
#   f           Uniform prioir between 0.1 and 0.5
#   logksi      Gaussian prioirt with mean 25.2 and standard deviation 0.15
#
#----------------------------
#   INPUTS:
#----------------------------
# epsilonfile      : epsilon fits tables (use *-wildcard to specify myltiple files)
#                    from balff_estimateEpsilon.py to plot.
# phistarfile      : phistar fits table from balff_estimatePhiStar.py
# contamfrac       : contamination fraction used when sampling in balff_run.py
#----------------------------
#   OPTIONAL INPUTS:
#----------------------------
# --createMultiD   : Create the MultiD corner plot (time consuming)?
# --kLstarchains   : Name of the mcmc chains of k and Lstar to create multi-d plot of
#                    alpha, M*, epsilon and phi* for
# --eps            : saving created plots as eps files
# --png            : saving created plots as png files
# --show           : showning plots on screen for manipulation and saving
# --verbose        : Toggle verbosity
# --help           : Printing help menu
#----------------------------
#   EXAMPLES/USAGE
#----------------------------
# see the shell script balff_run_commands.sh
#----------------------------
#   MODULES
#----------------------------
import argparse
import sys
import numpy as np
import pyfits
import glob
import pymc
import balff_utilities as butil
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------------------
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("epsilonfile", type=str, help="epsilon fits table from balff_estimateEpsilon.py")
parser.add_argument("phistarfile", type=str, help="phistar fits table from balff_estimatePhiStar.py")
parser.add_argument("contamfrac", type=float, help="The contamination fraction used when sampling in balff_run.py")
# ---- optional arguments ----
parser.add_argument("--kLstarchains", type=str, help="pickle file containing the k and L star MCMC chains")
parser.add_argument("--createMultiD", action="store_true", help="Set to create multi-D corner plot")
parser.add_argument("--eps", action="store_true", help="Turn plots into eps files")
parser.add_argument("--png", action="store_true", help="Turn plots into png files")
parser.add_argument("--show", action="store_true", help="Showing plots on screen")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose comments")
args = parser.parse_args()
#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- START OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------
# load phistar data
datfitsPS    = pyfits.open(args.phistarfile)
fitstabPS    = datfitsPS[1].data
NPS = len(fitstabPS)
#-------------------------------------------------------------------------------------------------------------
# load epsilon data
args.epsilonfile = args.epsilonfile.replace("'",'')
epsfiles  = glob.glob(args.epsilonfile)
Nepsfiles = len(epsfiles)
#-------------------------------------------------------------------------------------------------------------
# extracting Mlim values from epsilonfiles
Mlim      = np.zeros(Nepsfiles)
for ii in range(Nepsfiles):
    try:
        Mlim[ii] = float(epsfiles[ii].split('Mminint')[-1].split('.')[0].replace('p','.'))
        if args.verbose: print "Performed Mminint string-split for, i.e. will include data from:\n",epsfiles[ii]
    except:
        print "Couldn't perform Mminint string-split for ",epsfiles[ii]
#-------------------------------------------------------------------------------------------------------------
# Draw values for C, f, and ksi
Nmcmc     = NPS
Cval      = np.zeros([Nepsfiles,Nmcmc]) # array to contain draws of C
fval      = np.zeros([Nepsfiles,Nmcmc]) # array to contain draws of f
logksival = np.zeros([Nepsfiles,Nmcmc]) # array to contain draws of ksi
logepsilonval= np.zeros([Nepsfiles,Nmcmc])

for ii in range(Nepsfiles):
    datfitsEPS   = pyfits.open(epsfiles[ii])
    fitstabEPS   = datfitsEPS[1].data
    NEPS         = len(fitstabEPS)
    if NPS != NEPS:
        sys.exit('The number of phi* and epsilon values in the input files are not the same --> ABORTING')
    else:
        Nmcmc = NEPS

    draw_C           = np.random.uniform(low=1.0, high=6.0, size=Nmcmc)
    draw_f           = np.random.uniform(low=0.1, high=0.5, size=Nmcmc)
    draw_logksi      = np.random.normal(loc=25.2,scale=0.15,size=Nmcmc)

    Cval[ii,:]       = draw_C
    fval[ii,:]       = draw_f
    logksival[ii,:]  = draw_logksi
    logepsilonval[ii,:] = fitstabEPS['EPSILONPHISTAR'] + np.log10(fitstabPS['PHISTAR'])
#-------------------------------------------------------------------------------------------------------------
# Literature values
labelsB12   = ['Bradley et al. (2012)','Oesch et al. (2012)','Bouwens et al. (2011b)','Lorenzoni et al. (2011)','Trenti et al. (2011)','McLure et al. (2010)','Bouwens et al. (2010a)','Schenker et al. (2013)','McLure et al. (2013)']
linestyles  = ['b-','g-','y-','m-','b--','r--','y--','c-','r-']
lphistarB12 = np.array([-3.37,-3.17,-3.23,-3.0,-3.4,-3.46,-2.96,-3.5,-3.35]) # log \phi_{*}$ (Mpc$^{-3}$)
#epsilonB12  = np.array([25.50,25.45,25.52,25.33,25.45,25.17,25.38,25.46,25.46]) #calculated
epsilonB12  = np.array([25.50,25.58,25.65,25.23,25.45,25.17,25.18,25.46,25.46]) #calculated + literature vals
#-------------------------------------------------------------------------------------------------------------
#                                       PLOTTING
#-------------------------------------------------------------------------------------------------------------
Fsize = 18
plt.rc('text', usetex=True)                         # enabling LaTex rendering of text
plt.rc('font', family='serif',size=Fsize)           # setting text font
plt.rc('xtick', labelsize=Fsize) 
plt.rc('ytick', labelsize=Fsize) 
#-------------------------------------------------------------------------------------------------------------
pname = './balff_plots/'+args.epsilonfile.split('/')[-1].split('_epsilon_')[0]+'_QCfvsMlim.pdf'
if args.eps: pname = pname.replace('.pdf','.eps')
if args.png: pname = pname.replace('.pdf','.png')
if args.verbose: print ' - Plotting QC/f vs Mlim to \n  ',pname

logQCfvec      = np.zeros(Nepsfiles)
logQCfvec_errp = np.zeros(Nepsfiles)
logQCfvec_errm = np.zeros(Nepsfiles)
for ii in xrange(Nepsfiles):
    logQCfall     = logepsilonval[ii,:]+logksival[ii,:]-50.31
    logQCfvec[ii] = np.median(logQCfall )

    errm   = np.sort(logQCfall)[int(0.16*len(logQCfall))]
    errp   = np.sort(logQCfall)[int(0.84*len(logQCfall))]
    logQCfvec_errm[ii] = logQCfvec[ii] - errm
    logQCfvec_errp[ii] = errp - logQCfvec[ii]

    if args.verbose:
        print ' - The obtained value of log(QC/f) for Mlim='+str(Mlim[ii])+' is (68% conf):'
        print '   log(QC/f)      = ',str("%.2f" % logQCfvec[ii]),'+',str("%.2f" % logQCfvec_errp[ii]),'-',str("%.2f" % logQCfvec_errm[ii])

ymax,ymin = np.max(logQCfvec+logQCfvec_errp),np.min(logQCfvec-logQCfvec_errm)
plt.clf()
width = 2

plt.fill_between(Mlim,logQCfvec+logQCfvec_errp,logQCfvec-logQCfvec_errm,alpha=0.30,color='k')
plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.30,label='68.2\% confidence') # dummy label
plt.plot(Mlim,logQCfvec,'k-',label=r'median log$_{10}$( QC/f )', lw=width)
plt.plot([-17.7,-17.7],[ymin,ymax+0.05],'-k',label='HUDF limit (Bouwens et al. 2011)')

plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(1.0*3/0.2),':k',label='Q=1, C=3, f=0.2')
plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(0.5*3/0.2),'--k',label='Q=0.5, C=3, f=0.2')
plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(0.2*3/0.2),'-.k',label='Q=0.2, C=3, f=0.2')

plt.xlabel(r'M$_{lim}$')
plt.ylabel(r'log$_{10}$( QC/f )')

plt.ylim(ymin,1.2)
plt.xlim(np.max(Mlim),np.min(Mlim))

leg = plt.legend(fancybox=True, loc='lower left',numpoints=1,prop={'size':12})
leg.get_frame().set_alpha(0.6)

plt.savefig(pname)
if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
pname = './balff_plots/'+args.epsilonfile.split('/')[-1].split('_epsilon_')[0]+'_QvsMlim.pdf'
if args.eps: pname = pname.replace('.pdf','.eps')
if args.png: pname = pname.replace('.pdf','.png')
if args.verbose: print ' - Plotting Q vs Mlim to \n  ',pname

logQvec      = np.zeros(Nepsfiles)
logQvec_errp = np.zeros(Nepsfiles)
logQvec_errm = np.zeros(Nepsfiles)
for ii in xrange(Nepsfiles):
    logQall     = logepsilonval[ii,:]+logksival[ii,:]-50.31-np.log10(Cval[ii,:])+np.log10(fval[ii,:])
    logQvec[ii] = np.median(logQall )

    errm   = np.sort(logQall)[int(0.16*len(logQall))]
    errp   = np.sort(logQall)[int(0.84*len(logQall))]
    logQvec_errm[ii] = logQvec[ii] - errm
    logQvec_errp[ii] = errp - logQvec[ii]

    if args.verbose:
        print ' - The obtained value of log(Q) for Mlim='+str(Mlim[ii])+' is (68% conf):'
        print '   log(Q)      = ',str("%.2f" % logQvec[ii]),'+',str("%.2f" % logQvec_errp[ii]),'-',str("%.2f" % logQvec_errm[ii])

ymax,ymin = np.max(logQvec+logQvec_errp),np.min(logQvec-logQvec_errm)
plt.clf()
width = 2

plt.fill_between(Mlim,logQvec+logQvec_errp,logQvec-logQvec_errm,alpha=0.30,color='k')
plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.30,label='68.2\% confidence') # dummy label

plt.plot(Mlim,logQvec,'k-',label=r'median log$_{10}$( Q )', lw=width)
plt.plot([-17.7,-17.7],[ymin,ymax],'-k',label='HUDF limit (Bouwens et al. 2011)')

plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(1.0),':k',label='Q=1')
plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(0.5),'--k',label='Q=0.5')
plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(0.2),'-.k',label='Q=0.2')

plt.xlabel(r'M$_{lim}$')
plt.ylabel(r'log$_{10}$( Q )')

plt.ylim(ymin,ymax)
plt.xlim(np.max(Mlim),np.min(Mlim))

leg = plt.legend(fancybox=True, loc='lower left',numpoints=1,prop={'size':12})
leg.get_frame().set_alpha(0.6)

plt.savefig(pname)
if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
pname = './balff_plots/'+args.epsilonfile.split('/')[-1].split('_epsilon_')[0]+'_cfvsMlim.pdf'
if args.eps: pname = pname.replace('.pdf','.eps')
if args.png: pname = pname.replace('.pdf','.png')
if args.verbose: print ' - Plotting log(C/f) vs Mlim to \n  ',pname

logcfvec      = np.zeros(Nepsfiles)
logcfvec_errp = np.zeros(Nepsfiles)
logcfvec_errm = np.zeros(Nepsfiles)
Qfixed        = 0.5

for ii in xrange(Nepsfiles):
    logcfall      = logepsilonval[ii,:]+logksival[ii,:]-50.31-np.log10(Qfixed)
    #logcfall     = np.log10(Cval[ii,:])-np.log10(fval[ii,:])
    logcfvec[ii] = np.median(logcfall )

    errm   = np.sort(logcfall)[int(0.16*len(logcfall))]
    errp   = np.sort(logcfall)[int(0.84*len(logcfall))]
    logcfvec_errm[ii] = logcfvec[ii] - errm
    logcfvec_errp[ii] = errp - logcfvec[ii]

    if args.verbose:
        print ' - The obtained value of log(Q) for Mlim='+str(Mlim[ii])+' is (68% conf):'
        print '   log(C/f)      = ',str("%.2f" % logcfvec[ii]),'+',str("%.2f" % logcfvec_errp[ii]),'-',str("%.2f" % logcfvec_errm[ii])

ymax,ymin = np.max(logcfvec+logcfvec_errp),np.min(logcfvec-logcfvec_errm)
plt.clf()
width = 2

plt.fill_between(Mlim,logcfvec+logcfvec_errp,logcfvec-logcfvec_errm,alpha=0.30,color='k')
plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.30,label='68.2\% confidence') # dummy label

plt.plot(Mlim,logcfvec,'k-',label=r'median log$_{10}$( C/f ), Q=0.5', lw=width)
plt.plot([-17.7,-17.7],[ymin,ymax],'-k',label='HUDF limit (Bouwens et al. 2011)')

plt.plot(Mlim,np.ones(Nepsfiles)*np.log10(3.0/0.2),'--k',label='Q=0.5, C=3, f=0.2')

plt.xlabel(r'M$_{lim}$')
plt.ylabel(r'log$_{10}$( C/f )')

plt.ylim(ymin,ymax)
plt.xlim(np.max(Mlim),np.min(Mlim))

leg = plt.legend(fancybox=True, loc='lower left',numpoints=1,prop={'size':12})
leg.get_frame().set_alpha(0.6)

plt.savefig(pname)
if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
if args.phistarfile:
    pnamePS = './balff_plots/'+args.phistarfile.split('/')[-1].replace('.fits','_phistarhist_QCf.pdf')
    if args.eps: pnamePS = pnamePS.replace('.pdf','.eps')
    if args.png: pnamePS = pnamePS.replace('.pdf','.png')
    if args.verbose: print ' - Plotting histogram of phi* values to \n  ',pnamePS

    psval     = np.log10(fitstabPS['PHISTAR'])
    Npsval    = len(psval)
    medianps  = np.median(psval)
    meanps    = np.mean(psval)

    psvalm   = np.sort(psval)[int(0.16*len(psval))]
    psvalp   = np.sort(psval)[int(0.84*len(psval))]
    dpsm     = medianps - psvalm
    dpsp     = psvalp - medianps

    if args.verbose:
        print ' - The obtained value of phi* is (median +/- 68% conf):'
        print '   log(phi*)      = ',str("%.2f" % medianps),'+',str("%.2f" % dpsp),'-',str("%.2f" % dpsm)

    Nbins     = 100
    histparam = np.histogram(psval,bins=Nbins)
    counts    = histparam[0]
    binedge   = histparam[1]
    histcum   = np.cumsum(counts) # cumulative histogram
    err68ent  = [np.where(histcum > Npsval*0.159)[0][0],np.where(histcum < Npsval*0.841)[0][-1]]
    err95ent  = [np.where(histcum > Npsval*0.023)[0][0],np.where(histcum < Npsval*0.977)[0][-1]]
    err99ent  = [np.where(histcum > Npsval*0.0015)[0][0],np.where(histcum < Npsval*0.9985)[0][-1]]

    err68     = [binedge[err68ent[0]],binedge[err68ent[1]+1]]
    err95     = [binedge[err95ent[0]],binedge[err95ent[1]+1]]
    err99     = [binedge[err99ent[0]],binedge[err99ent[1]+1]]

    plt.clf()
    width = 2

    plt.hist(psval,bins=Nbins,color="k",histtype="step",lw=width,normed=True)
    xmin,xmax,ymin,ymax = plt.axis()

    plt.fill_between(err68,[ymin,ymin],[ymax,ymax],alpha=0.10,color='k')
    plt.fill_between(err95,[ymin,ymin],[ymax,ymax],alpha=0.10,color='k')
    plt.fill_between(err99,[ymin,ymin],[ymax,ymax],alpha=0.10,color='k')
    plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.30,label='68.2\% confidence') # dummy label
    plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.20,label='95.4\% confidence') # dummy label
    plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.10,label='99.7\% confidence') # dummy label

    plt.plot([medianps,medianps],[ymin,ymax],'k-',label=r'median $\phi^*$ MCMC', lw=width)
    plt.plot([meanps,meanps],[ymin,ymax],'k--',label=r'mean $\phi^*$ MCMC', lw=width)

    for ii in range(len(lphistarB12)):
        plt.plot([lphistarB12[ii],lphistarB12[ii]],[ymin,ymax],linestyles[ii],label=labelsB12[ii],lw=width)

    plt.xlabel(r'log10($\phi^*$ / [Mpc$^3$])')
    plt.ylabel('PDF')

    plt.ylim(ymin,ymax)
    plt.xlim(-5.5,-2.5)

    leg = plt.legend(fancybox=True, loc='upper left',numpoints=1,prop={'size':12})
    leg.get_frame().set_alpha(0.6)

    plt.savefig(pnamePS)
    if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
for ii in xrange(Nepsfiles):
    pnameEPS = './balff_plots/'+epsfiles[ii].split('/')[-1].replace('.fits','_epsilonhist_QCf.pdf')
    if args.eps: pnameEPS = pnameEPS.replace('.pdf','.eps')
    if args.png: pnameEPS = pnameEPS.replace('.pdf','.png')
    if args.verbose: print ' - Plotting histogram of epsilon values to \n  ',pnameEPS

    epsval    = logepsilonval[ii,:]
    Nepsval   = len(epsval)
    medianeps = np.median(epsval)
    meaneps   = np.mean(epsval)

    epsvalm   = np.sort(epsval)[int(0.16*len(epsval))]
    epsvalp   = np.sort(epsval)[int(0.84*len(epsval))]
    depsm     = medianeps - epsvalm
    depsp     = epsvalp - medianeps

    if args.verbose:
        print ' - The obtained value of log(epsilon) is (68% conf):'
        print '   log(epsilon)      = ',str("%.2f" % medianeps),'+',str("%.2f" % depsp),'-',str("%.2f" % depsm)

    Nbins     = 50
    histparam = np.histogram(epsval,bins=Nbins)
    counts    = histparam[0]
    binedge   = histparam[1]
    histcum   = np.cumsum(counts) # cumulative histogram
    err68ent  = [np.where(histcum > Nepsval*0.159)[0][0],np.where(histcum < Nepsval*0.841)[0][-1]]
    err95ent  = [np.where(histcum > Nepsval*0.023)[0][0],np.where(histcum < Nepsval*0.977)[0][-1]]
    err99ent  = [np.where(histcum > Nepsval*0.0015)[0][0],np.where(histcum < Nepsval*0.9985)[0][-1]]

    err68     = [binedge[err68ent[0]],binedge[err68ent[1]+1]]
    err95     = [binedge[err95ent[0]],binedge[err95ent[1]+1]]
    err99     = [binedge[err99ent[0]],binedge[err99ent[1]+1]]

    plt.clf()
    width = 2

    histval = plt.hist(epsval,bins=Nbins,color="k",histtype="step",lw=width,normed=True)
    xmin,xmax,ymin,ymax = plt.axis()
    ymin = 0.0
    ymax = np.max(histval[0])*1.05
    plt.fill_between(err68,[ymin,ymin],[ymax,ymax],alpha=0.10,color='k')
    plt.fill_between(err95,[ymin,ymin],[ymax,ymax],alpha=0.10,color='k')
    plt.fill_between(err99,[ymin,ymin],[ymax,ymax],alpha=0.10,color='k')
    plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.30,label='68.2\% confidence') # dummy label
    plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.20,label='95.4\% confidence') # dummy label
    plt.hist(np.linspace(1000,1010,10),bins=10,color='k',alpha=0.10,label='99.7\% confidence') # dummy label

    plt.plot([medianeps,medianeps],[ymin,ymax],'k-',label=r'median $\epsilon$ MCMC', lw=width)
    plt.plot([meaneps,meaneps],[ymin,ymax],'k--',label=r'mean $\epsilon$ MCMC', lw=width)

    for jj in range(len(epsilonB12)):
        plt.plot([epsilonB12[jj],epsilonB12[jj]],[ymin,ymax],linestyles[jj],label=labelsB12[jj],lw=width)

    plt.xlabel(r'log10($\epsilon$ [erg / s / Hz / Mpc$^3$])')
    plt.ylabel('PDF')

    plt.ylim(ymin,ymax)
    plt.xlim(25.1,25.8)
    #plt.xlim(25,27)

    leg = plt.legend(fancybox=True, loc='upper left',numpoints=1,prop={'size':12})
    leg.get_frame().set_alpha(0.6)

    plt.savefig(pnameEPS)
    if args.show: plt.show()  # draw plot on screen
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#                                        MULTI-D PLOT
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
if args.kLstarchains and args.createMultiD:
    if args.verbose: print ' - Creating multi-D plot'
    MM = pymc.database.pickle.load(args.kLstarchains)
    kdrawn         = MM.trace('theta')[:,0]
    logLstardrawn  = MM.trace('theta')[:,1]
    logNdraw       = MM.trace('theta')[:,2]
    #import modifytrace as mt; kdrawn, logLstardrawn, logNdraw = mt.modifytrace(kdrawn,logLstardrawn,logNdraw)

    epsent       = -2
    datfitsEPS   = pyfits.open(epsfiles[epsent])
    fitstabEPS   = datfitsEPS[1].data
    if args.verbose:
        print '   will use epsilon data from:'
        print epsfiles[epsent]

    rasterval = False # draw points as bmp instead of vector based
    phidim    = np.log10(fitstabPS['PHISTAR'])
    epsdim    = fitstabEPS['EPSILONPHISTAR'] + phidim
    alphdim   = kdrawn-1.0
    mstardim  = butil.L2Mabs(10**logLstardrawn,MUVsun=5.48)

    import matplotlib.pyplot as plt
    multidname = './balff_plots/'+args.kLstarchains.split('/')[-1].replace('.pickle','_multiDplot.pdf')
    if args.eps: multidname = multidname.replace('.pdf','.eps')
    if args.png: multidname = multidname.replace('.pdf','.png')

    fig        = plt.figure(figsize=(13,10))
    Fsize      = 10
    plt.rc('text', usetex=True)                         # enabling LaTex rendering of text
    plt.rc('font', family='serif',size=Fsize)           # setting text font
    plt.rc('xtick', labelsize=Fsize)
    plt.rc('ytick', labelsize=Fsize)

    left  = 0.06    # the left side of the subplots of the figure
    right = 0.98    # the right side of the subplots of the figure
    bottom = 0.05   # the bottom of the subplots of the figure
    top = 0.98      # the top of the subplots of the figure
    wspace = 0.05   # the amount of width reserved for blank space between subplots
    hspace = 0.05   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    nbins      = 30
    contbins   = 200
    markersize = 2.5
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 1) # Mstar hist
    histval = mstardim
    hist    = plt.hist(histval, bins=nbins,color='k',normed=True,histtype="step")
    plt.xlim(np.min(histval),np.max(histval))
    plt.ylim(0,np.max(hist[0])*1.1)
    #plt.xlabel(r'$M^*$')
    plt.ylabel(r'PDF')
    plt.xticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #plt.subplot(4, 4 , 2) # None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #plt.subplot(4, 4 , 3) # None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #plt.subplot(4, 4 , 4) # None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 5) # eps vs M*
    xval = mstardim
    yval = epsdim
    plt.plot(xval, yval, 'ok',alpha=0.05,ms=markersize,rasterized=rasterval)
    gridsigma, extent = butil.confcontours(xval,yval,binx=contbins,biny=contbins)
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    plt.xlim(np.min(xval),np.max(xval))
    plt.ylim(np.min(yval),np.max(yval))
    #plt.xlabel(r'$M^*$')
    plt.ylabel(r'log10($\epsilon$ [erg/s/Hz/Mpc$^3$])')
    plt.xticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 6) # eps hist
    histval = epsdim
    hist    = plt.hist(histval, bins=nbins,color='k',normed=True,histtype="step")
    plt.xlim(np.min(histval),np.max(histval))
    plt.ylim(0,np.max(hist[0])*1.1)
    #plt.xlabel(r'$M^*$')
    #plt.ylabel(r'PDF')
    plt.xticks([])
    plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #plt.subplot(4, 4 , 7) # None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #plt.subplot(4, 4 , 8) # None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 9) # phi* vs M*
    xval = mstardim
    yval = phidim
    plt.plot(xval, yval, 'ok',alpha=0.05,ms=markersize,rasterized=rasterval)
    gridsigma, extent = butil.confcontours(xval,yval,binx=contbins,biny=contbins)
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    plt.xlim(np.min(xval),np.max(xval))
    plt.ylim(np.min(yval),np.max(yval))
    #plt.xlabel(r'$M^*$')
    plt.ylabel(r'log10($\phi^*$/[Mpc$^3$])')
    plt.xticks([])
    #plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 10)# phi* vs eps
    xval = epsdim
    yval = phidim
    plt.plot(xval, yval, 'ok',alpha=0.05,ms=markersize,rasterized=rasterval)
    gridsigma, extent = butil.confcontours(xval,yval,binx=contbins,biny=contbins)
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    plt.xlim(np.min(xval),np.max(xval))
    plt.ylim(np.min(yval),np.max(yval))
    #plt.xlabel(r'$M^*$')
    #plt.ylabel(r'log10($\phi^*$/[Mpc$^3$])')
    plt.xticks([])
    plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 11)# phi* hist
    histval = phidim
    hist    = plt.hist(histval, bins=nbins,color='k',normed=True,histtype="step")
    plt.xlim(np.min(histval),np.max(histval))
    plt.ylim(0,np.max(hist[0])*1.1)
    #plt.xlabel(r'$M^*$')
    #plt.ylabel(r'PDF')
    plt.xticks([])
    plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #plt.subplot(4, 4 , 12)# None
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 13)# alpha vs M*
    xval = mstardim
    yval = alphdim
    plt.plot(xval, yval, 'ok',alpha=0.05,ms=markersize,rasterized=rasterval)
    gridsigma, extent = butil.confcontours(xval,yval,binx=contbins,biny=contbins)
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    plt.xlim(np.min(xval),np.max(xval))
    plt.ylim(np.min(yval),np.max(yval))
    plt.xlabel(r'$M^*$')
    plt.ylabel(r'$\alpha$')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 14)# alpha vs eps
    xval = epsdim
    yval = alphdim
    plt.plot(xval, yval, 'ok',alpha=0.05,ms=markersize,rasterized=rasterval)
    gridsigma, extent = butil.confcontours(xval,yval,binx=contbins,biny=contbins)
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    plt.xlim(np.min(xval),np.max(xval))
    plt.ylim(np.min(yval),np.max(yval))
    plt.xlabel(r'log10($\epsilon$ [erg/s/Hz/Mpc$^3$])')
    #plt.ylabel(r'$\alpha$')
    plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 15)# alpha vs phi*
    xval = phidim
    yval = alphdim
    plt.plot(xval, yval, 'ok',alpha=0.05,ms=markersize,rasterized=rasterval)
    gridsigma, extent = butil.confcontours(xval,yval,binx=contbins,biny=contbins)
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    plt.xlim(np.min(xval),np.max(xval))
    plt.ylim(np.min(yval),np.max(yval))
    plt.xlabel(r'log10($\phi^*$/[Mpc$^3$])')
    #plt.ylabel(r'$\alpha$')
    plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.subplot(4, 4 , 16)# alpha hist
    histval = alphdim
    hist    = plt.hist(histval, bins=nbins,color='k',normed=True,histtype="step")
    plt.xlim(np.min(histval),np.max(histval))
    plt.ylim(0,np.max(hist[0])*1.1)
    plt.xlabel(r'$\alpha$')
    #plt.ylabel(r'PDF')
    plt.yticks([])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    plt.savefig(multidname, dpi=300) # dpi = dot per inch for rasterized points
    if args.verbose: print '\n - Saved ',multidname

#-------------------------------------------------------------------------------------------------------------
if args.verbose: print '\n:: '+sys.argv[0]+' :: -- END OF PROGRAM -- \n'
#-------------------------------------------------------------------------------------------------------------

