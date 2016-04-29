# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import balff_mpd as mpd
import sys
import numpy as np
import pdb
from time import localtime, strftime
import multiprocessing
import datetime
from matplotlib import cm
import matplotlib.pyplot as plt
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def createlookuptable(datafile,allfields=True,kval=[-3.0,6.0,10], Lstarval=[1.0e-6,1.0e3,10],
                      dummyval=False,selfctval=0,loadselfct=False, sigmasample='5', errdist='normal',
                      verbose=True,parallelize=False):
    """
    Creating a lookuptable for balff_mpd.py

    --- INPUT ---
    datafile       Fits datafile containing data which the look-up table should correspond to.
                   Uses this binary fits table as input to the mpd class and to determine the fields
                   to create look-up tables for.
    allfields      create a lookuptable including all fields. Otherwie, only look-up tables for fields
                   including LF objects are gerenrated.
    kval           3 values giving mininum k, maximum k and number of steps, i.e., the k-dimension
                   to calculate look-up values for.
                   The DEFAULT is:
                        min  k =  -3
                        max  k =  6
                        Nstepk =  10
    Lstarval       3 values giving mininum Lstar, maximum Lstar and (log)spacing, i.e., the
                   Lstar-dimension to calculate look-up values for. Given in units of 1e44 erg/s
                   The DEFAULT is:
                         min  Lstar = 1.0e-6
                         max  Lstar = 1.0e+3
                         NstepLstar = 10     (i.e. step size of 0.5 dex)
    dummyval       Set this keyword to fill look-up tables with dummy values for testing. I.e. skips
                   performing the actual integral and is therefore faster.
    selfctval      The selection function to use in calculations. Choices are:
                        0 = Step function (balff_mpd.stepfct) - DEFAULT
                        1 = Tabulated (real) selection functions (balff_mpd.tabselfct)
                            In this case the files will be looked for in './balff_data/selectionfunctions/'
    loadselfct     Set this keyword to load the selection functions when initializing the balff_mpd class
                   instead of every time balff_mpd.selfctFIELD() is called in the balff_mpd class.
    sigmasample    The sigma (S/N) cut on sample. Used to find selection and completeness function files
    errdist        The error distribution to use in balff_mpd model. Chose between
                       'normal'               A normal (gauss) distribution of errors (DEFAULT)
                       'lognormal'            A log normal distribution of errors
                       'normalmag'            A log normal in L turned into normal in mag
                       'magbias'              Use error distribution taking magnification bias into account
                                              where p(mu) is a sum of up to 3 gaussian distributions
    verbose        Toggle verbosity
    parallelize    Prallelize the integrations using the multiprocessing package? True or False

    --- EXAMPLE OF USE ---

    import balff_createLookupTable as clt

    ======== MASTER LOOKUP 1604XX ===========

    datafile = 'dataarray_BoRG13sample130924_5sig.fits'
    sigsam   = '5'
    kval     = [-2.0,2.0,80.0]
    Lstarval = [1.0e-2,1.0e2,80.0]

    outfile = clt.createlookuptable(datafile,allfields=True,kval=kval, Lstarval=Lstarval,dummyval=False,
                                selfctval=1,loadselfct=True, sigmasample=sigsam, errdist='magbias',
                                verbose=True,parallelize=True)

    datafile = 'dataarray_BoRG13sample130920_8sig.fits'
    sigsam   = '8'
    kval     = [-2.0,2.0,80.0]
    Lstarval = [1.0e-2,1.0e2,80]

    outfile = clt.createlookuptable(datafile,allfields=True,kval=kval, Lstarval=Lstarval,dummyval=False,
                                selfctval=1,loadselfct=True, sigmasample=sigsam, errdist='magbias',
                                verbose=True,parallelize=True)


    ======== SMALL TABLE - TEST ===========

    datafile = './balff_data/objects_info.fits'
    sigsam   = '5'
    kval     = [-2.0,2.0,3.0]
    Lstarval = [1.0e-2,1.0e2,3.0]

    outfile = clt.createlookuptable(datafile,allfields=True,kval=kval, Lstarval=Lstarval,dummyval=False,selfctval=1,loadselfct=True, sigmasample=sigsam, errdist='normal',verbose=True,parallelize=False)


    ======== TO LOAD FROM RESULTING DICTIONARY DO: ========
    dict = np.load('balff_createLookupTable_OUTPUT.npz')
    kdim, Lstardim   = dict['kdim'], dict['Lstardim']
    lim, err, lookup = [dict['field1'+arg] for arg in ['lim','err','lookup']]

    """
    if verbose:
        print ' --- SET-UP: ---'
        print ' - datafile     : ',datafile
        print ' - kval         : ',kval[0:2]
        print ' - L*val        : ',Lstarval[0:2]
        print ' - errdist      : ',errdist
        print ' - table size   : ',kval[2],'x',Lstarval[2]
        if parallelize:
            print ' - parallelize  :  True\n'
        else:
            print ' - parallelize  :  False\n'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Getting field information
    # NB! intSelintFtilde which is in the look-up tables is independent of the
    # contamination fraction hence contamfrac = 0.0 has no effect on the lookup table
    if verbose: print ' - Initializing mpdclass '
    mpdclass  = mpd.balff_mpd(datafile,verbose=1,loadselfct=loadselfct,
                                errdist=errdist,sigsamp=sigmasample, contamfrac=0.0)

    fields    = mpdclass.data['FIELD'] # fields in data table (i.e. fields with highz candidates)

    if allfields: # Create lookup tables for all fields
        fieldname = mpdclass.finfonames.tolist()
        fieldlim  = mpdclass.finfoLJlim.tolist()
        fielderr  = mpdclass.finfodLJlim.tolist()
    else: # only creating lookup table for fields with high z candidates in them
          # should not be used except for simulated data... (KBS 131025)
        fieldname = np.unique(np.sort(fields))
        fieldlim  = []
        fielderr  = []
        for ii in xrange(len(fieldname)):
            fieldlim.append(mpdclass.data['LFIELDLIM'][np.where(fields == fieldname[ii])[0][0]])
            fielderr.append(mpdclass.data['LFIELDLIMERR'][np.where(fields == fieldname[ii])[0][0]])

    Nfields   = len(fieldname)
    if verbose: print ' - Found '+str(Nfields)+' fields to create look-up tables for'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Creating vectors of dimensions of look-up table
    Nk           = int(kval[2])
    kvalall      = np.linspace(kval[0],kval[1],Nk)
    NLstar       = int(Lstarval[2])
    Lstaralldex  = np.linspace(np.log10(Lstarval[0]),np.log10(Lstarval[1]),NLstar) # making sure to use log spacing
    Lstarall     = np.power(10,Lstaralldex)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Creating dictionary to contain look-up tables and adding limits and errors to it
    dict             = {}
    dict['kdim']     = kvalall
    dict['Lstardim'] = Lstarall

    for ii in xrange(Nfields):
        arg    = fieldname[ii]+'lim'
        dict[arg] = fieldlim[ii]
        arg    = fieldname[ii]+'err'
        dict[arg] = fielderr[ii]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Creating vectors of dimensions of look-up table
    Nentries  = Nk*NLstar

    if verbose: print '\n   --- CREATING LOOK-UP TABLES ---'
    if verbose: print ' - Each look-up table will contain '+str(Nentries)+\
                      ' - entries and have shape '+str(np.zeros([Nk,NLstar]).shape)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print '   --- START @ '+strftime("%a, %d %b %Y %H:%M:%S", localtime())+' ---\n'
    if parallelize:
        dict      = run_parallel(dict,mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,
                                 kvalall,Lstarall,selfctval,errdist=errdist,verbose=verbose,dummyval=dummyval)
        runstring = 'parallelrun'
    else:
        dict      = run_standard(dict,mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,
                                 kvalall,Lstarall,selfctval,errdist=errdist,verbose=verbose,dummyval=dummyval)
        runstring = 'standardrun'
    if verbose: print '   --- DONE @ '+strftime("%a, %d %b %Y %H:%M:%S", localtime())+' ---\n'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if verbose: print ' - Saving look-up table dictionary to binary'
    outfile = datafile.replace('.fits','_lookuptable_'+str(Nk)+'x'+str(NLstar)+'tab_'+runstring+'.npz')
    if allfields: outfile = outfile.replace('.npz','_allfields.npz')
    if dummyval:  outfile = outfile.replace('.npz','_dummyval.npz')
    np.savez(outfile,**dict) # save array as binary file
    if verbose: print ' - Saved output dictionary with the '+str(Nfields)+' look-up tables to \n   '+outfile

    return outfile
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def singlefield(mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,kvalall,Lstarall,selfctval,
                dummyval=False,verbose=False,printstep=1000.0,errdist='normal'):
    """
    Calculate the look-up table for a single field
    """
    if verbose: print '\nInitiating table for '+fieldname+' on '+strftime("%a, %d %b %Y %H:%M:%S", localtime())
    fieldtable = np.zeros([Nk,NLstar])
    count      = 0.0 # resetting counter
    dL         = fielderr
    if mpdclass.limNsig != None: # make sure 1sigma uncertainties are used
        dL = dL/mpdclass.limNsig
    Llim       = fieldlim

    if errdist == 'magbias':
        mpdclass.setmagbiasval(fieldname,verbose=True)

    for kk in xrange(Nk):
        for ll in xrange(NLstar):
            count     = count + 1.0
            if dummyval:
                intvalue  = np.abs(kvalall[kk]) # dummy values for quick run used when testing
            else:
                intvalue  = mpdclass.intSelintFtilde(kvalall[kk],Lstarall[ll],dL,Llim,
                                                     selfct=selfctval,field=fieldname,Npoints=100)
                if intvalue < 0:
                    print '--- WARNING...ERROR balff_createLookupTable.singlefield():', \
                          fieldname, ' intvalue < 0 - it should not be possible! ---'
                    pdb.set_trace()

            fieldtable[kk,ll] = intvalue
            if verbose and (count/printstep == round(count/printstep)):
                print ' - wrote value '+str(count)+' to table'

    return fieldtable
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def run_standard(dict,mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,kvalall,Lstarall,selfctval,errdist='normal',
                 dummyval=False,verbose=True):
    """
    Loop over individual fields and create the look-up tables

    """
    Nfields = len(fieldname)
    for ff in xrange(Nfields):
        fieldtable = singlefield(mpdclass,fieldname[ff],fielderr[ff],fieldlim[ff],Nk,NLstar,kvalall,Lstarall,
                                 selfctval,dummyval=dummyval,verbose=verbose,errdist=errdist)

        dict[fieldname[ff]+'lookup'] = fieldtable  # filling dictionary

    return dict
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def run_parallel(dict,mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,kvalall,Lstarall,selfctval,errdist='normal',
                 verbose=True,dummyval=False):
    """
    Loop over individual fields calculating the look-up tables in parallel using the multiprocessing package

    """
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def worker(mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,kvalall,Lstarall,selfctval,dummyval,return_dict):
        """
        Multiprocessing worker function
        """
        fieldtable = singlefield(mpdclass,fieldname,fielderr,fieldlim,Nk,NLstar,kvalall,Lstarall,
                                 selfctval,errdist=errdist,dummyval=dummyval,verbose=verbose)
        return_dict[fieldname+'lookup'] = fieldtable
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if verbose: print ' ---- Starting multiprocess run of field look-up tables: ---- '
    tstart  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Nfields = len(fieldname)

    mngr = multiprocessing.Manager() # initialize Manager too kepp track of worker function output
    return_dict = mngr.dict()        # define Manager dictionar to store output from Worker function in
    jobs = []

    for ff in xrange(Nfields):
        job = multiprocessing.Process(target=worker,
                                      args=(mpdclass,fieldname[ff],fielderr[ff],fieldlim[ff],Nk,NLstar,
                                            kvalall,Lstarall,selfctval,dummyval,return_dict),name=fieldname[ff])

        jobs.append(job)
        job.start()
        #job.join() # wait until job has finished

    for job in jobs:
        job.join()

    tend = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        print '\n ---- The parallel_run finished running the jobs for all fields ----'
        print '      Start        : '+tstart
        print '      End          : '+tend
        print '      Exitcode = 0 : job produced no error '
        print '      Exitcode > 0 : job had an error, and exited with that code (signal.SIGTERM)'
        print '      exitcode < 0 : job was killed with a signal of -1 * exitcode (signal.SIGTERM)'

        for job in jobs:
            print ' - The job running field ',job.name,' exited with exitcode: ',job.exitcode


    if verbose: print ' - Adding output from parallelized run to dictionary'

    for key in return_dict.keys():
        dict[key] = return_dict[key]  # filling dictionary

    return dict
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def compare_dictionaries(dictionarylist,verbose=True,fieldcompare=None,showplot=True):
    """
    Compare the content of a list of dictionaries

    --- EXAMPLE OF USE ---
    d1  = 'dataarray_BoRG13sample130924_5sig_lookuptable_3x3tab_standardrun_allfields_top3.npz'
    d2  = 'dataarray_BoRG13sample130924_5sig_lookuptable_3x3tab_standardrun_allfields_top3.npz'
    d2  = 'dataarray_BoRG13sample130924_5sig_lookuptable_15x15tab_allfields_BADnorm.npz'
    clt.compare_dictionaries([d1,d2],fieldcompare='borg_0214+1255')

    d1  = 'dataarray_BoRG13sample130924_5sig_lookuptable_3x3tab_allfields_BADnorm.npz'
    d2  = 'dataarray_BoRG13sample130924_5sig_lookuptable_3x3tab_standardrun_allfields_Full.npz'
    d3  = 'dataarray_BoRG13sample130924_5sig_lookuptable_3x3tab_parallelrun_allfields_Full.npz'
    clt.compare_dictionaries([d1,d2,d3],fieldcompare='borg_1632+3733')

    """
    Ndict = len(dictionarylist)

    if Ndict == 1:
        sys.exit(' - Only one dictionary in list. Expected at least [dict1,dict2] ) --> Aborting')

    if verbose: print ' - The following dictionaries were found:'
    for dd in xrange(Ndict):
        if verbose: print '   Dict'+str("%.3d" % dd)+' :  ',dictionarylist[dd]

    Nkvals = []
    NLvals = []

    for dd in xrange(Ndict):
        dictstr = str("%.3d" % dd)
        dict    = np.load(dictionarylist[dd])

        if verbose: print '\n ---- '+dictstr+' ---- '
        if verbose: print ' - Nfields     :   ',(len(dict.keys())-2.)/3.

        if verbose: print ' - Fields      :   ',
        for key in dict.keys():
            if 'lookup' in key:
                if verbose: print key[:-6],
        if verbose: print ''

        kdim, Lstardim   = dict['kdim'], dict['Lstardim']
        if verbose: print ' - kdim        :   ',kdim
        if verbose: print ' - Lstardim    :   ',Lstardim
        Nkvals.append(len(kdim))
        NLvals.append(len(Lstardim))

    if fieldcompare:
        if verbose: print '\n ---- Comparing data for ---- ',fieldcompare

    for dd in xrange(Ndict):
        dictstr = str("%.3d" % dd)
        dict    = np.load(dictionarylist[dd])

        if verbose: print '\n ---- '+dictstr+' ---- '

        lim, err, lookup = [dict[fieldcompare+arg] for arg in ['lim','err','lookup']]
        if verbose: print ' - lim value     :   ',lim
        if verbose: print ' - err value     :   ',err


    if (Ndict == 2) & (len(np.unique(Nkvals)) == 1) & (len(np.unique(NLvals)) == 1):
        dict1    = np.load(dictionarylist[0])
        lookup1  = dict1[fieldcompare + 'lookup']

        dict2    = np.load(dictionarylist[0])
        lookup2  = dict2[fieldcompare + 'lookup']

        if verbose: print ' - The difference between the two lookup tables '
        print '   Tab1: \n',lookup1
        print '   Tab2: \n',lookup2
        print '   Diff: \n',lookup1-lookup2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig  = plt.figure()
    ax   = fig.gca(projection='3d')
    pnamesurf = 'compare_dictionaries_surfaceplot.pdf'

    colormaps = [cm.gray,cm.winter,cm.autumn,cm.summer]

    for dd in xrange(Ndict):
        dict = np.load(dictionarylist[dd])

        X = np.log10(dict['Lstardim'])
        Y = dict['kdim']
        if dd == 1:
            Y = Y.T
        X, Y = np.meshgrid(X, Y)
        Z = np.log10(dict[fieldcompare + 'lookup'])

        #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colormaps[dd], linewidth=0,
                               antialiased=False, alpha=0.4)
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=np.min(Y), cmap=cm.coolwarm)

        dict.close()

    ax.set_xlabel('log10( L*/[1e44 erg/s] )')
    ax.set_ylabel('k')
    ax.set_zlabel('log10( int S int Ft dLtrue dLobs )')

    if showplot:
        plt.show()
    else:
        plt.savefig(pnamesurf)
        if verbose: print ' - Plotting look-up table surfaces to',pnamesurf

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


