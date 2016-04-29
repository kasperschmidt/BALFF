# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import sys
import numpy as np
import os
import pdb
from matplotlib import cm
import matplotlib.pyplot as plt
import balff_utilities as butil
from mpl_toolkits.mplot3d import axes3d, Axes3D
import balff_createSelectionFunctionFiles as bSF
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def create_npz_files(fieldname,magapp,redshift,selfct,SN,completeness=1.0,
                     outputdic = './balff_data/selectionfunctions/',verbose=True):
    """

    Storing the selection function (and completeness function) in an .npz format
    which is expected to be found in balff_data/selectionfunctions/ by balff_mpd.py

    --- INPUT ---

    fieldname      Name of field to store files for
    magapp         Apparant magnitudes for which the selection function (and completeness) is given for - size Nm
    redshift       Redshifts for which the selection function (and completeness) is given for           - size Nz
    selfct         A Nm x Nz sized array with the selection function values [ selfct.shape = (Nm,Nz) ]
    SN             The signal-to-noise ratio the selection (and completeness) functions correpond to
                   (only used for file naming)
    completeness   A Nm sized array with the completeness function values; if a single value is provided
                   (default is 100% completeness, i.e. completeness=1) this completeness is assumed
                   for all magapp values.
    verbose        Toggle verbosity

    --- EXAMPLE OF USE ---
    import balff_createSelectionFunctionFiles as bSF

    SN = 8
    zval, mval, SF, COMP = bSF.load_selfct('borg_1230+0750',SN)

    selfctdict, comfctdict = bSF.create_npz_files('test_field',mval,zval,SF,SN,completeness=COMP)

    Sval, Cval = bSF.get_selfctvalue('test_field',8,25.1,7.9,plot=True)

    """
    if verbose: print ' - Building selection function dictionary'
    SFdic = {}
    SFdic['Szm']    = np.asarray(selfct)
    SFdic['zGrid'] = np.asarray(redshift)
    SFdic['mGrid'] = np.asarray(magapp)

    selfctdict = outputdic + 'Szm_' + fieldname + '_SN' + str(SN) + '.npz'
    if verbose: print ' - Saving selection function to binary:\n   '+selfctdict
    np.savez(selfctdict,**SFdic)


    if verbose: print ' - Building completeness function dictionary'
    if len(np.atleast_1d(completeness)) == 1:
        completeness = np.zeros(len(magapp))+completeness/1.

    Cdic = {}
    Cdic['C'] = np.asarray(completeness)
    Cdic['mGrid'] = np.asarray(magapp)

    comfctdict = outputdic + 'C_' + fieldname + '_SN' + str(SN) + '.npz'
    if verbose: print ' - Saving selection function to binary:\n   '+comfctdict
    np.savez(comfctdict,**Cdic)


    return selfctdict, comfctdict

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def get_selfctvalue(fieldname,SN,magapp,redshift,selfctpath = './balff_data/selectionfunctions/',
                    plot=False,verbose=True):
    """
    Get selection function value from *fieldname_SN*.npz selection and completeness function

    Routine similar to balff_mpd.balff_mpd.selfctFIELD()

    --- INPUT ---
    fieldname       Field to load and retrive selection function (and completeness) value for
    SN              SN selection function corresponds to (only used for file loading)
    magapp          Apparant magnitude to obtain value for (interpolating if not included in selection function)
    redshift        Redshift to obtaine value for (interpolating if not included in selection function)
    selfctpath      Directory containing selection function npz file
    verbose         Toggle verbosity

    --- EXAMPLE OF USE ---
    import balff_createSelectionFunctionFiles as bSF
    Sval, Cval = bSF.get_selfctvalue('borg_1230+0750',5,25.1,7.9,plot=True)

    """
    if verbose: print ' - Loading selection and completeness functions '
    zval, mval, SF, COMP = bSF.load_selfct(fieldname,SN)

    # ------------------------------------- SEL FCT -------------------------------------
    if (magapp < np.min(mval)):
        selfctvalue = 1.0
    elif (magapp > np.max(mval)) or (redshift > np.max(zval)) or (redshift < np.min(zval)):
        selfctvalue = 0.0
    else:
        if len(np.where(zval == redshift)[0]) == 1: # checking that value is not already in Grid
            znew = zval
        else:
            znew = np.sort(np.append(zval, redshift))

        if len(np.where(mval == magapp)[0]) == 1: # checking that value is not already in Grid
            mnew = mval
        else:
            mnew = np.sort(np.append(mval, magapp))
        SFnew = butil.interpn(mval, zval, SF, mnew, znew)
        zent = np.where(znew == redshift)[0][0]
        ment = np.where(mnew == magapp)[0][0]
        selfctvalue = SFnew[ment, zent] # the desired value of the interpolated surface

    if plot:
        fig = plt.figure()
        ax  = Axes3D(fig)

        X = zval
        Y = mval
        X, Y = np.meshgrid(X, Y)
        Z = SF
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0,
                               antialiased=False, alpha=0.4)
        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)

        # overplot intepolated value:
        ax.scatter(redshift, magapp, selfctvalue, c='k', zorder=5)
        print ' - Value at (redshift,magapp) = (', redshift, ',', magapp, ') was estimated to be ', selfctvalue

        ax.set_xlabel('z')
        ax.set_ylabel('m apparent')
        ax.set_zlabel('Selection Function')
        plt.show()
    # ------------------------------------- COMP FCT -------------------------------------
    if magapp < np.min(mval):
        compfctvalue = 1.0
    elif magapp > np.max(mval):
        compfctvalue = 0.0
    else:
        if len(np.where(mval == magapp)[0]) == 1: # checking that value is not already in Grid
            mnew = mval
        else:
            mnew = np.sort(np.append(mval, magapp))

        COMPnew = butil.interpn(mval, COMP, mnew)
        ment = np.where(mnew == magapp)[0][0]
        compfctvalue = COMPnew[ment] # the desired value of the interpolated function

    if plot:
        X = mnew
        Y = COMPnew

        fig = plt.figure()
        plt.plot(X, Y, 'k-', label='Completeness function')
        plt.plot(magapp, compfctvalue, 'ko', label='Interpolated value for magapp=' + str(magapp))
        print ' - Completeness value at magapp =', magapp, ' was estimated to be ', compfctvalue

        plt.xlim([np.max(mnew),np.min(mnew)])
        plt.xlabel('m apparent')
        plt.ylabel('Completeness Function')
        leg = plt.legend(fancybox=True, loc='lower right', numpoints=1)
        leg.get_frame().set_alpha(0.6)
        plt.show()

    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = zval
        Y = mval
        X, Y = np.meshgrid(X, Y)
        Cgrid = COMP + SF.transpose() * 0.0 # Creating array of C curve at each redshift
        Z = SF * Cgrid.transpose()
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0,
                               antialiased=False, alpha=0.4)
        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)

        # overplot intepolated value:
        ax.scatter(redshift, magapp, selfctvalue, c='g', marker='o', edgecolor='g',
                   zorder=6, label='S(z,m)')
        ax.scatter(redshift, magapp, selfctvalue * compfctvalue, c='k', marker='*', s=100,
                   zorder=5, label='S(z,m)*C(m)')
        print ' - Value at (redshift,magapp) = (', redshift, ',', magapp, ') was estimated to be ', selfctvalue

        leg = ax.legend(fancybox=True, loc='upper right', numpoints=1)
        leg.get_frame().set_alpha(0.6)

        ax.set_xlabel('z')
        ax.set_ylabel('m apparent')
        ax.set_zlabel('Selection Function * Completeness Function ')

        plt.show()

    return selfctvalue, compfctvalue
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def load_selfct(fieldname,SN,selfctpath = './balff_data/selectionfunctions/',verbose=True):
    """

    Loading the .npz selection functions

    --- INPUT ---
    fieldname      Field to load and retrive selection function (and completeness) value for
    SN              SN selection function corresponds to (only used for file loading)
    selfctpath      Directory containing selection function npz file
    verbose         Toggle verbosity

    --- EXAMPLE OF USE --
    import balff_createSelectionFunctionFiles as bSF
    zval, mvalSF, SF, COMP = bSF.load_selfct('borg_1230+0750',8)

    """
    selfctdict = selfctpath + 'Szm_' + fieldname + '_SN' + str(SN).replace('.','p') + '.npz'
    comfctdict = selfctpath + 'C_' + fieldname + '_SN' + str(SN).replace('.','p') + '.npz'

    if os.path.exists(selfctdict) and os.path.exists(comfctdict):
        if verbose: print ' - Loading S(z,m) selection function dictionary:\n   '+selfctdict
        dictSF = np.load(selfctdict, mmap_mode='r+')
        zval   = dictSF['zGrid']
        mvalSF = dictSF['mGrid']
        SF     = dictSF['Szm']
        dictSF.close()        # closing file after use

        if verbose: print ' - Loading C(m) completeness function dictionary:\n   '+comfctdict
        dictCF = np.load(comfctdict, mmap_mode='r+')
        mvalC  = dictCF['mGrid']
        COMP   = dictCF['C']
        dictCF.close()
    else:
        sys.exit(' - Could not find selection function and/or completeness function for ' + fieldname +
                 ' Looked for ' + selfctdict + ' and ' + comfctdict)

    if (mvalSF != mvalC).all():
        sys.exit('The magnitudes in the selection function and completeness function dictionary do not match')

    return zval, mvalSF, SF, COMP
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


