"""
----------------------------
   NAME
----------------------------
 balff_mpd.py
----------------------------
   PURPOSE/DESCRIPTION
----------------------------
 class used to calculate ln() of the marginal posterior probability for dropouts
 Can be run using the pymc MCMC samler with balff_run.py
----------------------------
EXAMPLES/USAGE
----------------------------
import numpy as np
import balff_mpd as mpd
fitstable = 'dataarray_Bradley2012.fits'
lookup    = 'dataarray_Bradley2012_lookuptable_11x11tab.npz'
mpdclass  = mpd.balff_mpd(fitstable,lookuptable=lookup,verbose=True)
param     = np.array([-2.1,1.0]) 
pN        = mpdclass.getlnprob(param)

"""
#-------------------------------------------------------------------------------------------------------------
# IMPORTING MODULES
import balff_utilities as butil
import sys
import numpy as np
import pdb
import pyfits
import scipy
import mpmath
import os
#-------------------------------------------------------------------------------------------------------------
class balff_mpd:
    def __init__(self, datafitstable, lookuptable=None, selectionfunction=0, verbose=False,
                 loadselfct=False, errdist=None, sigsamp='5', contamfrac=0.0, logLmin=-4,
                 schechmin=1E-5, datafileonly=False,emptysim=False, LFredshift=8):
        """
        Setting up the marginal posterior distribution class

        --- INPUT ---
        datafitstable     : Binary Fits table containing the data used to calculate the mpd
                            The class will turn this into a numpy array with dimensions N x M where N is
                            the number of objects in the selection (one for each row) and M is the number
                            of parameters. The following M parameters (columns) are expected in the fits
                            table (at least):
                                OBJNAME       Name of object
                                FIELD         Field the object was found in
                                LOBJ          Luminoisity of object [10^44 erg/s]
                                LOBJERR       Uncertainty on lumintoisiy LOBJ
                                LFIELDLIM     X sigma limiting luminoisity of FIELD [10^44 erg/s]
                                LFIELDLIMERR  Uncertainty on luminoisity limit LFIELDLIM
                            Furthermore the fits table header should contain the keyword
                                LLIMSIFG      Number of sigmas (X) the limitng luminosity LFIELDLIM corresponds to
                            The fits table can be created with either balff_createDataArray.py or
                            balff_createDataArray_sim.py
        lookuptable         The name of the lookuptable to use. Default is None, i.e., all calculations are
                            done while running, which is very time-consuming.
                            A look-up table can be created with balff_createLookupTable.py
        selectionfunction : The selection function to use in calculations. Choices are:
                                0 = Step function (balff_mpd.stepfct) - DEFAULT
                                1 = Tabulated (real) selection functions (balff_mpd.tabselfct)
                            (ignored if lookuptable is provided as slection function is only
                            used when calculating intSelintFtilde).
        errdist           : The error distribution to use in model. Chose between
                            'normal'       A normal (gauss) distribution of errors (DEFAULT)
                            'lognormal'    A log normal distribution of errors
                            'normalmag'    A log normal in L turned into normal in mag
                            'magbias'      Use gauss error distribution taking magnification bias into account
                                           where p(mu) is a sum of up to 3 gaussian distributions
        verbose           : Toggle verbosity
        loadselfct        : Set this key to load field selection function dictionaries just once, instead
                            of every time balff_mpd.selfctFIELD() is called.
        sigsamp           : The sigma cut of the sample in data array. Used to determine which selection
                            functions to load; the ./balff_data/selectionfunctions/*_SN???*.
                            DEFAULT is SN5 selection functions.
        contamfrac        : contamination fraction to correct probabilities with.
                            Note that fields beginning with 'UDF' or 'ERS' are set to have contamination = 0.0
                            autopmatically as UDF and ERS selections functions already include contamination.
        logLmin           : The lower integration limit of the incomplete gamma function normalizing
                            the detection probabilities
        schechmin         : Minimum luminosity for the Schechter function, needed to stop integral blowing up
                            at low luminosity end
        datafielonly      : If True the fields without high z objects (candidates) are ignored, i.e., only the
                            fields in the datafitstable are considered.
        emptysim          : if True use "simulated" empty fields instead of the actual empty fields.
        LFredshift        : The redshift at which the luminoisity function is determined. Used for distance
                            modulus when turning 5-sigma limiting magnitude into absolute luminosity and
                            selection function integrations.
        """
        self.vb = verbose
        self.lutab = lookuptable
        self.selfct = selectionfunction
        self.data, self.Nobj, self.limNsig = self.readdata(datafitstable)
        self.errdist = errdist
        self.sigsamp = sigsamp
        self.datafileonly = datafileonly
        self.LFz = LFredshift

        # getting the number of objects in each field
        self.fields = self.data['FIELD']
        self.ufield = np.unique(self.fields)
        self.Nfields = len(self.ufield)
        self.Nobjfield = np.zeros(self.Nfields)
        for ff in xrange(self.Nfields):
            objent = np.where(self.fields == self.ufield[ff])[0]
            self.Nobjfield[ff] = len(objent)

        # creating array to contain contamination fraction to apply
        self.contamfrac    = contamfrac
        self.contamfracarr = np.ones(self.Nobj) * self.contamfrac
        for ff in xrange(self.Nobj): # setting HUDF contamination to 0 (already accounted for in selection function)
            if (self.fields[ff][0:3] == 'UDF') or (self.fields[ff][0:3] == 'ERS'):
                self.contamfracarr[ff] = 0.0

        # default integration limits
        self.Lmin = 10**logLmin
        self.Lmax = 1000 #np.inf; no need to integrate out to LJobs ~ 100s-1000s times L* (unphsyical)
        
        # Minimum luminosity for schechter function
        self.schechmin = schechmin

        # getting areas of fields and sky
        self.Asky = 41252.96 * 60 * 60     # area on sky in square arcmin
        if emptysim == True: # use "simulated" empty fields
            self.emptysim = True
            self.Nfields_nocand = 4
            self.aareas  = np.zeros(self.Nfields_nocand) + self.Asky/(self.Nfields+self.Nfields_nocand+0.0)
            self.afields = np.asarray(['FIELDEMPTY'+str(ii+1) for ii in xrange(self.Nfields_nocand)])
            self.ufield_nocand = self.afields
        else: # use actual empty fields
            self.emptysim = False

            if self.fields[0][0:5] != 'FIELD': # only load infor if not simulated data
                # loading limits from field info file for fields w/o high-z candidates
                self.finfonames, self.finfoLJlim, self.finfodLJlim = self.loadfinfo()

                # getting fields not in dataarray (i.e. fields w/o high z candidates)
                self.ufield_nocand = []
                for nn in xrange(len(self.finfonames)):
                    if len(np.where(self.ufield == self.finfonames[nn])[0]) == 0:
                        self.ufield_nocand.append(self.finfonames[nn])
                self.ufield_nocand  = np.asarray(self.ufield_nocand)
                self.Nfields_nocand = len(self.ufield_nocand)
            else:
                self.Nfields_nocand = 0.0

            self.afields, self.aareas = self.loadareas() # load field areas

        if self.vb: print ':: balff_mpd :: Datafile loaded               : ', datafitstable
        if self.vb: print ':: balff_mpd :: Objects found                 : ', self.Nobj
        if self.vb: print '                Spread over                   : ', self.Nfields, '/',(self.Nfields_nocand+self.Nfields), ' fields'
        if self.vb and (self.lutab == None): print ':: balff_mpd :: Look-up table               :  None provided'
        if self.vb and (self.lutab != None): print ':: balff_mpd :: Look-up table               : ', self.lutab
        if self.vb: print ':: balff_mpd :: Selection functions to load   : *SN', self.sigsamp, '*'
        if self.vb: print ':: balff_mpd :: Contamination fraction used   : ', contamfrac

        self.loadselfct = loadselfct
        if self.loadselfct == True:
            if self.vb: print ':: balff_mpd :: Loading tabulated selection functions as requested'

            borgfield = self.finfonames.tolist()
            bf        = []
            for borgf in borgfield:
                if (not borgf.startswith('UDF')) & (not borgf.startswith('ERS')):
                    bf.append(borgf)
            borgfield = bf

            self.dictall = {}
            for jj in xrange(len(borgfield)):
                selfctpath = './balff_data/selectionfunctions/'
                selfctdict = selfctpath + 'Szm_' + borgfield[jj] + '_SN' + self.sigsamp + '.npz'
                dictSF = np.load(selfctdict, mmap_mode='r+')
                self.dictall[borgfield[jj] + 'SF'] = dictSF
                #dictSF.close()

                comfctdict = selfctpath + 'C_' + borgfield[jj] + '_SN' + self.sigsamp + '.npz'
                dictCF = np.load(comfctdict, mmap_mode='r+')
                self.dictall[borgfield[jj] + 'CF'] = dictCF
                #dictCF.close()
        if self.vb: print '\n'


        if self.errdist == 'magbias':
            print ':: balff_mpd :: Loading tabulated parameters for magbias errdist with multiple gaussians'
            mbiasfields = np.genfromtxt('./balff_data/fields_magbiaspdf.txt',usecols=0, dtype='S30')
            mbiastab    = np.genfromtxt('./balff_data/fields_magbiaspdf.txt',comments='#')[:,1:]

            # turn into dictionary for easy extraction
            self.magbiasdic = {}
            for mm in xrange(len(mbiasfields)):
                self.magbiasdic[mbiasfields[mm]] = mbiastab[mm]

            ignoremagbias = False
            if ignoremagbias:
                print ' - **NB** Setting all magbias dic keys to [1.0,1.0,1.0,0.0] (ignore magbias)'
                for key in self.magbiasdic.keys():
                    self.magbiasdic[key] = [1.0,1.0,1.0,0.0]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def readdata(self, fitstable):
        """
        Returning the data array for the individual objects from the
        provided binary fits table.
        """
        dat = pyfits.open(fitstable)
        datTB = dat[1].data
        Nobj = len(datTB['OBJNAME'])
        try:
            LJlimNsig = dat[1].header['LLIMSIG']
        except:
            LJlimNsig = None
        return datTB, Nobj, LJlimNsig

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def gauss(self, L, Lmean, dL, nonorm=0):
        """
        Return gaussian value

        --- INPUT ---
        L       : luminsoity (same units as Lmean)
        Lmean   : center of gaussian
        dL      : the standard deviation of the gaussian around Lmean
        nonorm  : if nonorm = 1 the normalizing factor 'norm' is ignored
        """
        norm = 1.0
        if nonorm == 0: # only calculating norm if nonorm == 0 (DEFAULT)
            norm = 1 / dL / np.sqrt(2 * np.pi)
        exp = np.exp(-(L - Lmean) ** 2 / (2 * dL ** 2))
        gval = norm * exp
        return gval

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Get areas and names of all fields
    def loadareas(self):
        """
        Function to load the areas of each of the fields.
        Returns the names of each field as well as the corresponding area in two separate arrays
        """
        anames = self.fieldinfo['fieldname']
        aareas = self.fieldinfo['fieldarea']
        return anames, aareas

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Load field info
    def loadfinfo(self, Nsiglim=3.0, Msun=5.48, band='Jbradley2012',
                  finfofile='./balff_data/fields_info.txt'):
        """
        Function to load field info for all fields.
        Done to obtain names and dLlim for fields w/o high-z objects (candidates).

        """
        self.fieldinfo = np.genfromtxt(finfofile, names=True, skip_header=7,dtype=None, comments='#')
        finfo = np.unique(self.fieldinfo['fieldname'], return_index=True)
        fnames = finfo[0]
        Nfields = len(fnames)

        if Nsiglim == None:
            print " -----> NB! balff_mpd.loafinfo didn't get a Nsiglim value from self.limNsig --> Aborting"
            pdb.set_trace()
        elif Nsiglim != self.limNsig:
            print " -----> NB! balff_mpd.loafinfo the Nsiglim != self.limNsig"

        LJlim = np.zeros(Nfields)
        dLJlim = np.zeros(Nfields)
        for ff in xrange(Nfields):
            FieldJent = np.where(self.fieldinfo['fieldname'] == fnames[ff])
            J5siglim = self.fieldinfo['maglim5sigma'][FieldJent]
            Avval    = self.fieldinfo['Av'][FieldJent]

            Mabslim = butil.magapp2abs(J5siglim, self.LFz, 'dummy', 'dummy', Av=Avval, band=band, cos='dummy')
            LJlimval = butil.Mabs2L(Mabslim, MUVsun=Msun) / 5.0 * Nsiglim # turn 5sigma into Nsiglim sigma limits
            if len(LJlimval[0]) > 1:
                sys.exit(' balff_mpd ERROR: Found more than 1 magnitude limit for '+fnames[ff]+
                         ' in ./balff_data/fields_info.txt')
            LJlim[ff]  = LJlimval[0]
            magerrJ    = self.fieldinfo['magmedfield'][FieldJent]

            dLJlim[ff] = LJlim[ff] * np.log(10) / 2.5 * magerrJ
        return fnames, LJlim, dLJlim

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def magbiasfct_multigauss(self, Lobs, Ltrue, dL, gaussparam=None, nonorm=0):
        """
        Return the error term accounting for the magnificantion bias in the data.
        p(mu) assumed to be the combination of N normal distributions
        where N=1,2,3

        --- INPUT ---
        Lobs    : Observed luminosity (same units as Ltrue)
        Ltrue   : True instrinsic luminosity
        dL      : Uncertainty (standard dev) on the observed luminosities
        nonorm  : if nonorm = 1 the normalizing factor 'norm' is ignored
        """

        if gaussparam != None:
            p1, meanmu1, dmu1, p2, meanmu2, dmu2 , p3, meanmu3, dmu3 = gaussparam
        else: # to prevent carrying through the gauss parameters they are grabbed
              # from "self.*" which can all be set with self.setmagbiasval
            p1       = self.magbiasfield_dic['p1_field']
            meanmu1  = self.magbiasfield_dic['meanmu1']
            dmu1     = self.magbiasfield_dic['dmu1']
            
            if p1 == -99: # use standard gauss if p == -99 NO magnification correction
                magbiasval = self.gauss(Ltrue, Lobs, dL, nonorm=nonorm)
            else:
                G1 = self.gaussfct_magbias(Lobs, Ltrue, dL, dmu1, meanmu1, nonorm=nonorm)
                magbiasval = p1 * G1

                if self.magbiasfield_dic.has_key('p2_field'):
                    p2       = self.magbiasfield_dic['p2_field']
                    meanmu2  = self.magbiasfield_dic['meanmu2']
                    dmu2     = self.magbiasfield_dic['dmu2']
                    
                    G2 = self.gaussfct_magbias(Lobs, Ltrue, dL, dmu2, meanmu2, nonorm=nonorm)
                    magbiasval += p2 * G2

                if self.magbiasfield_dic.has_key('p3_field'):
                    p3       = self.magbiasfield_dic['p3_field']
                    meanmu3  = self.magbiasfield_dic['meanmu3']
                    dmu3     = self.magbiasfield_dic['dmu3']
                    
                    G3 = self.gaussfct_magbias(Lobs, Ltrue, dL, dmu3, meanmu3, nonorm=nonorm)
                    magbiasval += p3 * G3

        return magbiasval

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def gaussfct_magbias(self, Lo, Lt, dL, dmu, meanmu, nonorm=0, normtype='Lo'):
        """
        Gaussian-like functions used in magbiasfct_multigauss
        """
        norm = 1.0
        if nonorm == 0: # only calculating normalization if nonorm == 0 (DEFAULT)
            norm = 1.0 / np.sqrt(dmu**2 * Lt**2 + dL**2) / np.sqrt(2 * np.pi)

            # capturing the difference between normalizing wrt. Lt or Lo
            if normtype == 'Lt':
                norm = norm * meanmu
            elif normtype == 'Lo':
                pass
            else:
                sys.exit(' - Invalid "normtype" ('+normtype+') selected in gaussfct_magbias --> ABORTING')

        exp = np.exp(-(Lo - meanmu * Lt) ** 2 / (2 * (dmu**2 * Lt**2 + dL**2)) )

        gaussval = norm * exp
        return gaussval
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def setmagbiasval(self, fieldname, verbose=False):
        """
        Setting the values for the magnifiaction bias error distribution
        Magbias files must be in the format:
            Fieldname
            Field mean magnification mu
            Weighting of 1st gaussian (p=0-1)
            Mean of 1st gaussian (meanmu)
            Std of 1st gaussian (dmu)
            ...
            Weighting of Nth gaussian
            Mean of Nth gaussian
            Std of Nth gaussian  
        where N=1,2,3      
        """
        # All should have at least 1 gauss function
        self.fieldname  = fieldname
        self.fieldmean  = self.magbiasdic[fieldname][0]
        self.magbiasfield_dic = {}
        self.magbiasfield_dic['fieldmean']  = self.magbiasdic[fieldname][0]
        self.magbiasfield_dic['p1_field']   = self.magbiasdic[fieldname][1]
        self.magbiasfield_dic['meanmu1']    = self.magbiasdic[fieldname][2]
        self.magbiasfield_dic['dmu1']       = self.magbiasdic[fieldname][3]

        # 2 gaussians?
        if len(self.magbiasdic[fieldname]) > 4:
            self.magbiasfield_dic['p2_field']   = self.magbiasdic[fieldname][4]
            self.magbiasfield_dic['meanmu2']    = self.magbiasdic[fieldname][5]
            self.magbiasfield_dic['dmu2']       = self.magbiasdic[fieldname][6]
            
        # 3 gaussians?
        if len(self.magbiasdic[fieldname]) > 7:
            self.magbiasfield_dic['p3_field']   = self.magbiasdic[fieldname][7]
            self.magbiasfield_dic['meanmu3']    = self.magbiasdic[fieldname][8]
            self.magbiasfield_dic['dmu3']       = self.magbiasdic[fieldname][9]
        
        if verbose:
            print ' - Setting the magnification bias values for ',fieldname
            print '  ',self.magbiasfield_dic

        dmu = ['dmu1','dmu2','dmu3']
        self.magbiasfield_dic['dmus'] = [self.magbiasfield_dic[i] for i in dmu if i in self.magbiasfield_dic]
        
        return

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def lognormal(self, L, Lmean, dL, nonorm=0):
        """
        Return lognormal value

        --- INPUT ---
        L       : luminosity (same units as Lmean)
        Lmean   : center of gaussian
        dL      : the standard deviation of the gaussian around Lmean
        nonorm  : if nonorm = 1 the normalizing factor 'norm' is ignored
        """
        norm = 1.0
        if nonorm == 0: # only calculating norm if nonorm == 0 (DEFAULT)
            norm = 1 / dL / np.sqrt(2 * np.pi)

        lnfact = np.log(10) / 2.5
        exp = np.exp(-(np.log10(L) - np.log10(Lmean)) ** 2 / (2 * dL ** 2) * (np.log(10) * L) ** 2)

        lgval = lnfact * norm * L * exp
        return lgval

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def schechter(self, L, kval, Lstar, phistar, nonorm=0):
        """
        The Schechter function

        --- INPUT ---
        L       : luminsoity (same units as Lstar)
        kval    : k = alpha+1 is the shape parameter of the gamma distribution
        Lstar   : the scale parameter. Characteristic luminosity
        phistar : normalizing factor. The number density, i.e., Mpc^-3
                  Is not used if nonorm = 1
        nonorm  : if nonorm = 1 the normalizing factor 'norm' is ignored
        """
        if (np.asarray(L) < 0).any():
            sys.exit(':: balff_mpd.schechter :: Some of the input luminosities are < 0 which is unphysical --> ABORTING')
        norm = 1.0
        if nonorm == 0: # only calculating norm if nonorm == 0 (DEFAULT)
            gam = self.incompletegamma(kval, Lmin=self.Lmin, Lstar=Lstar)
            norm = phistar / Lstar / gam
        pow = (L / Lstar) ** (kval - 1.0)
        exp = np.exp(-(L / Lstar))

        schval = norm * pow * exp

        # To stop integral blowing up at low luminosity end
        schval[L < self.schechmin] = 0.0

        return schval

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def incompletegamma(self, kval, Lmin=0, Lstar=0):
        """
        The Incomplete Gamma function, i.e., integral from Lmin to inf of
        t**(alpha+1)*exp(-t)

        --- COMMENT ---
        This function is also given by mpmath.gammainc(kval,Lmin,np.inf)
        [NOT by scipy.sepcial.gammainc which is normalized by gamma(kval)]

        --- INPUT ---
        kval    : k = alpha+1 is the shape parameter of the gamma distribution.
        Lmin    : the lower integration limit.
        """
        incomg = np.double(mpmath.gammainc(kval, a=Lmin / Lstar, b=np.inf))
        return incomg

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def stepfct(self, L, Llim, gt=True):
        """
        A step selection function
        If L > Llim then 1 is returned, otherwise 0.0
        Default is L > Llim. If gt=False is set then 1 is returned if L < Llim
        """
        value = 0.0 # default output
        if gt:
            if (L > Llim): value = 1.0
        else:
            if (L < Llim): value = 1.0
        return value

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def tabselfct(self, L, field, zobj=8, selUDF='step'):
        """
        Tabulated selection functions for BoRG13 fields and ERS+UDF09

        ---INPUT ---
        L        Object luminosity in 1e44 erg/s
        field    Name of field to return selection function for.
        zobj     If slection function is dependent on reshift (as for BoRG) 
                 use this redshift to obtain value.
        selUDF   Choose selection function for UDF/ERS fields.
                   'step'     Step function (DEFAULT)
                   'interp'   Interpolate values between bincenters
        """
        Mabs = butil.L2Mabs(L, MUVsun=5.48) # Convert L_J to Mabs using MUVsun from www.ucolick.org/~cnaw/sun.html

        if (field[0:3] == 'UDF') or (field[0:3] == 'ERS'):
            value = self.selfctUDFERS(Mabs, selUDF=selUDF, plot=0)
        elif (field in self.afields):
            mapp = butil.magabs2app(Mabs, 'zdummy', 'RAdummy', 'DECdummy', Av='Avdummy', band='Jbradley2012',
                                  cos='WMAP7BAOH0')
            value = self.selfctFIELD(mapp, field, zobj=zobj)
        else:
            if self.vb:
                print ':: balff_mpd.tabselfct :: WARNING: No selection function found for field ' + field
                print '                                   Using step selection function instead.'
            fieldent = np.unique(self.data['FIELD'], return_index=True)
            fields = fieldent[0]
            LJlim = self.data['LFIELDLIM'][fieldent[1]]
            LJlimobj = LJlim[np.where(fields == field)]
            value = self.stepfct(L, LJlimobj, gt=True)

        return value
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def selfctUDFERS(self, Mabs, selUDF='step', plot=0):
        """XXX
        Returning UDF/ERS selection function value for L

        --- INPUT ---
        Mabs     Absolute magnitude of object
        selUDF   Choose selection function for UDF/ERS fields.
                   'step'     Step function (DEFAULT)
                   'interp'   Interpolate values between bincenters
        """
        bincen = [-23.04, -20.74, -20.14, -19.54, -18.94, -18.34, -17.74]
        binedges = [-25.04, -21.04, -20.44, -19.84, -19.24, -18.64, -18.04, -17.44]
        Nobj = [0, 5, 7, 10, 19, 10, 9]
        volumes = [100000.0, 75757.6, 46666.7, 42735.0, 30744.3, 10683.8, 3318.58]
        bincen = np.asarray(bincen)
        binedges = np.asarray(binedges)
        Nobj = np.asarray(Nobj)
        volumes = np.asarray(volumes)
        selfct = volumes / np.max(volumes) # selection function as normaized volumes
        # - - - - - - - - - - - - - - - - - - - - - - - - 
        if selUDF == 'step': # Selection function steps
            if Mabs < np.min(binedges):
                selfctvalue = 1.0
            elif Mabs > np.max(binedges):
                selfctvalue = 0.0
            else:
                brightedge = np.where(binedges < Mabs) # edges brighter than Mabs
                selfctvalue = selfct[brightedge[0][-1]]  # value at closest edge returned
            # - - - - - - - - - - - - - - - - - - - - - - - -
        if selUDF == 'interp': # Selection function interpolated
            if Mabs < np.min(bincen):
                selfctvalue = 1.0
            elif Mabs > np.max(bincen):
                selfctvalue = 0.0
            else:
                bincenadd = np.sort(np.append(bincen, Mabs))
                selfctinterp = butil.interpn(bincen, selfct, bincenadd)
                selfctvalue = selfctinterp[np.where(bincenadd == Mabs)]

        if plot == 1:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(bincen, selfct, 'k-', label='Selection function')
            plt.plot(Mabs, selfctvalue, 'ko', label='Value for Mabs=' + str(Mabs) + ' using ' + selUDF + ' method')
            print 'Selection function value at Mabs =', Mabs, ' was estimated to be ', selfctvalue
            plt.xlabel('M absolute')
            plt.ylabel('Selection Function')
            leg = plt.legend(fancybox=True, loc='lower left', numpoints=1)
            leg.get_frame().set_alpha(0.6)
            plt.show()

        return selfctvalue

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def selfctFIELD(self, mapp, field, zobj=8):
        """
        Returning field selection function value for L

        --- INPUT ---
        mapp     Apparent magnitude of object
        field    Name of field to return selection for.
        zobj     If slection function is dependent on reshift (as for BoRG) 
                 use this redshift to obtain value.
        """
        selfctpath = './balff_data/selectionfunctions/'  #+field+'/'
        # ------------------------------------- SEL FCT -------------------------------------
        selfctdict = selfctpath + 'Szm_' + field + '_SN' + self.sigsamp.replace('.','p') + '.npz'
        if os.path.exists(selfctdict):
            if self.loadselfct == False:
                dictSF = np.load(selfctdict, mmap_mode='r+')
                zval = dictSF['zGrid']
                mval = dictSF['mGrid']
                SF = dictSF['Szm']
                dictSF.close()        # closing file after use
            else:
                zval = self.dictall[field + 'SF']['zGrid']
                mval = self.dictall[field + 'SF']['mGrid']
                SF = self.dictall[field + 'SF']['Szm']

            if (mapp < np.min(mval)):
                selfctvalue = 1.0
            elif (mapp > np.max(mval)) or (zobj > np.max(zval)) or (zobj < np.min(zval)):
                selfctvalue = 0.0
            else:
                if len(np.where(zval == zobj)[0]) == 1: # checking that value is not already in Grid
                    znew = zval
                else:
                    znew = np.sort(np.append(zval, zobj))

                if len(np.where(mval == mapp)[0]) == 1: # checking that value is not already in Grid
                    mnew = mval
                else:
                    mnew = np.sort(np.append(mval, mapp))
                SFnew = butil.interpn(mval, zval, SF, mnew, znew)
                zent = np.where(znew == zobj)[0][0]
                ment = np.where(mnew == mapp)[0][0]
                selfctvalue = SFnew[ment, zent] # the desired value of the interpolated surface
        else:
            sys.exit(':: balff_mpd.selfctFIELD :: Selection function dictionatry for ' + field +
                     ' does not exist. Looked for ' + selfctdict + ' --> ABORTING')

        # ------------------------------------- COMP FCT -------------------------------------
        comfctdict = selfctpath + 'C_' + field + '_SN' + self.sigsamp.replace('.','p') + '.npz'
        if os.path.exists(comfctdict):
            if self.loadselfct == False:
                dictCF = np.load(comfctdict, mmap_mode='r+')
                mval = dictCF['mGrid']
                COMP = dictCF['C']
                dictCF.close()
            else:
                mval = self.dictall[field + 'CF']['mGrid']
                COMP = self.dictall[field + 'CF']['C']

            if mapp < np.min(mval):
                compfctvalue = 1.0
            elif mapp > np.max(mval):
                compfctvalue = 0.0
            else:
                if len(np.where(mval == mapp)[0]) == 1: # checking that value is not already in Grid
                    mnew = mval
                else:
                    mnew = np.sort(np.append(mval, mapp))

                COMPnew = butil.interpn(mval, COMP, mnew)
                ment = np.where(mnew == mapp)[0][0]
                compfctvalue = COMPnew[ment] # the desired value of the interpolated function

        else:
            sys.exit(':: balff_mpd.selfctFIELD :: Completeness function dictionatry for ' + field +
                     ' does not exist. Looked for ' + comfctdict + ' --> ABORTING')

        return selfctvalue * compfctvalue

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def Ftilde(self, L, kval, Lstar, Lmean, dL):
        """
        The Schechter function combined with an error distribution representing measurement errors.

        --- INPUT ---
        L       : luminosity (same units as Lstar)
        alpha   : shape parameter; alpha = k-1 where k is the
                  shape parameter of the gamma distribution
        Lstar   : the scale parameter. Characteristic luminosity
        Lmean   : The mean value of L around which the error gaussian scatter
        dL      : The photometric error on L
        errdist : The error distribution to use in model. Chose between
                         'normal'               A normal (gauss) distribution of errors (DEFAULT)
                         'lognormal'            A log normal distribution of errors
                         'normalmag'            A log normal in L turned into normal in mag
                         'magbias'              Error distribution taking magnification bias into account
                                                where p(mu) is a set of gaussian distributions
        """

        if self.errdist == 'normal':
            errval  = self.gauss(L, Lmean, dL, nonorm=1)           # gaussian value w/o normalising factor
        elif self.errdist == 'lognormal':
            errval  = self.lognormal(L,Lmean,dL,nonorm=1)          # lognormal value w/o normalising factor
        elif self.errdist == 'normalmag':
            errval  = self.normalmag(L,Lmean,dL,nonorm=1)          # normal in mag w/o normalising factor
        elif self.errdist == 'magbias':
            Ltrue   = L
            Lobs    = Lmean
            if self.magbiasfield_dic['p1_field'] == -99: # no normalization if using standard gaussian errval (norm in getlnprob_field_contam)
                errval  = self.magbiasfct_multigauss(Lobs, Ltrue, dL, nonorm=1)
            else: # The norm of multigauss depends on Ltrue so needs to be included for each value
                errval  = self.magbiasfct_multigauss(Lobs, Ltrue, dL, nonorm=0)

        else:
            sys.exit('balff_mpd :: Selected error distribution "' + self.errdist + '" is not valid --> ABORTING')

        sval = self.schechter(L, kval, Lstar, 1.0, nonorm=1)  # schechter value w/o normalising factor

        Ftval = sval * errval

        if (Ftval < 0).any():
            if self.vb: print ' - WARNING: There are values in Ftilde which are < 0... that is bad!'
            if self.vb: print '            Stopping to enable an investigation'
            import pylab as plt
            plt.plot(L,Ftval)
            plt.plot(L,errval)
            plt.plot(L,sval)
            plt.show()
            pdb.set_trace()

        return Ftval

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def intFtilde(self, kval, Lstar, Lobs, Lcent, dL, Nsig=8):
        """
        Integrating the Ftilde(L) function wrt. luminsoity
        
        --- INPUT ---
        k       : k = alpha+1 where k is the shape parameter of the gamma distribution
        Lstar   : Scale parameter. Characteristic luminosity
        Lobs    : The observed value of L around which the error gaussian scatter
        Lcent   : The centre of the the gaussian --> Ltrue
        dL      : Photometric error on L (standard devitation of error-gaussian) 
        Nsig    : Number of sigmas to integrate within (to save computation time)
                  Ideally Ftilde should be integrated over L from 0 to infty
                  (If LJlimsig keyword doesn't exist in datafitstable dL is assumed to be 1 sigma limit)
        """
        if self.errdist == 'magbias':         # Work out sigma of the magnification bias distributions
            sigmax = max(self.magbiasfield_dic['dmus'])
            onesig = np.sqrt(dL**2 + (sigmax*Lcent)**2)
        else:
            onesig = dL # NB! make sure it is actually dL of 1 sigma which is supplied (check self.limNsig)
        
        Lmin = Lcent - Nsig*onesig
        Lmax = Lcent + Nsig*onesig

        if Lmin < 0: 
          Lmin = 1E-3 #1e-32

        xvals     = np.linspace(np.log10(Lmin), np.log10(Lmax), 1000)
        integrand = self.Ftilde(10 ** xvals, kval, Lstar, Lobs, dL)* 10 ** xvals * np.log(10.0)
        result    = scipy.integrate.trapz(integrand, xvals)

        if integrand[0] > 1E-2:
          print '--- WARNING: the lower value of Ftilde > 0.01 - intSelintFtilde may diverge '

        if result < 0:
            print '--- WARNING...ERROR balff_mpd.intFtilde result < 0 - it should not be possible! ---'
            print result
            pdb.set_trace()

        return result

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def plotF(self, Lmean, dL, Nsig=5):
        """
        Plotting the F(L) function wrt. luminosity (Eq A11 in Mason+ 2014)
        
        --- INPUT ---
        Lmean   : The mean value of L around which the error gaussian scatter
        dL      : Photometric error on L (standard devitation of error-gaussian) 
        Nsig    : Number of sigmas to integrate within (to save computation time)
                  (If LJlimsig keyword doesn't exist in datafitstable dL is assumed to be 1 sigma limit)
        """
        onesig = dL # NB! make sure it is actually dL of 1 sigma which is supplied (check self.limNsig)

        Lmin = Lmean - Nsig * 1.6 * onesig # ideally 0
        Lmax = Lmean + Nsig * 1.6 * onesig # ideally np.inf
        
        if Lmin < 0: Lmin = 1e-32
        xvals = np.linspace(np.log10(Lmin), np.log10(Lmax), 2000)

        schmidt_errval  = self.gauss(L, Lmean, dL, nonorm=0)           # gaussian value w/o normalising factor

        Ltrue   = L # orig
        Lobs    = Lmean # orig
        new_errval  = self.magbiasfct_multigauss(Lobs, Ltrue, dL, nonorm=0)

        old_errval  = self.gauss(L, Lmean, dL, nonorm=0)
        L = 10 ** xvals
        import pylab as plt
        plt.clf()
        plt.plot(L,old_errval, label='Schmidt et al. (2014)')
        plt.plot(L, errval, label='This Work - $p(\mu)$ multigauss')
        plt.xlabel(r'$L_t$')
        plt.ylabel(r'$f(L_t)$')
        plt.legend()
        plt.show()        
        
        return

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def iSiFintegrand(self, Lobs, field, kval, Lstar, dL, zobj=8, selUDF='interp'):
        """
        The integrand for balff_mpd.intSelintFtilde containing the tabulated selection functions

        """
        sfval = self.tabselfct(Lobs, field, zobj=zobj, selUDF=selUDF)

        if self.errdist == 'magbias':
            fieldmean = self.fieldmean
        else:
            fieldmean = 1.0

        # if sfval <= 0.005:
        #     iSiFi = sfval
        if sfval == 0.0:
            iSiFi = sfval
        elif sfval >= 1.0:
            iSiFi = self.intFtilde(kval, Lstar, Lobs, Lobs/fieldmean, dL)
        else:
            iSiFi = self.intFtilde(kval, Lstar, Lobs, Lobs/fieldmean, dL) * sfval
        return iSiFi

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def intSelintFtilde(self, kval, Lstar, dL, Llim, selfct=0, field=None, Npoints=100, plot=0):
        """
        Integrating the (Selection function * int Ftilde(L) dL) wrt. the mean of the gaussian
        For a step selection function this corresponds to integrating intFtilde from Llim to np.inf
        
        --- INPUT ---
        k       : k = alpha+1 where k is the shape parameter of the gamma distribution
        Lstar   : Scale parameter. Characteristic luminosity
        selfct  : The selection fucntion to use. Choices are:
                     selfct = 0   A step function at Llim (DEFAULT)
                     selfct = 1   Tabulated selection functions (Bradley et al. 2012)
                                  Use 'field' keyword to specify selection function.
                                  'Llim' becomes dummy variable.
        field   : Name of field if selfct = 1
        Npoints : Number of points log10(L) dimension contains for trapezoidal integration of
                  curve when selfct = 1 
        """
        if self.errdist == 'magbias':
            fieldmean = self.magbiasdic[field][0]
            dmu1      = self.magbiasdic[field][3]
            dmu2      = self.magbiasdic[field][5]
        else:
            fieldmean = 1.0

        if selfct == 0: # using step selection function (i.e. integrating from Llim instead of Lmin)
            result = scipy.integrate.quad(lambda x: self.intFtilde(kval, Lstar, 10 ** x, (10**x)/fieldmean, dL) * 10 ** x * np.log(10), np.log10(Llim), np.log10(self.Lmax))[0]

        elif selfct == 1: # using tabulated (real) selection functions
            Lminval = 0.001 #0.001 #self.Lmin - no need to integrate down to 1e-4 as selfct is 0 there either way
            xxlog   = np.linspace(np.log10(Lminval), np.log10(self.Lmax), Npoints)
            xx      = 10 ** xxlog
            yy      = np.zeros(Npoints)
            for ii in xrange(Npoints):
                yy[ii] = self.iSiFintegrand(xx[ii],field,kval,Lstar,dL,zobj=self.LFz,
                                            selUDF='interp')*xx[ii]*np.log(10)
            result = scipy.integrate.trapz(yy, xxlog) # much faster than quad integration!
            if plot == 1:
              if result > 1E-4:
                import matplotlib.pyplot as plt

                fig = plt.figure()
                kplot = kval
                Lstarplot = Lstar
                xxlog = np.linspace(np.log10(self.Lmin), 2, Npoints)
                xx = 10 ** xxlog
                yy1 = np.zeros(Npoints)
                yy2 = np.zeros(Npoints)
                for ii in xrange(Npoints):
                    yy1[ii] = self.iSiFintegrand(xx[ii], field, kval, Lstar, dL, zobj=self.LFz, selUDF='interp')
                    yy2[ii] = self.tabselfct(xx[ii], field, zobj=self.LFz, selUDF='interp')

                plt.plot(xx, yy1, 'k-', label='k=' + str(kplot) + ' L*=' + str(Lstarplot))
                plt.plot(xx, yy2 * np.max(yy1), 'r--', label='(scaled) Selection function')
                plt.title(field)
                plt.xlabel('L/[1e44erg/s]')
                plt.ylabel('Selection Function * int Ftilde')
                leg = plt.legend(fancybox=True, loc='upper right', numpoints=1)
                leg.get_frame().set_alpha(0.6)
                plt.show()
            if result < 0.: 
              print 'intval = ', result
              result = 0.
        return result

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def lookupvalue(self, kval, Lstar, field, table):
        """
        Obtaining value from provided k-Lstar look-up table.
        Table is expected to be a *.npz file containing a dictionary with
        a k-Lstar grid of look-up values for various fields/surveys
        Table can be created with balff_createLookupTable.py

        -- INPUT --
        kval   : k  value to look up   
        Lstar  : L* value to look up
        field  : string with name of field/survey to use look-up table for
        table  : look-up table to use (i.e. the path/name to the *.npz file)
                 containing a dictionary with a k-L* table for each field/survey
        """
        dict = np.load(table) # loading dictionary with look-up tables

        if field + 'lookup' not in dict:
            sys.exit('Data for ' + field + ' was not found in the look-up dictionary ' + table + ' --> ABORTING')
        lookup = dict[field + 'lookup']  # get look-up table. np.array with shape (Nk,NLstar)
        kdim = dict['kdim']
        Lstardim = dict['Lstardim']

        kadd = np.sort(np.append(kdim, kval))
        Lstaradd = np.sort(np.append(Lstardim, Lstar))
        newsurf = butil.interpn(kdim, Lstardim, lookup, kadd, Lstaradd)
        kent = np.where(kadd == kval)[0][0]
        Lstarent = np.where(Lstaradd == Lstar)[0][0]
        lookupval = newsurf[kent, Lstarent] # the desired value of the interpolated surface
        dict.close()
        return lookupval

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def lnp_object(self, kval, Lstar, Lmean, dL, field):
        """
        Returning the integral over Ftilde
        --- INPUT ---

        """
        if self.errdist == 'magbias':
            fieldmean = self.magbiasdic[field][0]
            dmu1      = self.magbiasdic[field][3]
            dmu2      = self.magbiasdic[field][5]
        else:
            fieldmean = 1.0

        pnum = self.intFtilde(kval, Lstar, Lmean, Lmean/fieldmean, dL)
        if pnum < 1e-128:
            if self.vb: print ':: balff_mpd.lnpnumerator :: Changing pnum = ', pnum, ' to pnum = 1e-128',
            pnum = 1e-128 # making sure that pnum == 0 is not passed to np.log
        return np.log(pnum)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def lnp_detect(self, kval, Lstar, dL, Llim, field):
        """
        Calculating the probability of detection in mpd
        Returning ln() of value.

        The input should be INDEPENDENT of the observations of individual objects, i.e., general 
        values for each field.
        
        --- INPUT ---
        kval     : k = alpha+1 where k is the shape parameter of the gamma distribution
        Lstar    : Scale parameter. Characteristic luminosity
        dL       : Assume average photometric scatter in given field, i.e., expected average
                   photometric errors (standard dev. of error-gaussian) on observations. 
        Lim      : observation limit of field
        field    : field observation stem from (only use as info for look-up tables)

        """
        if self.lutab == None: # deciding wheather to calculate or lookup integral value
            pden = self.intSelintFtilde(kval, Lstar, dL, Llim, selfct=self.selfct, field=field)
        else:
            pden = self.lookupvalue(kval, Lstar, field, self.lutab)

        if pden < 1e-128:
            if self.vb: print ':: balff_mpd.lnpdenominator :: Changing pden = ', pden, ' to pden = 1e-128',
            pden = 1e-128 # making sure that pden == 0 is not passed to np.log
        return np.log(pden)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def getlnprob_field_contam(self, param, highz):
        """
        Calculate ln of the marginal posterior probability for the given parameters
        Same as getlnprob but here only calculating lnpden for each field once to
        save computation time.

        --- INPUT ---
        param       : the paramters k, Lstar and Ngaluniverse
        highz       : booliean list with True/False to determine what
                      highz candidates to include in calculations
        """
        nhighztotal = np.sum(highz)
        if nhighztotal != self.Nobj: # checking that all dropouts are actually used
            print ' - Total number of dropouts after contamination applied = ',np.sum(nhighztotal)
            print '   That is different from the number of high-z objects in datafile --> ABORTING'
            pdb.set_trace()

        kval         = param[0]
        Lstar        = param[1]
        Ngaluniverse = param[2]
        incomgamnorm = self.incompletegamma(kval, Lmin=self.Lmin, Lstar=Lstar) # incomplete gamma for normalizing

        Llim = self.data['LFIELDLIM']
        dLlim = self.data['LFIELDLIMERR']
        Lobs = self.data['LOBJ']

        if self.limNsig != None: # making sure 1sigma limiting mags are used
            dLlim = dLlim / self.limNsig

        # ------------- C1 * C2 * (1- A/Asky * p(I=0 | Theta)^(N-n) * p(L | Theta) terms -------------
        lnp1 = 0.0
        lnp2 = 0.0
        for ff in xrange(self.Nfields):
            objent = np.where(self.fields == self.ufield[ff])[0]
            
            field_cand = self.ufield[ff]

            fent = objent[0] # entry for field

            this_highz = highz[objent]
            nhighz = np.sum(this_highz)


            if self.errdist == 'magbias':
                self.setmagbiasval(field_cand)
            else:
                self.fieldmean = 1.0
            areafactor = self.fieldmean

            Afield = self.aareas[self.afields == field_cand]
            if len(Afield) == 0: # if no area was found use Asky/Nfield (for sims)
                if field_cand[0:5] != 'FIELD': print 'NB - no area found in list for field ',field_cand
                Afield = self.Asky/(self.Nfields)
                if (self.emptysim == True) and (self.datafileonly == False):
                    Afield = self.Asky/(self.Nfields+self.Nfields_nocand)
            Afrac = Afield / areafactor / self.Asky
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.errdist == 'normal':
                gaussnorm = dLlim[fent] * np.sqrt(2 * np.pi)
            elif self.errdist == 'magbias':
                if self.magbiasfield_dic['p1_field'] == -99:
                    gaussnorm = dLlim[fent] * np.sqrt(2 * np.pi)
                else:
                    gaussnorm = 1.0

            prob = np.exp(self.lnp_detect(kval,Lstar,dLlim[fent],Llim[fent],self.fields[fent])) 
            norm = incomgamnorm * gaussnorm * Lstar
            detectionprob = prob / norm

            if detectionprob >= 0.999:
                print '--------------------'
                print 'Field =',field_cand     
                         
                if detectionprob >= 2.0: # if not just numerical errors/uncertainty to 1 break
                    print 'ERROR in getlnprob_field_contam detectionprob > 2.0 for [k,L*,dL,Llim]=[', \
                           kval, Lstar, dLlim[fent], Llim[fent], '] (detectionprob = ', detectionprob,')'
                    #pdb.set_trace()

                print 'WARNING in getlnprob_field_contam detectionprob > 0.999 for [k,L*,dL,Llim]=[', \
                       kval, Lstar, dLlim[fent], Llim[fent], '] (detectionprob = ', detectionprob,')'
                lnp1 = lnp1 + 0 # if integral is ~1 ignore term

            else:
                ffcontam = None # reset ffcontam
                if (field_cand[0:3] == 'ERS') or (field_cand[0:3] == 'UDF'):
                    ffcontam = 0.0
                elif field_cand[0:4] == 'borg':
                    ffcontam = self.contamfrac
                elif  field_cand[0:5] == 'FIELD': # For simulations
                    ffcontam = self.contamfrac

                lnp1 = lnp1 + \
                       np.log(1.0 - Afrac * detectionprob) * \
                       (Ngaluniverse - (1.0-ffcontam)*nhighz)/(1.0-ffcontam)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if nhighz > 0:
                highz_cand = objent[this_highz == True]
                for oo in xrange(nhighz): # looping over all objects in data file
                    oent = highz_cand[oo] # entries for objects
                    if self.errdist == 'normal':
                        gaussnorm = dLlim[oent] * np.sqrt(2 * np.pi)
                    elif self.errdist == 'magbias':
                        if self.magbiasfield_dic['p1_field'] == -99:
                            gaussnorm = dLlim[oent] * np.sqrt(2 * np.pi)
                        else:
                            gaussnorm = 1.0

                    lnp_obj = self.lnp_object(kval, Lstar, Lobs[oent], dLlim[oent], field_cand) \
                              - np.log(incomgamnorm) - np.log(gaussnorm) - np.log(Lstar)

                    lnp2 = lnp2 + lnp_obj
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # ------------- terms for fields w/o highz candidates -------------
        lnp1_nocand = 0.0
        if self.datafileonly == False:
            for ff in xrange(self.Nfields_nocand):
                field_nocand = self.ufield_nocand[ff]
                if field_nocand == 'borg_1437+5043':
                    #print 'skipping field: borg_1437+5043'
                    continue

                if self.errdist == 'magbias':
                    self.setmagbiasval(self.ufield_nocand[ff])
                else:
                    self.fieldmean = 1.0
                areafactor = self.fieldmean

                if self.emptysim == True: # for simulated empty fields use values for FIELD1 in data table
                    dLlim_nocand = dLlim[0]
                    Llim_nocand  = Llim[0]
                    if self.lutab: field_nocand = 'FIELD1' # using lookuptable for FIELD1 for empty fields
                else:
                    dLlim_nocand = self.finfodLJlim[self.finfonames == self.ufield_nocand[ff]]
                    Llim_nocand  = self.finfoLJlim[self.finfonames == self.ufield_nocand[ff]]

                    if self.limNsig != None: # making sure 1sigma values are used
                        dLlim_nocand = dLlim_nocand / self.limNsig

                Afield       = self.aareas[self.afields == self.ufield_nocand[ff]]
                Afrac        = Afield / areafactor / self.Asky
                if self.errdist == 'normal':
                    gaussnorm    = dLlim_nocand * np.sqrt(2 * np.pi)
                elif self.errdist == 'magbias':
                    if self.magbiasfield_dic['p1_field'] == -99:
                        gaussnorm = dLlim_nocand * np.sqrt(2 * np.pi)
                    else:
                        gaussnorm = 1.0

                prob_nocand    = np.exp(self.lnp_detect(kval,Lstar,dLlim_nocand,Llim_nocand,field_nocand))
                norm_nocand    =  incomgamnorm * gaussnorm * Lstar
                dp_nocand = prob_nocand / norm_nocand

                if dp_nocand > 0.999:
                    print '--------------------'
                    print 'Field =',field_nocand

                    if dp_nocand >= 2.0: # if not just numerical errors/uncertainty to 1 break
                        print 'ERROR in getlnprob_field_contam dp_nocand > 2.0 for [k,L*,dL,Llim]=[', \
                               kval, Lstar, dLlim_nocand, Llim_nocand, '] (dp_nocand = ', dp_nocand, ')'
                        #pdb.set_trace()

                    print 'WARNING in getlnprob_field_contam dp_nocand > 0.999 for [k,L*,dL,Llim]=[', \
                           kval, Lstar, dLlim_nocand, Llim_nocand, '] (dp_nocand = ', dp_nocand, ')'
                    lnp1_nocand = lnp1_nocand + 0 # if integral is ~1 ignore term

                else:
                    ffcontam = None # reset ffcontam
                    if (self.ufield_nocand[ff][0:3] == 'ERS') or (self.ufield_nocand[ff][0:3] == 'UDF'):
                        ffcontam = 0.0
                    elif self.ufield_nocand[ff][0:4] == 'borg':
                        ffcontam = self.contamfrac

                    lnp1_nocand = lnp1_nocand + \
                                  np.log(1.0 - Afrac * dp_nocand) * \
                                  (Ngaluniverse - 0.0)/(1.0-ffcontam)

        # ------------- normalizing terms -------------
        lnprior = np.log(1.0)   # constant log-priors => can be set to 1
        lnpN    = np.log(1.0)   # unifrom prior on p(Ngaluniverse)
        lnpVnon = np.log(1.0)   # p(Vmag is non-detect | y,J,H,Theta)
        # ------------- putting together ln(marginal posterior) -------------

        lnBC_highz   = self.lnBC(Ngaluniverse,(1.0-self.contamfrac)*nhighztotal)
        lnBC_contam  = self.lnBC(self.contamfrac/(1.0-self.contamfrac)*Ngaluniverse,self.contamfrac*nhighztotal)

        lnpprob = lnprior + lnpVnon + lnpN + lnp1 + lnp2 + lnp1_nocand + lnBC_highz + lnBC_contam

        return lnpprob
    #-------------------------------------------------------------------------------------------------------------
    #                                 XTRA UTILITIES (NOT USED IN lnprob CALC)
    #-------------------------------------------------------------------------------------------------------------

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def lnBC(self, aval, bval):
        """
        ln(binomial coefficient), i.e., C^a_b
        """
        lnBC = scipy.special.gammaln(aval + 1.0) - \
               scipy.special.gammaln(bval + 1.0) - \
               scipy.special.gammaln(aval - bval + 1.0)

        return lnBC

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def schechterMabs(self, Mabs, phistar, Mstar, kval):
        """
        Schechter function in terms of absolute magnitude.
        Number volume density of galaxies in range (M, M + dM) is phi(M) * dM

        """
        alpha = kval - 1.0
        Mdiff = 10 ** ( (Mstar - Mabs) / 2.5 )
        phi_m = phistar * np.log(10.0) / 2.5 * Mdiff ** (alpha + 1.0) * np.exp(-Mdiff)
        return phi_m

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def test_Ftilde(self):
        """
        Tests that Ftilde does as expected
        """
        # --------------- L = Lmean -------------
        kval, Lstar, dL = 2.0, 1.0, 0.1
        L = Lmean = 0.9
        ftval = self.Ftilde(L, kval, Lstar, Lmean, dL)
        schval = self.schechter(L, kval, Lstar, 'dummy', nonorm=1)
        byhand = (Lmean / Lstar) ** (kval - 1.0) * np.exp(-Lmean / Lstar)
        if self.vb: print ':: mpd.test_Ftilde :: L=Lmean case: Ftilde, schechter, byhand = ', ftval, schval, byhand
        # --------------- L = Lstar -------------
        kval, Lmean, dL = 2.0, 1.0, 0.1
        L = Lstar = 0.9
        ftval = self.Ftilde(L, kval, Lstar, Lmean, dL)
        gval = self.gauss(L, Lmean, dL, nonorm=1) * np.exp(-1.0)
        byhand = np.exp(-(Lstar - Lmean) ** 2 / 2. / dL ** 2.) * np.exp(-1)
        if self.vb: print ':: mpd.test_Ftilde :: L=Lstar case: Ftilde, gauss*C, byhand = ', ftval, gval, byhand
        # --------------- k = 1 -------------
        L, Lstar, Lmean, dL = 0.7, 1.0, 0.8, 0.1
        kval = 1.0
        ftval = self.Ftilde(L, kval, Lstar, Lmean, dL)
        byhand = np.exp(-(L - Lmean) ** 2 / 2. / dL ** 2. - L / Lstar)
        if self.vb: print ':: mpd.test_Ftilde :: k=1     case: Ftilde, byhand = ', ftval, byhand

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def test_intFtilde(self):
        """
        Tests that infFtilde does as expected
        """
        # --------------- k = 1 -------------
        L, Lstar, Lmean, dL = 0.7, 1.0, 0.8, 0.1
        kval = 1.0
        Nsigma = 5
        iftval = self.intFtilde(kval, Lstar, Lmean, Lmean/self.fieldmean, dL, Nsig=Nsigma)
        Lmin = Lmean - Nsigma * dL
        Lmax = Lmean + Nsigma * dL
        byhand, err = scipy.integrate.quad(lambda x: np.exp(-(x - Lmean) ** 2 / 2. / dL ** 2. - x / Lstar), Lmin, Lmax)
        if self.vb: print ':: mpd.test_intFtilde :: k=1 case: intFtilde, byhand = ', iftval, byhand

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def normalmag(self, L, Lmean, dL, nonorm=0):
        """
        Return normal value in mag (corresponds to lognormal in L)

        --- INPUT ---
        L       : luminsoity (same units as Lmean)
        Lmean   : center of gaussian
        dL      : the standard deviation of the gaussian around Lmean
        nonorm  : if nonorm = 1 the normalizing factor 'norm' is ignored
        """
        lnfact = np.log(10) / 2.5
        dM = dL / L / lnfact
        M = -2.5 * np.log10(L)
        Mmean = -2.5 * np.log10(Lmean)

        norm = 1.0
        if nonorm == 0: # only calculating norm if nonorm == 0 (DEFAULT)
            norm = 1 / dL / np.sqrt(2 * np.pi)

        exp = np.exp(-(M - Mmean) ** 2 / (2 * dM ** 2) * lnfact ** 2)

        gval = lnfact * norm * L * exp
        return gval
#-------------------------------------------------------------------------------------------------------------
def modifytrace(kvaldraw,logLstar,logNdraw,logNmax=14.5):
    """
    modifying the trace from a pymc MCMC run

    --- EXAMPLE OF USE ---
    import modifytrace as mt; knew, logLnew, logNnew = mt.modifytrace(kvaldraw,logLstar,logNdraw)

    """

    knew     = kvaldraw[logNdraw < logNmax]
    logLnew  = logLstar[logNdraw < logNmax]
    logNnew  = logNdraw[logNdraw < logNmax]

    return knew, logLnew, logNnew

#-------------------------------------------------------------------------------------------------------------
#                                                      END
#-------------------------------------------------------------------------------------------------------------

