#-------------------------------------------------------------------------------------------------------------
import balff_utilities as butil
import time
import types
import sys
import pyfits
import numpy as np
from scipy.interpolate import interp1d
import astropysics
import astropysics.obstools
from astropysics import coords
from astropysics.constants import choose_cosmology
#-------------------------------------------------------------------------------------------------------------
def getAv(RA,DEC,filter,dustmaps='/Users/kschmidt/work/dustmaps/SFD_dust_4096_%s.fits'):
    """
    returns redening Av (and E(B-V)) for ra and dec (degrees; scalar or numpy arrays) for a given
    HST filter usning E(B-V) from the Schlegel dust maps (also returned)

    --- EXAMPLE OF USE ---
    Avval, EBVval = butil.getAv(51,219,'F125W')

    The extinction corrected apparent mag is then:
    magband_corr = magband - Av_band

    Could also correct for extinction using:
    extlaw    = astropysics.obstools.CardelliExtinction(EBmV=extval, Rv=Rvval)
    magcorr   = extlaw.correctPhotometry(mag,bandwavelength)

    """
    if isinstance(RA,types.FloatType):
        Nvals = 1
    elif isinstance(RA,types.IntType):
        Nvals = 1
    else:
        Nvals = range(len(RA))

    if Nvals > 1:
        gall        = []
        galb        = []
        for ii in Nvals: # looping over RA and Decs and converting to galactic coordiantes
            gcoords = coords.ICRSCoordinates(RA[ii],DEC[ii]).convert(coords.GalacticCoordinates)
            gall.append(gcoords.l.degrees)
            galb.append(gcoords.b.degrees)
    else:
        gcoords = coords.ICRSCoordinates(RA,DEC).convert(coords.GalacticCoordinates)
        gall = gcoords.l.degrees
        galb = gcoords.b.degrees

    Ebv = astropysics.obstools.get_SFD_dust(gall,galb,dustmaps,interpolate=True) # redening from Schlegel maps

    av_ebv = {} # ebv2Av values for HST filters; CCM reddening curve with R_V = 3.1
    av_ebv['F300X']  = 6.78362003559
    av_ebv['F475X']  = 3.79441819047
    av_ebv['F475W']  = 3.82839055809
    av_ebv['F606W']  = 3.01882984135
    av_ebv['F600LP'] = 2.24159324026
    av_ebv['F098M']  = 1.29502816006
    av_ebv['F105W']  = 1.18148250758
    av_ebv['F125W']  = 0.893036743585
    av_ebv['F160W']  = 0.633710427959

    try:
        av_ebv[filter]
    except KeyError:
        sys.exit(':: getAv :: The filter '+filter+' is not accepted as input --> ABORTING')

    Av          = av_ebv[filter] * Ebv

    return Av,Ebv
#-------------------------------------------------------------------------------------------------------------
def magapp2abs(Mapp,zobj,RA,DEC,Av=-99,band='Jbradley2012',cos='WMAP7BAOH0',verbose=False):
    """
    Converting apparent magnitude(s) into absolut magnitude(s)

    Av    : The extinction. If not given it's estimated from the Schlegel maps (time consuming)
            Note that RA and DEC is only used if Av is not given; otherwise they are 'dummys'

    band  : the band to do the calculations for. The default is to use the J band
            conversion used in Bradley et al. (2012). In this case the (extinction correted)
            J-band magnitude is expected and MUV = MJ125 - 47.14 is returned. This
            corresponds to
                  Mabs = mobs - 5.0 * (np.log10(lumdist) - 1.0) + (2.5 * np.log10(1.0 + zobj))
            With k-correction (last term) assuming the source has a flat (beta = -2) SED using
            a 0.3 0.7 0.7 cosmology
            NB! for band='Jbradley2012' zobj, RA, DEC and Av are all dummy values
    cos   : the cosmology to use, e.g.
            'WMAP7BAOH0' (Default) from
            http://lambda.gsfc.nasa.gov/product/map/dr4/params/lcdm_sz_lens_wmap7_bao_h0.cfm
            'WMAP7' from
            http://lambda.gsfc.nasa.gov/product/map/dr4/params/lcdm_sz_lens_wmap7.cfm
    """
    if band == 'Jbradley2012':
        Mabs          = np.array([Mapp - 47.14])
    else:
        if verbose: print ' - No valid band provided so calculating Dlum and K-correction to get Mabs'
        cosmo = choose_cosmology(cos)
        Dlum   = coords.funcs.cosmo_z_to_dist(zobj, zerr=None, disttype='luminosity')*1e6 # luminosity distance in pc
        Kcorrection   = (2.5 * np.log10(1.0 + zobj)) # assumes source has flat (beta = -2) SED.
                                                     # A bluer beta will likely give you an additional
                                                     # correction of about ~0.1 mag or so.
        if isinstance(Mapp,types.FloatType) and Av == -99: # if Av is -99, calculate it
            Av, Ebv = butil.getAv(RA,DEC,band)
            Mabs    = Mapp - 5*np.log10(Dlum)+5 + Kcorrection - Av # corrected absolut magnitude of objects\
        else:
            Mabs    = None
    return Mabs
#-------------------------------------------------------------------------------------------------------------
def magabs2app(Mabs,zobj,RA,DEC,Av=-99,band=None,cos='WMAP7BAOH0'):
    """
    Converting absolute magnitude(s) into apparent magnitude(s)

    Av    : The extinction. If not given it's estimated from the Schlegel maps (time consuming)
            Note that RA and DEC is only used if Av is not given; otherwise they are 'dummys'

    band  : the band to do the calculations for. The default is to use the J band
            conversion used in Bradley et al. 2012. In this case the (extinction correted)
            J-band magnitude is expected and MJ125 = MUV + 47.14 is returned. This
            corresponds to inverting
                  Mabs = mobs - 5.0 * (np.log10(lumdist) - 1.0) + (2.5 * np.log10(1.0 + zobj))
            With k-correction (last term) assuming the source has a flat (beta = -2) SED using
            a 0.3 0.7 0.7 cosmology.
            NB! for band='Jbradley2012' zobj, RA, DEC and Av are all dummy values
    cos   : the cosmology to use, e.g.
            'WMAP7BAOH0' (Default) from
             http://lambda.gsfc.nasa.gov/product/map/dr4/params/lcdm_sz_lens_wmap7_bao_h0.cfm
            'WMAP7' from
            http://lambda.gsfc.nasa.gov/product/map/dr4/params/lcdm_sz_lens_wmap7.cfm
    """
    if band == 'Jbradley2012':
        Mapp          = np.array([Mabs + 47.14])
    else:
        cosmo = choose_cosmology(cos)
        Dlum          = coords.funcs.cosmo_z_to_dist(zobj, zerr=None, disttype='luminosity')*1e6 # luminosity distance in pc
        Kcorrection   = (2.5 * np.log10(1.0 + zobj)) # assumes source has flat (beta = -2) SED.
                                                     # A bluer beta will likely give you an additional
                                                     # correction of about ~0.1 mag or so.
        if isinstance(Mabs,types.FloatType) and Av == -99: # if Av is -99, calculate it
            Av, Ebv = getAv(RA,DEC,band)
            Mapp    = Mabs + 5*np.log10(Dlum) - 5 - Kcorrection + Av # corrected absolut magnitude of objects
        else:
            Mapp    = None
    return Mapp
#-------------------------------------------------------------------------------------------------------------
def Mabs2L(Mabs,MUVsun=5.5):
    """
    Converting absolute magnitude(s) to luminosity in erg/s
    Using a default absolute magnitude of the sun (in UV) of 5.5 from http://www.ucolick.org/~cnaw/sun.html
    """
    Lsun        = 3.839e-11 # 1e44 erg/s
    Lobj        = 10**((MUVsun-Mabs)/2.5)*Lsun  # Luminosity in erg/s
    return Lobj
#-------------------------------------------------------------------------------------------------------------
def L2Mabs(Lobj,MUVsun=5.5):
    """
    Converting luminsoity 10^44 erg/s into absolute magnitude(s)
    Using a default absolute magnitude of the sun (in UV) of 5.5 from http://www.ucolick.org/~cnaw/sun.html
    """
    Lsun        = 3.839e-11 # 1e44 erg/s
    Mabs        = MUVsun - 2.5*np.log10(Lobj/Lsun)
    return Mabs
#-------------------------------------------------------------------------------------------------------------
def interpn(*args, **kw):
    """Interpolation on N-Dimensions

    ai = interpn(x, y, z, ..., a, xi, yi, zi, ...)
    where the arrays x, y, z, ... define a rectangular grid
    and a.shape == (len(x), len(y), len(z), ...)

    KBS:
    Taken from http://projects.scipy.org/scipy/ticket/1727#comment:3
    An alternative is to use scipy.interpolate.LinearNDInterpolator
    but slow according to http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid (problems getting it to work on default Python install)

    -- OPTIONAL INPUT --
    method     Interpolation method to use. Options are
               'linear','nearest', 'zero', 'slinear', 'quadratic', 'cubic'

    -- EAXMPLE --
    newy = butil.interpn(oldx,oldy,newx)

    """
    method = kw.pop('method', 'linear')
    if kw:
        raise ValueError("Unknown arguments: " % kw.keys())
    nd = (len(args)-1)//2
    if len(args) != 2*nd+1:
        raise ValueError("Wrong number of arguments")
    q = args[:nd]
    qi = args[nd+1:]
    a = args[nd]
    for j in range(nd):
        a = interp1d(q[j], a, axis=j, kind=method)(qi[j])
    return a
#-------------------------------------------------------------------------------------------------------------
def simulate_schechter_distribution(alpha, L_star, L_min, N,trunmax=10):
    """
    Generate N samples from a Schechter distribution. Essentially a gamma distribution with
    a negative alpha parameter and cut-off somewhere above zero so that it converges.

    If you pass in stupid enough parameters then it will get stuck in a loop forever, and it
    will be all your own fault.

    Based on algorithm in http://www.math.leidenuniv.nl/~gill/teaching/astro/stanSchechter.pdf

    KBS:-------------------------------------------------------------------------------------
          Code taken from https://gist.github.com/joezuntz/5056136 and modified.
          Schechter distribution with -1 < alpha+1 (k) < -0

          trunmax : To prevent an infinite loop trunmax gives the maximum allowed run time [s].
                    If this time is surpased any found entries are retured or an array of 0s
        -------------------------------------------------------------------------------------
    """
    output = []
    n      = 0
    Nvals  = N
    t0     = time.time()
    while n<N:
        t1   = time.time()
        Lgam = np.random.gamma(scale=L_star, shape=alpha+2, size=N)  # drawing values from gamma dist with k+1
        Lcut = Lgam[Lgam>L_min]                                      # removing L values from gamma dist > L_min
        ucut = np.random.uniform(size=Lcut.size)                     # random values [0:1]
        Lval = Lcut[ucut<L_min/Lcut]                                 # only keeping L values where ucut < L_min/L
        output.append(Lval)                                          # append thes to output array
        n+=Lval.size                                                 # increase counter

        if (t1-t0) > trunmax:                                        # check that runtime is not too long
            Nvals = n                                                # set Nvals to found values
            if Nvals < 2.:
                output.append(np.zeros(N))                           # if not even 2 values were found return array of 0s
                Nvals  = N                                           # updating Nvals
            n += N-n                                                 # make sure loop ends
    values = np.concatenate(output)[:Nvals]                          # generate output by reformatting
    return values
#-------------------------------------------------------------------------------------------------------------
def appendfitstable(tab1,tab2,newtab='appendfitstable_results.fits'):
    """
    Appending 1 fits table to another.
    It is assumed that the two tables contain the same columns.
    see http://pythonhosted.org/pyfits/users_guide/users_table.html#appending-tables

    Note that columns with object IDs are also added, hence, the be aware of duplicate ids

    Parameters
    ----------
        tab1 : primariy fits table
        tab2 : fits table to append to tab1 (should contain the same columns)

    Returns
    -------
        the name 'newtab' of the created table

    Example
    -------
    import balff_utilities as butil
    tab1   = 'simulatedsamples/dataarraySim_pdistschechter_Ntot1000_k-0p5_Lstar0p5_LJlim0p1_Nobj17.fits'
    tab2   = 'simulatedsamples/dataarraySim_pdistschechter_Ntot2000_k-0p5_Lstar0p5_LJlim0p1_Nobj25.fits'
    newtab = 'simulatedsamples/testname.fits'
    output = butil.appendfitstable(tab1,tab2,newtab=newtab)

    """
    t1     = pyfits.open(tab1)
    t2     = pyfits.open(tab2)

    nrows1 = t1[1].data.shape[0] # counting rows in t1
    nrows2 = t2[1].data.shape[0] # counting rows in t2

    nrows  = nrows1 + nrows2 # total number of rows in the table to be generated
    hdu    = pyfits.new_table(t1[1].columns, nrows=nrows)

    for name in t1[1].columns.names:
        hdu.data.field(name)[nrows1:]=t2[1].data.field(name)

    hdu.writeto(newtab,clobber=False)

    return newtab
#-------------------------------------------------------------------------------------------------------------
def confcontours(xpoints,ypoints,binx=200,biny=200):
    """
    Function estimating confidence contours for a given 2D distribution of points.

    @return: gridsigma, extent

    which can be plotted with for instance
    plt.contour(gridsigma.transpose(),[1,2,3],extent=extent,origin='lower',colors=['r','r','r'],label='contours',zorder=5)
    """
    from fast_kde import fast_kde # used to create confidence curves for contours
    xmin        = np.min(xpoints)
    xmax        = np.max(xpoints)
    ymin        = np.min(ypoints)
    ymax        = np.max(ypoints)
    extent      = [xmax,xmin,ymin,ymax]

    Nval        = binx*biny

    kde_grid    = fast_kde(ypoints,xpoints, gridsize=(binx,biny), weights=None,extents=[ymin,ymax,xmin,xmax])

    binarea     = (xmax-xmin)/binx * (ymax-ymin)/biny
    kde_int     = kde_grid * binarea # ~integrated value in grid
    kde_flat    = np.ravel(kde_int)
    sortindex   = np.argsort(kde_int,axis=None)[::-1]
    gridsigma   = np.zeros((binx,biny))

    sum = 0.0
    for ss in xrange(Nval):
        xx  = np.where(kde_int == kde_flat[sortindex[ss]])
        sum = sum + np.sum(kde_int[xx])
        if (sum < 0.68): gridsigma[xx] = 1.0
        if (sum > 0.68) and (sum < 0.95): gridsigma[xx] = 2.0
        if (sum > 0.95) and (sum < 0.99): gridsigma[xx] = 3.0

    return gridsigma, extent

#-------------------------------------------------------------------------------------------------------------