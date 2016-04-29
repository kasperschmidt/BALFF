#-------------------------------------------------------------------------------------------------------------
import pdb
import pyfits
import balff_createDataArray as bcda
#-------------------------------------------------------------------------------------------------------------
def write_fitsfile(objname,field,Lobj,Lobjerr,Lfieldlim,Lfieldlimerr,Llimsig,
                   outputname='./balff_data/objects_info.fits',verbose=True):
    """
    Create a (minimal) fits file containg data array of objects found in the fields
    described in ./balff_data/fields_info.txt on the format expected by balff_mpd.py

    --- INPUT ---
    objname          Name of objects (list of strings)
    field            Field the object was found in (list of strings)
    Lobj             Absolute luminoisity of object. Can be obtained from observed apparent magnitude with
                     balff_utilities.Mabs2L( balff_utilities.magapp2abs(magapp,zobj,Av=Avval,band=magband) )
    Lobjerr          Uncertainty on absolute Lobj
    Lfieldlim        Absolute luminosity corresponding to the limiting magnitude for the filed provided in
                     ./balff_data/fields_info.txt
    Lfieldlimerr     Uncertainty on Lfieldlim
    Llimsig          Number of sigmas the limitng luminoisity Lfieldlim corresponds to

    --- EXAMPLE OF USE ---
    see bcda.test_write_fitsfile_onBoRG() below

    """
    if verbose: print ' - Setting up output file '
    col1     = pyfits.Column(name='OBJNAME'      , format='A30', array=objname)
    col2     = pyfits.Column(name='FIELD'        , format='A30', array=field)
    col3     = pyfits.Column(name='LOBJ'         , format='D'  , array=Lobj)
    col4     = pyfits.Column(name='LOBJERR'      , format='D'  , array=Lobjerr)
    col5     = pyfits.Column(name='LFIELDLIM'    , format='D'  , array=Lfieldlim)
    col6     = pyfits.Column(name='LFIELDLIMERR' , format='D'  , array=Lfieldlimerr)
    cols     = pyfits.ColDefs([col1 ,col2 ,col3 ,col4 ,col5 ,col6])
    tbhdu    = pyfits.new_table(cols)          # creating table header

    # writing hdrkeys:   '---KEY--',                  '----------------MAX LENGTH COMMENT-------------'
    tbhdu.header.append(('LLIMSIG ',Llimsig           ,'Sigmas field limiting lum. corresponds to'),end=True)

    if verbose: print ' - Writing simulated data to fits table '+outputname
    hdu      = pyfits.PrimaryHDU()             # creating primary (minimal) header
    thdulist = pyfits.HDUList([hdu, tbhdu])    # combine primary and table header to hdulist
    thdulist.writeto(outputname,clobber=True)  # write fits file (clobber=True overwrites excisting file)
    return outputname
#-------------------------------------------------------------------------------------------------------------
def test_write_fitsfile_onBoRG(inputfits,verbose=True):
    """
    Testing bcda.write_fitsfile() on and existing BoRG data array

    --- EXAMPLE OF USE ---

    bcda.test_write_fitsfile_onBoRG('./balff_data/dataarray_BoRG13sample130924_5sig.fits')

    """
    Bdata = pyfits.open(inputfits)[1].data

    objname      = Bdata['OBJNAME']
    field        = Bdata['FIELD']
    Lobj         = Bdata['LJ']
    Lobjerr      = Bdata['LJERR']
    Lfieldlim    = Bdata['LJLIM']
    Lfieldlimerr = Bdata['LJLIMERR']
    Llimsig      = pyfits.open(inputfits)[1].header['LJLIMSIG']

    output = bcda.write_fitsfile(objname,field,Lobj,Lobjerr,Lfieldlim,Lfieldlimerr,Llimsig,
                                 outputname='./balff_data/objects_info.fits',verbose=verbose)

    return output
#-------------------------------------------------------------------------------------------------------------



