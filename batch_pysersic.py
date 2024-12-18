import os
from copy import deepcopy
import uuid
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathos.multiprocessing import ProcessingPool
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, MinMaxInterval
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
import astropy.units as uu
from matplotlib.colors import LogNorm
import webbpsf

from pysersic.priors import autoprior
from pysersic import FitSingle
from pysersic.loss import student_t_loss, gaussian_loss
from pysersic.rendering import PixelRenderer
from pysersic.results import plot_residual
import jax.numpy as jnp
from jax.random import PRNGKey

class batch:
    def __init__(self, mosaic, whtmosaic, expmosaic, badpixelmosaic, catalog, scidir, sigdir, psfdir, segdir, outdir, pixel_scale, cutout_size_key, ncores=30, filt='F200W', instrument='NIRISS', generate_cutouts=True):
        self.scidir = scidir
        self.sigdir = sigdir
        self.psfdir = psfdir
        self.segdir = segdir
        self.outdir = outdir
        self.ncores = ncores
        self.pixel_scale = pixel_scale
        self.cutout_size_key = cutout_size_key
        self.mosaic = mosaic
        self.whtmosaic = whtmosaic
        self.expmosaic = expmosaic
        self.badpixelmosaic = badpixelmosaic
        self.catalog = catalog
        if isinstance(self.catalog,str):
            self.catalog = pd.read_csv(self.catalog).reset_index()
        else:
            assert isinstance(self.catalog, pd.DataFrame), "Catalog must be path or pandas dataframe."
            self.catalog = self.catalog.reset_index()
        self.filt=filt
        self.instrument=instrument
        self.generate_cutouts = generate_cutouts

    def fit(self):
        self.prepare_attributes()
        self.prepare_cutouts()
        for sid in self.catalog.sourceid.to_numpy():
            try:
                self.fit_single_object(sid)
            except ValueError:
                print(f"Warning: Failed on fit for sourceid {sid}")
    
    def fit_single_object(self, sid):
        multiple_objects_in_field = False
        if multiple_objects_in_field:
            pass
        else:
            self.fit_single_sersic(sid)

    def fit_single_sersic(self, sid):
        im = fits.getdata(self.scidir+'src{:d}.fits'.format(sid))
        mask = fits.getdata(self.segdir+'src{:d}.fits'.format(sid))
        sig = fits.getdata(self.sigdir+'src{:d}.fits'.format(sid))
        if self.psfdir.endswith('.fits'):
            psf = self.psfdata
        else:
            psf = fits.getdata(self.psfdir+'src{:d}.fits'.format(sid))
        prior = autoprior(image=im, profile_type='sersic', mask=mask, sky_type='none')
        fitter = FitSingle(data=im,rms=sig,mask=mask,psf=psf,prior=prior,loss_func=student_t_loss)
        res = fitter.estimate_posterior(rkey = PRNGKey(1001), method='laplace')
        model_params = res.retrieve_param_quantiles([0.5])
        model_params = {kk:item[0] for (kk,item) in zip(model_params.keys(), list(model_params.values()))}
        model_image = PixelRenderer(im.shape, jnp.array(psf.astype(np.float32))).render_sersic(model_params)
        self.save_single_sersic_products(sid, im, mask, psf, res, model_image)

    def save_single_sersic_products(self, sid, im, mask, psf, res, model_image):
        # FITS block
        mod = np.array(model_image[1]).astype(np.float64)
        resid = im - mod
        prim = fits.PrimaryHDU(data=np.zeros_like(im))
        im1 = fits.ImageHDU(data=im, name='data')
        im2 = fits.ImageHDU(data=mod, name='model')
        im3 = fits.ImageHDU(data=resid, name='residuals')
        fits.HDUList([prim,im1,im2,im3]).writeto(self.outdir+'fits/'+'src{:d}.fits'.format(sid),overwrite=True)
        # corner plots
        fig = res.corner()
        fig.savefig(self.outdir+'corner/'+'src{:d}.png'.format(sid),dpi=300)
        plt.close(fig)
        # residual plots
        fig,ax=plot_residual(im, model_image[1], mask=mask)
        fig.savefig(self.outdir+'residuals/'+'src{:d}.png'.format(sid))
        plt.close(fig)
        # summary df's
        medians = res.retrieve_param_quantiles([0.5])
        df = res.summary().copy()
        df['median'] = [medians[kk][0] for kk in medians.keys()]
        res.summary().to_csv(self.outdir+'posteriors/'+'src{:d}.csv'.format(sid))

    def prepare_attributes(self):
        if isinstance(self.mosaic,str):
            self.ORIG_MOSAIC = fits.open(self.mosaic, memmap=False)
            if self.ORIG_MOSAIC[0].header['BUNIT'] == 'ELECTRONS/S':
                self.mosaic_cps = self.ORIG_MOSAIC
                self.mosaic_cps[0].header['BUNIT'] = 's-1'
                self.mosaic_cps[0].header['EXPTIME'] = 1
            else:
                self.mosaic_cps = self.convert_mosaic_to_cps(self.mosaic.split('.fits')[0]+'_CPS.fits')
            wcs = WCS(self.ORIG_MOSAIC[0].header)
        else:
            raise ValueError("`mosaic` must be str (path file) not type "+str(type(self.mosaic)))
        if isinstance(self.expmosaic,str):
            self.expmosaicpath = self.expmosaic
            self.expmosaic = fits.open(self.expmosaicpath,memmap=False)
        if isinstance(self.badpixelmosaic,str):
            self.badpixelmosaicpath = self.badpixelmosaic
            self.badpixelmosaic = fits.open(self.badpixelmosaicpath,memmap=False)
        if isinstance(self.whtmosaic,str):
            self.ORIG_WHT = fits.open(self.whtmosaic, memmap=False)
            self.sigmaimage = self.get_sigma_image(self.whtmosaic.split('.fits')[0]+'_SIGMA.fits')
        if self.psfdir.endswith('.fits'):
           self.psfdata = fits.getdata(self.psfdir)

    def prepare_cutouts(self):
       if self.generate_cutouts:
           if (self.psfdir is not None) and (not self.psfdir.endswith('.fits')):
               self.psfs = self.get_psfs()
           self.get_cutouts()
           if self.whtmosaic is not None:
               self.get_sigma_cutouts()
       else:
           pass # cutouts already generated
   
    def convert_mosaic_to_cps(self, savename):
        hdr = self.ORIG_MOSAIC[0].header
        image = uu.Quantity(self.ORIG_MOSAIC[0].data,unit=hdr['BUNIT']).to('Jy')/uu.Quantity(hdr['PHOTFNU'],unit='Jy s')
        assert image.unit==(1/uu.s)
        newhdr = deepcopy(self.ORIG_MOSAIC[0].header)
        newhdr['BUNIT'] = 's-1'
        newhdr['EXPTIME'] = 1
        image = fits.HDUList(fits.PrimaryHDU(data=image.value, header=newhdr))
        if savename is not None:
            image.writeto(savename,overwrite=True)
        return image

    def get_sigma_image(self, savename):
        image = deepcopy(self.ORIG_WHT)
        hdr = image[0].header
        image[0].data[np.where(image[0].data==0)]=np.nan
        bunit = image[0].header['BUNIT']
        if bunit != 'UNITLESS':
            pass
            sigma_wht = uu.Quantity(np.sqrt(1/image[0].data), unit=bunit).to('Jy') / uu.Quantity(image[0].header['PHOTFNU'], unit='Jy s')
            wht = 1/(sigma_wht * sigma_wht)
            wht[np.isnan(wht)] = 1e-30*wht.unit # not zero but close!
            assert wht.unit==(uu.s*uu.s), "weight map (1/sig2) must have units of sec^2, please check unit conversions"
            sigmamap = batch.get_cutler24_sigma_image(wht, uu.Quantity(self.mosaic_cps[0].data, unit='s-1'),\
                    uu.Quantity(self.expmosaic[0].data, unit='s'))
            sigmamap[np.where(np.isnan(sigmamap))] = 9999*sigmamap.unit # large value
            assert sigmamap.unit==(1/uu.s)
            assert not np.isnan(sigmamap.value).any(), "Sigma map contains NaNs!"
            assert not (sigmamap==0).any(), "Sigma map contains 0's"
            sighdr = deepcopy(hdr)
            sighdr['BUNIT'] = 's-1'
            sighdr['EXPTIME'] = 1
            sigmaimage = fits.HDUList(fits.PrimaryHDU(data=sigmamap.value, header=sighdr))
        else:
            raise ValueError("Units (BUNIT) of sigma image cannot be `%s`. " % str(bunit))
        if savename is not None:
            sigmaimage.writeto(savename,overwrite=True)
        return sigmaimage

    @staticmethod
    def get_cutler24_sigma_image(whtmap, fluxmap, tmap):
        return np.sqrt((1/whtmap) + (fluxmap/tmap))

    def get_cutouts(self):
        """
        Make cutouts of the cps mosaic (`self.mosaic_cps`).
        """
        cutout_worker = lambda inp: self.get_cutout(self.mosaic_cps,inp[0],position=SkyCoord(ra=inp[1]*uu.deg,dec=inp[2]*uu.deg,\
            frame='icrs'),size=inp[3],badpixmosaic=self.badpixelmosaic,segmpath=inp[4])
        inputs=[(self.scidir+'src{:d}.fits'.format(idv), self.catalog.loc[ii,'ra'], self.catalog.loc[ii,'dec'],\
            self.catalog.loc[ii,self.cutout_size_key], self.segdir+'src{:d}.fits'.format(idv)) for ii,idv in enumerate(self.catalog.sourceid)]
        pool = ProcessingPool(self.ncores)
        _ = pool.map(cutout_worker, inputs)

    def get_sigma_cutouts(self):
        """
        Make cutouts of the cps sigma image (`self.sigmaimage`).
        """
        cutout_worker = lambda inp: self.get_cutout(self.sigmaimage,inp[0],position=SkyCoord(ra=inp[1]*uu.deg,dec=inp[2]*uu.deg,\
        frame='icrs'),size=inp[3],segmpath=None)
        inputs=[(self.sigdir+'src{:d}.fits'.format(idv), self.catalog.loc[ii,'ra'], self.catalog.loc[ii,'dec'],\
                self.catalog.loc[ii,self.cutout_size_key]) for ii,idv in enumerate(self.catalog.sourceid)]
        pool = ProcessingPool(self.ncores)
        _ = pool.map(cutout_worker, inputs)
        #pool.close()

    @staticmethod
    def get_cutout(hdulist, outputpath, position, size, fill_value=0, badpixmosaic=None, segmpath=None):
        cutout = Cutout2D(hdulist[0].data, position=position, size=size, wcs=WCS(hdulist[0].header), fill_value=fill_value)
        if segmpath is not None:
            segm = batch.get_seg_map_wout_central_source(cutout.data)
            if segm is None:
                segm = fits.HDUList(fits.PrimaryHDU(data=np.zeros(cutout.shape), header=cutout.wcs.to_header()))
            else:
                segm = fits.HDUList(fits.PrimaryHDU(data=segm.data, header=cutout.wcs.to_header()))
            if badpixmosaic is not None:
                badpixcutout = Cutout2D(badpixmosaic[0].data, position=position, size=size,\
                        wcs=WCS(badpixmosaic[0].header), fill_value=0)
                segm[0].data += badpixcutout.data
                sel = (segm[0].data>0)
                segm[0].data[sel] = 1
                segm[0].data[~sel] = 0
            segm.writeto(segmpath, overwrite=True)
        output = fits.HDUList(fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header()))
        output.writeto(outputpath, overwrite=True)

    @staticmethod
    def get_seg_map_wout_central_source(data, nsigma=2):
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        thresh = detect_threshold(data, nsigma=nsigma, sigma_clip=sigma_clip)
        segm = detect_sources(data, threshold=thresh, npixels=5)
        if segm is not None:
            source_in_middle = segm.data[data.shape[0]//2, data.shape[1]//2]
            if source_in_middle == 0:
                print("Warning: could not mask data because source not at center")
            else:
                segm.remove_labels(labels=source_in_middle)
        return segm

    def get_psfs(self):
        if self.instrument == 'NIRISS':
            ins = webbpsf.NIRISS()
        else:
            raise ValueError("Only NIRISS is currently supported for `instrument`.")
        ins.filter = self.filt
        worker = lambda inp: ins.calc_psf(inp[0],fov_pixels=inp[1])
        inputs = [(self.psfdir+'src{:d}_psf.fits'.format(idv), self.catalog.loc[ii,self.cutout_size_key]) for ii,idv in enumerate(self.catalog.sourceid)]
        pool = ProcessingPool(self.ncores)
        psfs = pool.map(worker, inputs)
        return psfs
