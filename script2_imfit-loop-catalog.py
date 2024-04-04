### IMFIT LOOP ###

### DENTRO DE CASA ###
import numpy as np

#SELECCIONAR WEIGHT: .UNI, .ROBUST0, .ROBUST0.5
weight = 'ROBUST0.5'
#SELECCIONAR N_SIGMA: 3, 5
n_sigma = 5

regions = np.load('regions.'+weight+'.'+str(n_sigma)+'sigma.npz')['regions']
n_fuentes = len(regions)
for i in range(n_fuentes):    
    fit_i = imfit(imagename ='./FITS/CorAus.X.ALL.'+weight+'.fits', region = regions[i])
    #fit_numpy_i = convert_numpy(fit_i)
    np.savez('./npz/CorAus.X.ALL.'+weight+'.'+str(n_sigma)+'.fit{}.npz'.format(i), fit_i=fit_i)
    
###############################################################################################
    
### FUERA DE CASA ###

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord

#SELECCIONAR WEIGHT: .UNI, .ROBUST0, .ROBUST0.5
weight = 'ROBUST0.5'
#SELECCIONAR N_SIGMA: 3, 5
n_sigma = 5

outfile = './CATALOGS/CorAus.X.ALL.'+weight+'.'+str(n_sigma)+'sigma.csv'
outcolumns = ['Fuente', 'RA(J2000)', 'RA_err','Decl(J2000)', 'Decl_err',
              'Peak(mJy/beam)', 'Peak_err(mJy/beam)',
              'Flux(mJy)','Flux_err(mJy)',
              'Majax_dec(arcsec)','Majax_err_dec(arcsec)',
              'Minax_dec(arcsec)','Minax_err_dec(arcsec)',
              'PA_dec(deg)','PA_err_dec(deg)','Freq(GHz)']
fit_loop_info = pd.DataFrame(columns=outcolumns)

regions = np.load('regions.'+weight+'.'+str(n_sigma)+'sigma.npz')['regions']
n_fuentes = len(regions)
for i in range(n_fuentes):
    data_i = np.load('./npz/CorAus.X.ALL.'+weight+'.'+str(n_sigma)+'.fit{}.npz'.format(i), allow_pickle=True)
    fit_i = data_i['fit_i']

    if 'component0' in fit_i.item()['results']:
        flux = fit_i.item()['results']['component0']['flux']['value'][0]
        flux_formatted = '{:.3f}'.format(1e3*flux)
        
        flux_err = fit_i.item()['results']['component0']['flux']['error'][0]
        flux_err_formatted = '{:.3f}'.format(1e3*flux_err)    
  
        peak = fit_i.item()['results']['component0']['peak']['value']
        peak_formatted = '{:.3f}'.format(1e3*peak)
        
        peak_err = fit_i.item()['results']['component0']['peak']['error']
        peak_err_formatted = '{:.3f}'.format(1e3*peak_err)
    
        freq = fit_i.item()['results']['component0']['spectrum']['frequency']['m0']['value']
        freq_formatted = '{:.2f}'.format(freq)
        
        RA_rad = fit_i.item()['results']['component0']['shape']['direction']['m0']['value']
        Decl_rad = fit_i.item()['results']['component0']['shape']['direction']['m1']['value']
        coord = SkyCoord(ra=RA_rad, dec=Decl_rad, unit='radian')
        RA = coord.ra.to_string(unit='hour', sep=':')
        Decl = coord.dec.to_string(unit='deg', sep=':')
    
        RA_err_arcsec = fit_i.item()['results']['component0']['shape']['direction']['error']['longitude']['value']
        Decl_err_arcsec = fit_i.item()['results']['component0']['shape']['direction']['error']['latitude']['value'] #no transformar
        coord = SkyCoord(ra=RA_err_arcsec, dec=Decl_err_arcsec, unit='arcsec')
        RA_err = coord.ra.to_string(unit='hour', sep=':', precision=5)
        RA_err_formatted = RA_err.split(':')[2]
        Decl_err_arcsec_formatted = '{:.5f}'.format(Decl_err_arcsec)
        
    else:
        flux_formatted = '--'
        flux_err_formatted = '--'
        peak_formatted = '--'
        peak_err_formatted = '--'
        freq_formatted = '--'
        RA = '--'
        Decl = '--'
        RA_err_formatted = '--'
        Decl_err_arcsec_formatted = '--'
    
    #print(flux_formatted,flux_err_formatted,peak_formatted,peak_err_formatted,freq_formatted,RA,Decl,RA_err_formatted,Decl_err_arcsec_formatted)
    
    if 'deconvolved' in fit_i.item():
        
        if 'majoraxis' in fit_i.item()['deconvolved']['component0']['shape']:
            majax_dec = fit_i.item()['deconvolved']['component0']['shape']['majoraxis']['value']
            majax_dec_formatted = '{:.3f}'.format(majax_dec)
        #else:
        #    majax_dec_formatted = '--'
        if 'majoraxiserror' in fit_i.item()['deconvolved']['component0']['shape']:
            majax_dec_err = fit_i.item()['deconvolved']['component0']['shape']['majoraxiserror']['value']
            majax_dec_err_formatted = '{:.3f}'.format(majax_dec_err)
        #else:
        #    majax_dec_err_formatted = '--'
        if 'minoraxis' in fit_i.item()['deconvolved']['component0']['shape']:
            minax_dec = fit_i.item()['deconvolved']['component0']['shape']['minoraxis']['value']
            minax_dec_formatted = '{:.3f}'.format(minax_dec)
        #else:
        #    minax_dec_formatted = '--'
        if 'minoraxiserror' in fit_i.item()['deconvolved']['component0']['shape']:
            minax_dec_err = fit_i.item()['deconvolved']['component0']['shape']['minoraxiserror']['value']
            minax_dec_err_formatted = '{:.3f}'.format(minax_dec_err)
        #else:
        #    minax_dec_err_formatted = '--'
        if 'positionangle' in fit_i.item()['deconvolved']['component0']['shape']:
            pa_dec = fit_i.item()['deconvolved']['component0']['shape']['positionangle']['value']
            pa_dec_formatted = '{:.2f}'.format(pa_dec)
        #else:
        #    pa_dec_formatted = '--'
        if 'positionangleerror' in fit_i.item()['deconvolved']['component0']['shape']:
            pa_err_dec = fit_i.item()['deconvolved']['component0']['shape']['positionangleerror']['value']
            pa_err_dec_formatted = '{:.2f}'.format(pa_err_dec)
        #else:
        #    pa_err_dec_formatted = '--'
    
    else:
        majax_dec_formatted = '--'
        majax_dec_err_formatted = '--'
        minax_dec_formatted = '--'
        minax_dec_err_formatted = '--'
        pa_dec_formatted = '--'
        pa_err_dec_formatted = '--'
    
    #print(flux_formatted,flux_err_formatted,peak_formatted,peak_err_formatted,freq_formatted,RA,Decl,RA_err_formatted,Decl_err_arcsec_formatted,majax_dec_formatted,majax_dec_err_formatted,minax_dec_formatted,minax_dec_err_formatted,pa_dec_formatted,pa_err_dec_formatted)
    
    fit_loop_info = pd.concat([fit_loop_info, pd.DataFrame({'Fuente': [i],
                                                        'RA(J2000)': [RA], 'RA_err': [RA_err_formatted],
                                                        'Decl(J2000)': [Decl], 'Decl_err': [Decl_err_arcsec_formatted],
                                                        'Peak(mJy/beam)': [peak_formatted], 'Peak_err(mJy/beam)': [peak_err_formatted],
                                                        'Flux(mJy)': [flux_formatted], 'Flux_err(mJy)': [flux_err_formatted],
                                                        'Majax_dec(arcsec)': [majax_dec_formatted], 'Majax_err_dec(arcsec)': [majax_dec_err_formatted],
                                                        'Minax_dec(arcsec)': [minax_dec_formatted], 'Minax_err_dec(arcsec)': [minax_dec_err_formatted],
                                                        'PA_dec(deg)': [pa_dec_formatted], 'PA_err_dec(deg)': [pa_err_dec_formatted],
                                                        'Freq(GHz)': [freq_formatted]})], ignore_index=True)
    
print(fit_loop_info)

fit_loop_info.to_csv(outfile, index=False)
