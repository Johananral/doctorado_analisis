from astrodendro import Dendrogram
from astropy.io import fits
from astropy import units as u
import numpy as np
import os 
from astropy.stats import mad_std
from VLA import Image
import matplotlib.pyplot as plt

### DENDROGRAMA ###

#SELECCIONAR WEIGHT: .UNI, .ROBUST0, .ROBUST0.5
weight = 'ROBUST0.5'
n_sigma = 5

data_path = './FITS'
inputname = 'CorAus.X.ALL.'+ weight +'.fits'
imagefile = os.path.join(data_path,inputname)


dendrogram_path = './DENDROGRAM'
outname = inputname.replace('.fits','.dendro.'+str(n_sigma)+'sigma.fits')
outfile = os.path.join(dendrogram_path,outname)

img = Image(imagefile)
image = img.data
#image = fits.getdata(imagefile).squeeze()
header = img.header
beam = img.get_beam()

#Corrección del beam FWHM = 2.355sigma
n_beam = 2.355*1
beam.ma = n_beam*beam.major.to(u.arcsec)/2.355
beam.min = n_beam*beam.minor.to(u.arcsec)/2.355

pix_per_beam = ((1.442*np.pi/4)*header['BMAJ']*header['BMIN'])/(header['CDELT2']**2)
print('There are {:.3f} pixels in BMAJ'.format(header['BMAJ']/header['CDELT2']))
print('There are {:.3f} pixels in BMIN'.format(header['BMIN']/header['CDELT2']))
print('There are {:.3f} pixels in one beam area'.format(pix_per_beam))

sigma_noise = mad_std(image, ignore_nan=True) 
print('The noise in the image is {:.3e} Jy/beam'.format(sigma_noise))
min_value = n_sigma*sigma_noise
min_delta = n_sigma*sigma_noise
min_npix = int(pix_per_beam)

#Calcular y guardar dendrograma
d = Dendrogram.compute(image, min_value=min_value, min_delta=min_delta, min_npix=min_npix, verbose=True)
d.save_to(outfile)

#Visualizar dendrograma
#v = d.viewer()
#v.show()

#Contorno del mínimo nivel de jerarquía del dendrograma
#p = d.plotter()
#fig = plt.figure(figsize=(8, 8))
#plt.xlim(0, header['NAXIS1'])
#plt.ylim(0, header['NAXIS2'])
#ax = fig.add_subplot(1, 1, 1)
#p.plot_contour(ax, color='black')
#plt.show()

###########################################################

### FICHEROS ###

#Cargar dendrograma

#SELECCIONAR WEIGHT: .UNI, .ROBUST0, .ROBUST0.5
#weight = 'ROBUST0.5'
#SELECCIONAR N_SIGMA: 3, 5
#n_sigma = 5
dendrofile = './DENDROGRAM/CorAus.X.ALL.'+weight+'.dendro.'+str(n_sigma)+'sigma.fits'
d = Dendrogram.load_from(dendrofile)

#Dibujar regiones
outfile_regions = './REGIONS/regions.'+weight+'.'+str(n_sigma)+'sigma.crtf'
output = open(outfile_regions,'w')
output.write('#CRTFv0 CASA Region Text Format version 0 \n')
for leaf in d.leaves:
    #Coordenadas del baricentro
    (cy_bari, cx_bari) = img.center_of_mass(leaf.get_mask(),coords=False)
    print(f'ellipse [[{int(cx_bari)}pix,{int(cy_bari)}pix],[{beam.ma.value:.3f}arcsec, {beam.min.value:.3f}arcsec], {beam.pa.value:.3f}deg] color=yellow, linewidth=2 \n')
    output.write(f'ellipse [[{int(cx_bari)}pix,{int(cy_bari)}pix],[{beam.ma.value:.3f}arcsec, {beam.min.value:.3f}arcsec], {beam.pa.value:.3f}deg] color=magenta, linewidth=2 \n')
output.close()

#Dibujar regiones con su respectivo índice
outfile_regions = './REGIONS/regions-index.'+weight+'.'+str(n_sigma)+'sigma.crtf'
output = open(outfile_regions,'w')
output.write('#CRTFv0 CASA Region Text Format version 0 \n')
for index, leaf in enumerate(d.leaves):
    #Coordenadas del baricentro
    (cy_bari, cx_bari) = img.center_of_mass(leaf.get_mask(),coords=False)
    print(f'ellipse [[{int(cx_bari)}pix,{int(cy_bari)}pix],[{beam.ma.value:.3f}arcsec, {beam.min.value:.3f}arcsec], {beam.pa.value:.3f}deg] color=yellow, linewidth=2 \n')
    output.write(f'ellipse [[{int(cx_bari)}pix,{int(cy_bari)}pix],[{beam.ma.value:.3f}arcsec, {beam.min.value:.3f}arcsec], {beam.pa.value:.3f}deg] color=magenta, linewidth=2 \n')
    output.write(f'text[[{int(cx_bari)}pix,{int(cy_bari)}pix],\'' + str(index) + '\'], color=10161a, fontsize=10 \n')
    print(f'text[[{int(cx_bari)}pix,{int(cy_bari)}pix],\'' + str(index) + '\'], color=10161a, fontsize=20 \n')
output.close()


#Arreglo de regiones para imfit-loop
regions = []
for leaf in d.leaves:
    (cy_bari, cx_bari) = img.center_of_mass(leaf.get_mask(),coords=False)
    region_str = f"ellipse [[{cx_bari:.1f}pix,{cy_bari:.1f}pix],[{beam.ma.value:.3f}arcsec, {beam.min.value:.3f}arcsec], {beam.pa.value:.3f}deg]"
    regions.append(region_str)
regions_array = np.array(regions)
print(regions_array)
np.savez('regions.'+weight+'.'+str(n_sigma)+'sigma.npz', regions=regions_array)

#Ajustes en imfit-estimates
outfile_estimates = './ESTIMATES/CorAus.X.ALL'+weight+'.'+str(n_sigma)+'sigma.estimates'
output = open(outfile_estimates,'w')
output.write('#peak, x, y, bmaj, bmin, bpa')
for leaf in d.leaves:
    #Pico de emisión y sus coordenadas
    ((cy_peak,cx_peak), peak) = leaf.get_peak()
    #Coordenadas del baricentro
    (cy_bari, cx_bari) = img.center_of_mass(leaf.get_mask(),coords=False)
    #print(f'{int(cx_bari)},{int(cy_bari)},{peak}, {beam.ma:.3f}, {beam.min:.3f}, {beam.pa:.3f}\n')
    output.write(f'{peak}, {int(cx_bari)}, {int(cy_bari)}, {beam.ma:.3f}, {beam.min:.3f}, {beam.pa:.3f}\n'.replace(' ',''))
output.close()