import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import astropy.units as u
import astropy.constants as c
import itertools
import hasasia
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky
import hasasia.anisotropy as haniso
import pickle
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [5,3]
mpl.rcParams['text.usetex'] = False

savedir = '/home/oliverda/novus/anisotropy/plots'
datadir = '/home/oliverda/novus/anisotropy/Hasasia_Spectra/'
datadir_forecast = '/home/oliverda/novus/anisotropy/gpta_psrs_40yrs_v2/'

spectra_15yr = []

for filename in os.listdir(datadir):
    if filename.endswith('.has'):
        with open(datadir+filename, 'rb') as file:
            spec_15yr = pickle.load(file)
            spectra_15yr.append(spec_15yr)
            
NSIDE = 8
NPIX = hp.nside2npix(NSIDE)
IPIX = np.arange(NPIX)
theta_gw, phi_gw = hp.pix2ang(nside=NSIDE,ipix=IPIX)

ASM=haniso.Anisotropy(spectra_15yr,theta_gw, phi_gw, NPIX=NPIX)

Tspan_15 = max(
    (np.max(spectra.toas) - np.min(spectra.toas))
    for spectra in spectra_15yr
)

freqs = np.logspace(np.log10(spectra_15yr[0].freqs[0]),np.log10(spectra_15yr[0].freqs[-1]),len(spectra_15yr[0].freqs))

freq_idx_15 = int(np.argmin(np.abs(freqs - 5.08e-9)))


S_eff_aniso_15 = (ASM.S_eff_aniso_noprime /Tspan_15)**(-1/2)
h_c_aniso_15 = np.sqrt(freqs * S_eff_aniso_15)

# Plotting the effective strain and characteristic strain PSD
plt.loglog(spectra_15yr[0].freqs, S_eff_aniso_15, color='purple')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Effective Strain Noise PSD, $S_{\rm eff}$')
plt.savefig(savedir + 'ng15yr_S_eff.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

plt.loglog(spectra_15yr[0].freqs, h_c_aniso_15, color='purple')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.savefig(savedir + 'ng15yr_h_c.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

spectra_forecast = []

for filename in os.listdir(datadir_forecast):
    if filename.endswith('.pkl'):
        file_path = os.path.join(datadir_forecast, filename)
        with open(file_path, 'rb') as file:
            spec_forecast = pickle.load(file)
            spectra_forecast.append(spec_forecast)
            
freqs = np.logspace(np.log10(spectra_15yr[0].freqs[0]),np.log10(spectra_15yr[0].freqs[-1]),len(spectra_15yr[0].freqs))

spectra = []
for p in spectra_forecast:
    sp = hsen.Spectrum(p, freqs=freqs)
    sp.NcalInv
    spectra.append(sp)
    
ASM=haniso.Anisotropy(spectra,theta_gw, phi_gw, NPIX=NPIX)

Tspan_forecast = max(
    (np.max(spectra.toas) - np.min(spectra.toas))
    for spectra in spectra_forecast
)

freq_idx_forecast = int(np.argmin(np.abs(freqs - 5.08e-9)))

S_eff_aniso_forecast = (ASM.S_eff_aniso_noprime/Tspan_forecast)**(-1/2)
h_c_aniso_forecast =  np.sqrt(freqs * S_eff_aniso_forecast)

plt.loglog(freqs,S_eff_aniso_forecast, color='black')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Effective Strain Noise PSD, $S_{\rm eff}$')
plt.gca().set_yticklabels([])
plt.savefig(savedir + '40yr_S_eff.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

plt.loglog(freqs,h_c_aniso_forecast, color='black')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.gca().set_yticklabels([])
plt.savefig(savedir + '40yr_h_c.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

plt.loglog(spectra_15yr[0].freqs,S_eff_aniso_15, color='purple', label='NG 15yr')
plt.loglog(freqs,S_eff_aniso_forecast, color='black', label='40yr PTA (Simulated)')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Effective Strain Noise PSD, $S_{\rm eff}$')
plt.gca().set_yticklabels([])
plt.legend()
plt.savefig(savedir + 'Comparison_S_eff.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

plt.loglog(spectra_15yr[0].freqs,h_c_aniso_15, color='purple', label='NG 15yr')
plt.loglog(freqs,h_c_aniso_forecast, color='black', label='40yr PTA (Simulated)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.gca().set_yticklabels([])
plt.legend()
plt.savefig(savedir + 'Comparison_h_c.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

plt.loglog(spectra_15yr[0].freqs,S_eff_aniso_15, color='purple', label='NG 15yr')
plt.loglog(freqs,S_eff_aniso_forecast, color='black', label='40yr PTA (Simulated)')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Effective Strain Noise PSD, $S_{\rm eff}$')
plt.legend()
plt.savefig(savedir + 'Comparison_S_eff_labels.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

plt.loglog(spectra_15yr[0].freqs,h_c_aniso_15, color='purple', label='NG 15yr')
plt.loglog(freqs,h_c_aniso_forecast, color='black', label='40yr PTA (Simulated)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.legend()
plt.savefig(savedir + 'Comparison_h_c_labels.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()
ASM=haniso.Anisotropy(spectra_15yr,theta_gw, phi_gw, NPIX=NPIX)

hp.mollview(ASM.R_IJ.sum(axis=0), title="Combined Pulsar Response $R_{IJ}$", cmap='Purples', rot=(180,0,0))
hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
                            edgecolors='black', s=150)
plt.savefig(savedir + 'ng15yr_R_IJ.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

# max_value = np.max(ASM.h_c_aniso)
# max_index = np.argmax(ASM.h_c_aniso)
# max_freq = spectra_15yr[0].freqs[max_index]

# min_value = np.min(ASM.h_c_aniso)
# min_index = np.argmin(ASM.h_c_aniso)
# min_freq = spectra_15yr[0].freqs[min_index]

# median_value = np.median(ASM.h_c_aniso)
# median_index = (np.abs(ASM.h_c_aniso - median_value)).argmin()
# median_freq = spectra_15yr[0].freqs[median_index] 

# # Lists of frequencies, indices, and labels
# final_freqs = [max_freq, min_freq, median_freq]
# final_freqs_idx = [max_index, min_index, median_index]
# labels = ['max', 'min', 'med']

final_freqs_idx = [freq_idx_15]
final_freqs = [float(spectra_15yr[0].freqs[freq_idx_15])]
labels = ['min']
print(f'Final Frequencies: {final_freqs}')
print(f'final frequency type: {type(final_freqs)}')
print(f'Final Frequency Indices: {final_freqs_idx}')
print(f'Final Frequency Indices type: {type(final_freqs_idx)}')
print(f'Labels: {labels}')
print(f'Labels type: {type(labels)}')

# Mkkp = ASM.M_kkp 

# # Assume Mkkp has shape (N_freqs, N_pix, N_pix)
# fi = 0  # or any frequency index
# fixed_kp = 0  # Randomly choose a pixel index for k'
# fixed_k  = 0  # Randomly choose a pixel index for k

# map_k  = Mkkp[fi, :, fixed_kp]  # M(f, k, k'=fixed)
# map_kp = Mkkp[fi, fixed_k, :]   # M(f, k=fixed, k')

# hp.mollview(map_k, title=f"M(f, k, k'={fixed_kp})", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '15yr_Mkkp_fixed_kp.pdf', dpi=300, bbox_inches='tight', facecolor='w')
# hp.mollview(map_kp, title=f"M(f, k={fixed_k}, k')", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '15yr_Mkkp_fixed_k.pdf', dpi=300, bbox_inches='tight', facecolor='w')

# M_kk_scale = np.sqrt(Mkkp[fi, :, fixed_kp])/Tspan_15
# M_kkp_scale = np.sqrt(Mkkp[fi, fixed_k, :])/Tspan_15

# M_kk_norm = (M_kk_scale - np.min(M_kk_scale)) / (np.max(M_kk_scale) - np.min(M_kk_scale))
# M_kkp_norm = (M_kkp_scale - np.min(M_kkp_scale)) / (np.max(M_kkp_scale) - np.min(M_kkp_scale))

# hp.mollview(M_kk_norm, title=f"M(f, k, k'={fixed_kp})", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '15yr_Mkkp_scaled_fixed_kp.pdf', dpi=300, bbox_inches='tight', facecolor='w')
# hp.mollview(M_kkp_norm, title=f"M(f, k={fixed_k}, k')", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '15yr_Mkkp_scaled_fixed_k.pdf', dpi=300, bbox_inches='tight', facecolor='w')

plt.rc('text', usetex=False)
plt.rcParams.update({'font.size': 22})

def plot_mollview(data, title, filename, cbar_label):
    hp.mollview(data, title=title, cmap='Purples', rot=(180, 0, 0), cbar=None)
    hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red',
                            edgecolors='black', linewidths=0.25, s=150, alpha=1)
    hp.graticule()

    fig = plt.gcf()
    ax = plt.gca()
    
    # Get the image and add a colorbar
    image = ax.get_images()[0]
    cbar = fig.colorbar(image, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
    cbar.set_label(cbar_label, labelpad=10)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='w')
    plt.close()

# Iterate through the frequency, index, and label simultaneously
for freq, idx, label in zip(final_freqs, final_freqs_idx, labels):
    # # Plot h_c_aniso_pixel without normalization
    # h_c_pixel = ASM.h_c_aniso_pixel(idx)
    # plot_mollview(h_c_pixel, title=f"f = {freq * 1e9:.2f} nHz", filename=f'ng15yr_h_c_{label}.pdf', cbar_label=r'$h_c$')

    # # Plot h_c_aniso_pixel with normalization
    # h_c_pixel_norm = (h_c_pixel - np.min(h_c_pixel)) / (np.max(h_c_pixel) - np.min(h_c_pixel))
    # plot_mollview(h_c_pixel_norm, title=f"f = {freq * 1e9:.2f} nHz", filename=f'ng15yr_h_c_{label}_scale.pdf', cbar_label=r'$h_c$')

    # Plot M_kk_pixel without normalization
    M_kk_pixel = np.sqrt(ASM.M_kk[idx, :]) / Tspan_15
    plot_mollview(M_kk_pixel, title=f"f = {freq * 1e9:.2f} nHz", filename=f'ng15yr_Mkk_{label}.pdf', cbar_label=r'$\sqrt{\mathcal{M}}$')

    # Plot M_kk_pixel with normalization
    M_kk_pixel_norm = (M_kk_pixel - np.min(M_kk_pixel)) / (np.max(M_kk_pixel) - np.min(M_kk_pixel))
    plot_mollview(M_kk_pixel_norm, title=f"f = {freq * 1e9:.2f} nHz", filename=f'ng15yr_Mkk_{label}_scale.pdf', cbar_label=r'$\sqrt{\mathcal{M}}$')
    
    
ASM=haniso.Anisotropy(spectra,theta_gw, phi_gw, NPIX=NPIX)

# Mkkp = ASM.M_kkp 

# # Assume Mkkp has shape (N_freqs, N_pix, N_pix)
# fi = 0  # or any frequency index
# fixed_kp = 0  # Randomly choose a pixel index for k'
# fixed_k  = 0  # Randomly choose a pixel index for k

# map_k  = Mkkp[fi, :, fixed_kp]  # M(f, k, k'=fixed)
# map_kp = Mkkp[fi, fixed_k, :]   # M(f, k=fixed, k')

# hp.mollview(map_k, title=f"M(f, k, k'={fixed_kp})", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '40yr_Mkkp_fixed_kp.pdf', dpi=300, bbox_inches='tight', facecolor='w')
# hp.mollview(map_kp, title=f"M(f, k={fixed_k}, k')", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '40yr_Mkkp_fixed_k.pdf', dpi=300, bbox_inches='tight', facecolor='w')

# M_kk_scale = np.sqrt(Mkkp[fi, :, fixed_kp])/Tspan_15
# M_kkp_scale = np.sqrt(Mkkp[fi, fixed_k, :])/Tspan_15

# M_kk_norm = (M_kk_scale - np.min(M_kk_scale)) / (np.max(M_kk_scale) - np.min(M_kk_scale))
# M_kkp_norm = (M_kkp_scale - np.min(M_kkp_scale)) / (np.max(M_kkp_scale) - np.min(M_kkp_scale))

# hp.mollview(M_kk_norm, title=f"M(f, k, k'={fixed_kp})", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '40yr_Mkkp_scaled_fixed_kp.pdf', dpi=300, bbox_inches='tight', facecolor='w')
# hp.mollview(M_kkp_norm, title=f"M(f, k={fixed_k}, k')", cmap='Purples_r',
#             rot=(180, 0, 0))
# hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
#                             edgecolors='black', s=150, rot=(180,0,0))
# plt.savefig(savedir + '40yr_Mkkp_scaled_fixed_k.pdf', dpi=300, bbox_inches='tight', facecolor='w')

hp.mollview(ASM.R_IJ.sum(axis=0), title="Combined Pulsar Response $R_{IJ}$", cmap='Purples', rot=(180,0,0))
hp.visufunc.projscatter(ASM.thetas, ASM.phis, marker='*', color='red', 
                            edgecolors='black', s=150)
plt.savefig(savedir + '40yr_R_IJ.pdf', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

# max_value = np.max(ASM.h_c_aniso)
# max_index = np.argmax(ASM.h_c_aniso)
# max_freq = freqs[max_index]  

# min_value = np.min(ASM.h_c_aniso)
# min_index = np.argmin(ASM.h_c_aniso)
# min_freq = freqs[min_index] 

# median_value = np.median(ASM.h_c_aniso)
# median_index = (np.abs(ASM.h_c_aniso - median_value)).argmin()
# median_freq = freqs[median_index]  

# # Lists of frequencies, indices, and labels
# final_freqs = [max_freq, min_freq, median_freq]
# final_freqs_idx = [max_index, min_index, median_index]
# labels = ['max', 'min', 'med']

final_freqs_idx = [freq_idx_forecast]
final_freqs = [float(freqs[freq_idx_forecast])]
labels = ['min']
print(f'Final Frequencies: {final_freqs}')
print(f'final frequency type: {type(final_freqs)}')
print(f'Final Frequency Indices: {final_freqs_idx}')
print(f'Final Frequency Indices type: {type(final_freqs_idx)}')
print(f'Labels: {labels}')
print(f'Labels type: {type(labels)}')

# Iterate through the frequency, index, and label simultaneously
for freq, idx, label in zip(final_freqs, final_freqs_idx, labels):
    # # Plot h_c_aniso_pixel without normalization
    # h_c_pixel = ASM.h_c_aniso_pixel(idx)
    # plot_mollview(h_c_pixel, title=f"f = {freq * 1e9:.2f} nHz", filename=f'40yr_h_c_{label}.pdf', cbar_label=r'$h_c$')

    # # Plot h_c_aniso_pixel with normalization
    # h_c_pixel_norm = (h_c_pixel - np.min(h_c_pixel)) / (np.max(h_c_pixel) - np.min(h_c_pixel))
    # plot_mollview(h_c_pixel_norm, title=f"f = {freq * 1e9:.2f} nHz", filename=f'40yr_h_c_{label}_scale.pdf', cbar_label=r'$h_c$')

    # Plot M_kk_pixel without normalization
    M_kk_pixel = np.sqrt(ASM.M_kk[idx, :]) / Tspan_forecast
    plot_mollview(M_kk_pixel, title=f"f = {freq * 1e9:.2f} nHz", filename=f'40yr_Mkk_{label}.pdf', cbar_label=r'$\sqrt{\mathcal{M}}$')

    # Plot M_kk_pixel with normalization
    M_kk_pixel_norm = (M_kk_pixel - np.min(M_kk_pixel)) / (np.max(M_kk_pixel) - np.min(M_kk_pixel))
    plot_mollview(M_kk_pixel_norm, title=f"f = {freq * 1e9:.2f} nHz", filename=f'40yr_Mkk_{label}_scale.pdf', cbar_label=r'$\sqrt{\mathcal{M}}$')


