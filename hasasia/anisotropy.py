# -*- coding: utf-8 -*-
from __future__ import print_function
"""Main module."""
import numpy as np
import scipy.optimize as sopt
import scipy.special as ssp
import scipy.integrate as si
import scipy.stats as ss
import healpy as hp
import astropy.units as u
import astropy.constants as c
import itertools as it
from .sensitivity import GWBSensitivityCurve, DeterSensitivityCurve,  resid_response, get_dt, get_Tspan
from .utils import strain_and_chirp_mass_to_luminosity_distance

__all__ = ['SkySensitivity',
           'h_circ',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600


class Anisotropy(GWBSensitivityCurve):
    r'''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of :math:`\hat{n}=-\hat{k}`.

    Parameters
    ----------
    theta_gw : list, array
        Gravitational wave source sky location colatitude at which to
        calculate sky map.

    phi_gw : list, array
        Gravitational wave source sky location longitude at which to
        calculate sky map.

    pulsar_term : bool, str, optional [True, False, 'explicit']
        Flag for including the pulsar term in sky map sensitivity. True
        includes an idealized factor of two from Equation (36) of `[1]`_.
        The `'explicit'` flag turns on an explicit calculation of
        pulsar terms using pulsar distances. (This option takes
        considerably more computational resources.)

        .. _[1]: https://arxiv.org/abs/1907.04341

    pol: str, optional ['gr','scalar-trans','scalar-long','vector-long']
        Polarization of gravitational waves to be used in pulsar antenna
        patterns. Only one can be used at a time.
    '''
    def __init__(self, spectra, theta_gw, phi_gw,
                 pulsar_term=False, pol='gr', iota=None, psi=None, NPIX=8):
        super().__init__(spectra)
        self.pulsar_term = pulsar_term
        self.theta_gw = theta_gw
        self.phi_gw = phi_gw
        self.pos = - khat(self.thetas, self.phis)
        self.NPIX = NPIX
        self.Tspan = get_Tspan(spectra)
        # print("First few phis:", self.phis[:5], "thetas:", self.thetas[:5])

        if pulsar_term == 'explicit':
            self.pdists = np.array([(sp.pdist/c.c).to('s').value
                                    for sp in spectra]) #pulsar distances

        #Return 3xN array of k,l,m GW position vectors.
        self.K = khat(self.theta_gw, self.phi_gw)
        self.L = lhat(self.theta_gw, self.phi_gw)
        self.M = mhat(self.theta_gw, self.phi_gw)
        LL = np.einsum('ij, kj->ikj', self.L, self.L)
        MM = np.einsum('ij, kj->ikj', self.M, self.M)
        KK = np.einsum('ij, kj->ikj', self.K, self.K)
        LM = np.einsum('ij, kj->ikj', self.L, self.M)
        ML = np.einsum('ij, kj->ikj', self.M, self.L)
        KM = np.einsum('ij, kj->ikj', self.K, self.M)
        MK = np.einsum('ij, kj->ikj', self.M, self.K)
        KL = np.einsum('ij, kj->ikj', self.K, self.L)
        LK = np.einsum('ij, kj->ikj', self.L, self.K)
        self.eplus = MM - LL
        self.ecross = LM + ML
        self.e_b = LL + MM
        self.e_ell = KK # np.sqrt(2)*
        self.e_x = KL + LK
        self.e_y = KM + MK
        num = 0.5 * np.einsum('ij, kj->ikj', self.pos, self.pos)
        denom = 1 + np.einsum('ij, il->jl', self.pos, self.K)
        Npsrs = len(self.phis)
        # Npairs = np.int(Npsrs * (Npsrs-1) / 2.)
        psr_idx = np.arange(Npsrs)
        pairs = list(it.combinations(psr_idx,2))
        self.first, self.second = list(map(list, zip(*pairs)))
        self.cosThetaIJ = np.cos(self.thetas[self.first]) * np.cos(self.thetas[self.second]) \
                        + np.sin(self.thetas[self.first]) * np.sin(self.thetas[self.second]) \
                        * np.cos(self.phis[self.first] - self.phis[self.second])

        self.D = num[:,:,:,np.newaxis]/denom[np.newaxis, np.newaxis,:,:]
        if pulsar_term == 'explicit':
            Dp = self.pdists[:,np.newaxis] * denom
            Dp = self.freqs[:,np.newaxis,np.newaxis] * Dp[np.newaxis,:,:]
            pt = 1-np.exp(-1j*2*np.pi*Dp)
            pt /= 2*np.pi*1j*self.freqs[:,np.newaxis,np.newaxis]
            self.pt_sqr = np.abs(pt)**2

        if pol=='gr':
            self.Fplus = np.einsum('ijkl, ijl ->kl', self.D, self.eplus)
            self.Fcross = np.einsum('ijkl, ijl ->kl', self.D, self.ecross)
                        
            self.R_IJ = np.zeros((len(self.first), len(self.Fplus[0])))  # Shape: (N_pairs, NPIX)

            for i, (f, s) in enumerate(zip(self.first, self.second)):
                self.R_IJ[i, :] = self.Fplus[f, :] * self.Fplus[s, :] + self.Fcross[f, :] * self.Fcross[s, :]

            self.R_IJ *= 3 / (2 * self.NPIX)

        elif pol=='scalar-trans':
            self.Fbreathe = np.einsum('ijkl, ijl ->kl',self.D, self.e_b)
            self.sky_response = self.Fbreathe**2
        elif pol=='scalar-long':
            self.Flong = np.einsum('ijkl, ijl ->kl',self.D, self.e_ell)
            self.sky_response = self.Flong**2
        elif pol=='vector-long':
            self.Fx = np.einsum('ijkl, ijl ->kl',self.D, self.e_x)
            self.Fy = np.einsum('ijkl, ijl ->kl',self.D, self.e_y)
            self.sky_response = self.Fx**2 + self.Fy**2

        if pulsar_term == 'explicit':
            self.sky_response = (0.5 * self.sky_response[np.newaxis,:,:]
                                 * self.pt_sqr)
    
    # @property
    # def S_eff(self):
    #     """
    #     Strain power sensitivity. NOTE: The prefactors in these expressions are a factor of 4x larger than in 
    #     Hazboun, et al., 2019 `[1]` due to a redefinition of h0 to match the one in normal use in the PTA community.
    #     .. _[1]: https://arxiv.org/abs/1907.04341
    #     """
    #     if not hasattr(self, '_S_eff'):
    #         if self.pulsar_term == 'explicit':
    #             self._S_eff = 1.0 / (16./5 * np.sum(self.S_SkyIJ, axis=1))
    #         elif self.pulsar_term:
    #             self._S_eff = 1.0 / (48./5 * np.sum(self.S_SkyIJ, axis=1))
    #         else:
    #             self._S_eff = 1.0 / (24./5 * np.sum(self.S_SkyIJ, axis=1))
    #     return self._S_eff

    
    @property
    def S_eff(self): 
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            kk = np.arange(len(self.chiIJ))
            num = self.T_IJ[kk] / self.Tspan * self.chiIJ[kk]**2
            series = num[:,np.newaxis] / (self.SnI[ii] * self.SnI[jj])
            self._S_eff = np.power(np.sum(series, axis=0),-0.5)
        return self._S_eff

    @property
    def S_IJ(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_IJ'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            # kk = np.arange(len(self.ii))
            self._S_IJ =  np.sqrt((self.SnI[ii] * self.SnI[jj]))
        return self._S_IJ

    @property
    def M_kk(self):
        """Fisher Matrix for Anisotropy."""
        if not hasattr(self, '_M_kk'): # Shape (Nfreqs, Npix) -- (IJ, f, k)
            self._M_kk = np.sum(self.R_IJ[:,np.newaxis,:]**2 * self.T_IJ[:,np.newaxis,np.newaxis] / self.S_IJ[:,:,np.newaxis], axis=0)
        return self._M_kk
    
    @property
    def M_kkp(self):
        """Fisher Matrix M_{kk', f} for Anisotropy."""
        if not hasattr(self, '_M_kkp'): # Shape (Nfreqs, Npix, Npix) -- (IJ, f, k, k')
            self._M_kkp = np.sum(self.R_IJ[:,np.newaxis,:,np.newaxis] * self.R_IJ[:,np.newaxis,np.newaxis,:] * self.T_IJ[:,np.newaxis,np.newaxis,np.newaxis] / self.S_IJ[:,:,np.newaxis,np.newaxis], axis=0)
        return self._M_kkp
    
    @property
    def S_clean(self):
        """ Sclean """
        if not hasattr(self, '_S_clean'): # Shape (Nfreqs, Npix)
            num = np.sum(self.chiIJ[:,np.newaxis,np.newaxis] * self.T_IJ[:,np.newaxis,np.newaxis] * self.R_IJ[:,np.newaxis,:] / self.S_IJ[:,:,np.newaxis], axis=0)
            denom = np.sqrt(np.sum(self.T_IJ[:,np.newaxis,np.newaxis] * self.R_IJ[:,np.newaxis,:]**2 / self.S_IJ[:,:,np.newaxis], axis=0))
            self._S_clean = num / denom
        return self._S_clean

    @property
    def Seff_A_aniso(self): 
        """Strain power sensitivity."""
        if not hasattr(self, '_Seff_A_aniso'): # Shape (Nfreqs, Npix)
            self._Seff_A_aniso = np.sum(self.chiIJ[:,np.newaxis,np.newaxis] * self.T_IJ[:,np.newaxis,np.newaxis] * self.R_IJ[:,np.newaxis,:] / self.S_IJ[:,:,np.newaxis], axis=0)
        return self._Seff_A_aniso

    @property
    def S_eff_aniso_p(self): 
        """Effective Sky Sensitivity."""
        if not hasattr(self, '_S_eff_aniso_p'): # New version Shape (Nfreqs, Npix, Npix)
            # self._S_eff_aniso = (((self.Seff_A_aniso.sum(axis=0) ** 2 / self.M_kk.sum(axis=0)) / np.max(self.T_IJ)) ** (-1)) # Gives factor of 30 too low
            self._S_eff_aniso_p = (self.Seff_A_aniso[:,:,np.newaxis] * (self.M_kkp[:,:,:])**(-1) * self.Seff_A_aniso[:,np.newaxis,:]) # New version
        return self._S_eff_aniso_p
 
    @property
    def h_c_aniso_p(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c_aniso_p'):
            self._h_c_aniso_p = np.sqrt(self.freqs * np.sum(self.S_eff_aniso_p, axis=(1,2))/self.NPIX)
        return self._h_c_aniso_p
    
    @property
    def S_eff_aniso(self): 
        """Effective Sky Sensitivity."""
        if not hasattr(self, '_S_eff_aniso'): # New version Shape (Nfreqs, Npix)
            # self._S_eff_aniso = (((self.Seff_A_aniso.sum(axis=0) ** 2 / self.M_kk.sum(axis=0)) / np.max(self.T_IJ)) ** (-1)) # Gives factor of 30 too low
            self._S_eff_aniso = self.Seff_A_aniso[:,:] * (self.M_kk[:,:])**(-1) * self.Seff_A_aniso[:,:] # New version
        return self._S_eff_aniso
    
    @property
    def h_c_aniso(self):
        """Characteristic strain sensitivity"""
        if not hasattr(self, '_h_c_aniso'):
            self._h_c_aniso = np.sqrt(self.freqs * np.sum(self.S_eff_aniso/(self.NPIX*self.T_IJ), axis=1)**(-1/2))
        return self._h_c_aniso


    @property
    def S_eff_mean(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_eff_mean'):
            mean_sky = np.mean(np.sum(self.S_SkyI, axis=1), axis=1)
            if self.pulsar_term:
                self._S_eff_mean = 1.0 / (4./5 * mean_sky)
            else:
                self._S_eff_mean = 1.0 / (12./5 * mean_sky)
        return self._S_eff_mean

def khat(theta, phi):
    r'''Returns :math:`\hat{k}` from paper.
    Also equal to :math:`-\hat{r}=-\hat{n}`.'''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def lhat(theta, phi):
    r'''Returns :math:`\hat{l}` from paper. Also equal to :math:`-\hat{\phi}`.'''
    return np.array([np.sin(phi), -np.cos(phi), np.zeros_like(theta)])

def mhat(theta, phi):
    r'''Returns :math:`\hat{m}` from paper. Also equal to :math:`-\hat{\theta}`.'''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])
