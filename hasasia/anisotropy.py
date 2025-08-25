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
import hasasia.sensitivity as hsen


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
                 pulsar_term=False, pol='gr', iota=None, psi=None, NPIX=12):
        super().__init__(spectra)
        self.pulsar_term = pulsar_term
        self.theta_gw = theta_gw
        self.phi_gw = phi_gw
        self.pos = - khat(self.thetas, self.phis)
        self.NPIX = NPIX
        self.Tspan = get_Tspan(spectra)

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

            self.R_IJ = (
                self.Fplus[self.first] * self.Fplus[self.second] +
                self.Fcross[self.first] * self.Fcross[self.second]
            ) * (3 / (2 * self.NPIX))

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

    
    # @property
    # def S_eff(self): 
    #     """Strain power sensitivity. """
    #     if not hasattr(self, '_S_eff'):
    #         ii = self.pairs[0]
    #         jj = self.pairs[1]
    #         kk = np.arange(len(self.chiIJ))
    #         num = self.T_IJ[kk] / self.Tspan * self.chiIJ[kk]**2
    #         series = num[:,np.newaxis] / (self.SnI[ii] * self.SnI[jj])
    #         self._S_eff = np.power(np.sum(series, axis=0),-0.5)
    #     return self._S_eff

    @property
    def S_IJ(self):
        """Strain power sensitivity. """
        if not hasattr(self, '_S_IJ'):
            ii = self.pairs[0]
            jj = self.pairs[1]
            # kk = np.arange(len(self.ii))
            self._S_IJ =  np.sqrt((self.SnI[ii] * self.SnI[jj]))
        return self._S_IJ

    
    def M_kkp(self, chunk_size=12, diag=False):
        """
        Full or diagonal Fisher matrix M_{kk', f} using einsum, with optional chunking.
        
        Parameters
        ----------
        chunk_size : int or None
            Size of pixel chunks for block computation. If None, do all at once.
        diag : bool, default=False
            If True, compute only the diagonal (shape: Nfreqs x Npix).
            If False, compute full matrix (shape: Nfreqs x Npix x Npix).
        
        Returns
        -------
        np.ndarray
            Fisher matrix.
        """
        W_if = self.T_IJ[:, None] / self.S_IJ  # (Npairs, Nfreqs)
        Nfreqs, Npix = self.S_IJ.shape[1], self.R_IJ.shape[1]

        if diag:
            # Diagonal only: (Nfreqs, Npix)
            if chunk_size is None:
                return np.einsum('if,ik,ik->fk', W_if, self.R_IJ, self.R_IJ, optimize=True)
            else:
                M_diag = np.zeros((Nfreqs, Npix))
                for i in range(0, Npix, chunk_size):
                    M_diag[:, i:i+chunk_size] = np.einsum(
                        'if,ik,ik->fk',
                        W_if,
                        self.R_IJ[:, i:i+chunk_size],
                        self.R_IJ[:, i:i+chunk_size],
                        optimize=True
                    )
                return M_diag
        else:
            # Full matrix: (Nfreqs, Npix, Npix)
            M_full = np.zeros((Nfreqs, Npix, Npix))
            if chunk_size is None:
                return np.einsum('if,ik,il->fkl', W_if, self.R_IJ, self.R_IJ, optimize=True)
            else:
                for i in range(0, Npix, chunk_size):
                    for j in range(0, Npix, chunk_size):
                        M_full[:, i:i+chunk_size, j:j+chunk_size] = np.einsum(
                            'if,ik,il->fkl',
                            W_if,
                            self.R_IJ[:, i:i+chunk_size],
                            self.R_IJ[:, j:j+chunk_size],
                            optimize=True
                        )
                return M_full

    def S_eff_kkp(self, chunk_size=12, diag=False):
        """
        Efficient einsum calculation of S_eff_kkp = v_k * M_kkp^-1 * v_kp
        with optional pixel chunking and diagonal-only mode.
        
        Parameters
        ----------
        chunk_size : int or None
            If given, compute S_eff in pixel blocks. Otherwise compute full matrix at once.
        diag : bool, default=False
            If True, only compute the diagonal pixel blocks (i == j).
        
        Returns
        -------
        S_eff : np.ndarray
            Shape (Nfreqs, Npix, Npix) if diag=False
            Shape (Nfreqs, Npix) if diag=True
        """
        Nfreqs, Npix = self.S_IJ.shape[1], self.R_IJ.shape[1]

        # Precompute weights (Npairs, Nfreqs)
        W_if = (self.chiIJ[:, None] * self.T_IJ[:, None]) / self.S_IJ

        # v_k and v_kp (Nfreqs, Npix)
        v_k = np.einsum('if,ik->fk', W_if, self.R_IJ)
        v_kp = np.einsum('if,il->fl', W_if, self.R_IJ)

        # Get full M_kkp matrix and invert
        M_kkp_array = self.M_kkp(chunk_size=chunk_size, diag=diag)

        # Invert M_kkp
        if diag:
            # Just take reciprocal (element-wise inverse)
            M_kkp_inv = 1.0 / M_kkp_array  # shape (Nfreqs, Npix)
            S_eff_diag = np.einsum('fk,fk,fk->fk', v_k, M_kkp_inv, v_kp, optimize=True)
            S_safe = np.clip(S_eff_diag / self.Tspan, 1e-30, None)
            return S_safe**(-1/2)

        else:
            # Full matrix inversion per frequency
            M_kkp_inv = np.zeros_like(M_kkp_array)
            for f in range(Nfreqs):
                M_kkp_inv[f] = np.linalg.inv(M_kkp_array[f])

            # Compute S_eff
            S_eff = np.zeros((Nfreqs, Npix, Npix))
            if chunk_size is None:
                S_eff = np.einsum('fk,fkl,fl->fkl', v_k, M_kkp_inv, v_kp, optimize=True)
            else:
                for i in range(0, Npix, chunk_size):
                    for j in range(0, Npix, chunk_size):
                        S_eff[:, i:i+chunk_size, j:j+chunk_size] = np.einsum(
                            'fk,fkl,fl->fkl',
                            v_k[:, i:i+chunk_size],
                            M_kkp_inv[:, i:i+chunk_size, j:j+chunk_size],
                            v_kp[:, j:j+chunk_size],
                            optimize=True
                        )

            S_safe = np.clip(S_eff / self.Tspan, 1e-30, None)
            return S_safe**(-1/2)

    def S_clean(self, A=1e-14, alpha=-2/3):
        """
        Compute S_clean = 2 * S_h(f) * Tspan / S_eff
        using efficient einsum contractions with optimization.

        Parameters
        ----------
        A : float
            Amplitude for S_h(f).
        alpha : float
            Spectral index for S_h(f).

        Returns
        -------
        S_clean_fk : np.ndarray
            Shape (Nfreqs, Npix)
        """
        # seff_num: Tspan * sqrt(sum_over_pairs( T_IJ * R_IJ^2 / S_IJ ))
        seff_num = self.Tspan * np.sqrt(
            np.einsum('i,ik,if->fk',
                    self.T_IJ,
                    self.R_IJ**2,
                    1.0 / self.S_IJ,
                    optimize=True)
        )

        # seff_denom: sum_over_pairs( chiIJ * T_IJ * R_IJ / S_IJ )
        seff_denom = np.einsum('i,i,ik,if->fk',
                            self.chiIJ,
                            self.T_IJ,
                            self.R_IJ,
                            1.0 / self.S_IJ,
                            optimize=True)

        S_eff_fk = seff_num / seff_denom

        # S_h_f
        S_h_f = hsen.S_h(A, alpha, self.freqs)  # shape (Nfreqs,)

        # S_clean_fk: 2 * S_h_f * Tspan / S_eff_fk
        S_clean_fk = (2.0 * S_h_f[:, None] * self.Tspan) / S_eff_fk

        return S_clean_fk


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
