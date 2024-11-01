"""Module containing python functions, which generate first order Lindblad kernels."""

import numpy as np
import itertools

from ...wrappers.mytypes import complexnp
from ...wrappers.mytypes import doublenp

from ...specfunc.specfunc import func_pauli
from ..aprclass import Approach

from ..kernel_handler import KernelHandlerNoise

# ---------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------
class ApproachLindblad(Approach):

    kerntype = 'pyLindblad'

    def restart(self): # simon
        Approach.restart(self)
        self.Lpm = None
        self.current_noise = None
        self.energy_current_noise = None

    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandlerNoise(self.si)

    def prepare_arrays(self):
        Approach.prepare_arrays(self)
        Tba, mtype = self.leads.Tba, self.leads.mtype
        self.tLba = np.zeros(Tba.shape, dtype=mtype)
        ndm0r = self.si.ndm0r #simon
        self.Lpm = np.zeros((2, ndm0r, ndm0r), dtype = doublenp) #simon
        self.kernel_handler.set_lpm(self.Lpm) #simon
        self.current_noise = np.zeros(2) #simon
        self.energy_current_noise = np.zeros(1) #simon
        # create additional vectors/matrices: 1d array with noise at all leads, matrix with derivative (liouvillian/parts)
        # make sure kernel handler knows where to find new kernel (point to it)

    def clean_arrays(self):
        Approach.clean_arrays(self)
        self.tLba.fill(0.0)
        self.Lpm.fill(0.0) #
        self.current_noise.fill(0.0) #
        self.energy_current_noise.fill(0.0) #
        # create additional vectors/matrices: 1d array with noise at all leads, matrix with derivative (liouvillian/parts)


    def generate_fct(self):
        """
        Make factors used for generating Lindblad master equation kernel.

        Parameters
        ----------
        tLba : array
            (Modifies) Jump operator matrix in many-body basis.
        """
        Tba, E, si = self.leads.Tba, self.qd.Ea, self.si
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        itype = self.funcp.itype
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        tLba = self.tLba
        for charge in range(ncharge-1):
            bcharge = charge+1
            acharge = charge
            for b, a in itertools.product(statesdm[bcharge], statesdm[acharge]):
                Eba = E[b]-E[a]
                for l in range(nleads):
                    fct1, fct2 = func_pauli(Eba, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    tLba[l, b, a] = np.sqrt(fct1)*Tba[l, b, a]
                    tLba[l, a, b] = np.sqrt(fct2)*Tba[l, a, b]

    def generate_coupling_terms(self, b, bp, bcharge):
        tLba = self.tLba
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm
        Lpm = self.Lpm #simon
        countingleads = self.funcp.countingleads #simon

        acharge = bcharge-1
        ccharge = bcharge+1

        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                    if l in countingleads:
                        kh.set_matrix_element_lpm(1j*tLba[l, b, a]*tLba[l, bp, ap].conjugate(), 0, b, bp, bcharge, a, ap, acharge)
                kh.set_matrix_element(1j*fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                for c in statesdm[ccharge]:
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                kh.set_matrix_element(1j*fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                for c in statesdm[ccharge]:
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                kh.set_matrix_element(1j*fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                    if l in countingleads:
                        kh.set_matrix_element_lpm(1j*tLba[l, b, c]*tLba[l, bp, cp].conjugate(), 1, b, bp, bcharge, c, cp, ccharge)
                kh.set_matrix_element(1j*fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------
            
    def generate_current(self):
        self.generate_current_std()
        self.generate_current_noise()
        
    def generate_current_std(self):
        """
        Calculates currents using Lindblad approach.

        Parameters
        ----------
        current : array
            (Modifies) Values of the current having nleads entries.
        energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
        phi0p, E, tLba, si = self.phi0, self.qd.Ea, self.tLba, self.si
        ndm0, npauli = si.ndm0, si.npauli
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        current = self.current
        energy_current = self.energy_current

        kh = self.kernel_handler
        for charge in range(ncharge):
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1

            for b, bp in itertools.product(statesdm[bcharge], statesdm[bcharge]):
                if not kh.is_included(b, bp, bcharge):
                    continue
                phi0bbp = kh.get_phi0_element(b, bp, bcharge)

                for l in range(nleads):
                    current_l, energy_current_l = 0, 0

                    for a in statesdm[acharge]:
                        fcta = tLba[l, a, b]*phi0bbp*tLba[l, a, bp].conjugate()
                        current_l -= fcta
                        energy_current_l += (E[a]-0.5*(E[b]+E[bp]))*fcta
                    for c in statesdm[ccharge]:
                        fctc = tLba[l, c, b]*phi0bbp*tLba[l, c, bp].conjugate()
                        current_l += fctc
                        energy_current_l += (E[c]-0.5*(E[b]+E[bp]))*fctc

                    current[l] += current_l.real
                    energy_current[l] += energy_current_l.real

        self.heat_current[:] = energy_current - current*self.leads.mulst
            
    def generate_current_noise(self): #simon
        """
        Calculates currents using Pauli master equation approach and noise via the C.Emary approach summed over countingleads passed

        Returns
        ----------
        current : float
            Value of the current attaching the counting field to countingleads.
        noise : array
            Value of the current noise attaching the counting field to countingleads.
        """
        phi0, E, si = self.phi0, self.qd.Ea, self.si
        nleads = si.nleads
        ndm0r, npauli = si.ndm0r, si.npauli
        kern, Lpm = self.kern, self.Lpm
        Lp, Lm = self.Lpm
        
        # auxilliary quantities
        # right eigenvector
        P = phi0[...,None]
        # left eigenvector
        O = np.zeros(ndm0r)[None,...]
        O[0,:npauli].fill(1.0)
        
        # projector
        Q = (np.eye(np.size(P)) - P @ O)
        # pseudoinverse
        eps = 1e-10
        R   = Q @ np.linalg.inv(1j*eps*np.eye(ndm0r) + kern) @ Q 
        
        # current and noise
        Jp  = 1j*Lp - 1j*Lm
        Jpp = -Lp - Lm
        c = -1j*(O @ Jp @ P)
        s = -O @ (Jpp - 2*(Jp @ R @ Jp)) @ P
        self.current_noise[0] = c.real.item()
        self.current_noise[1] = s.real.item()
# ---------------------------------------------------------------------------------------------------
