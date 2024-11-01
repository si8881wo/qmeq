"""Module containing python functions, which generate first order Redfield kernel.
   For docstrings see documentation of module neumann1."""

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

from ..aprclass import Approach
from .neumann1 import Approach1vN

from ..kernel_handler import KernelHandlerNoise

# ---------------------------------------------------------------------------------------------------
# Redfield approach
# ---------------------------------------------------------------------------------------------------
class ApproachRedfield(Approach):

    kerntype = 'pyRedfield'

    def restart(self): # simon
        Approach.restart(self)
        self.Lpm = None
        self.current_noise = None
        self.energy_current_noise = None

    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandlerNoise(self.si)

    def prepare_arrays(self):
        Approach1vN.prepare_arrays(self)

    def clean_arrays(self):
        Approach1vN.clean_arrays(self)

    def generate_fct(self):
        Approach1vN.generate_fct(self)

    def generate_coupling_terms(self, b, bp, bcharge):
        Tba, phi1fct = self.leads.Tba, self.phi1fct
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm
        Lpm = self.Lpm #simon
        countingleads = self.funcp.countingleads #simon

        acharge = bcharge-1
        ccharge = bcharge+1

        # --------------------------------------------------
        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                bpap = si.get_ind_dm1(bp, ap, acharge)
                ba = si.get_ind_dm1(b, a, acharge)
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += (+ Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate()
                                - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, ba, 0])
                    if l in countingleads:
                        kh.set_matrix_element_lpm(Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate() - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, ba, 0], 0, b, bp, bcharge, a, ap, acharge)
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    bppa = si.get_ind_dm1(bpp, a, acharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bppa, 1].conjugate()
                for c in statesdm[ccharge]:
                    cbpp = si.get_ind_dm1(c, bpp, bcharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbpp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    bppa = si.get_ind_dm1(bpp, a, acharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, bppa, 1]
                for c in statesdm[ccharge]:
                    cbpp = si.get_ind_dm1(c, bpp, bcharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cbpp, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                cpbp = si.get_ind_dm1(cp, bp, bcharge)
                cb = si.get_ind_dm1(c, b, bcharge)
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += (+ Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpbp, 1]
                                - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cb, 1].conjugate())
                    if l in countingleads:
                        kh.set_matrix_element_lpm(Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpbp, 1] - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cb, 1].conjugate(), 1, b, bp, bcharge, c, cp, ccharge)
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------
           
    def generate_current(self):
        self.generate_current_std()
        self.generate_current_noise()
        
    def generate_current_std(self):
        E, Tba = self.qd.Ea, self.leads.Tba
        phi1fct, phi1fct_energy = self.phi1fct, self.phi1fct_energy

        si = self.si
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        phi1 = self.phi1
        current = self.current
        energy_current = self.energy_current

        kh = self.kernel_handler
        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)

                for l in range(nleads):
                    current_l, energy_current_l = 0, 0

                    for bp in statesdm[bcharge]:
                        if not kh.is_included(bp, b, bcharge):
                            continue
                        phi0bpb = kh.get_phi0_element(bp, b, bcharge)

                        cbp = si.get_ind_dm1(c, bp, bcharge)
                        fct1 = phi1fct[l, cbp, 0]
                        fct1h = phi1fct_energy[l, cbp, 0]

                        phi1[l, cb] += Tba[l, c, bp]*phi0bpb*fct1
                        current_l += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1
                        energy_current_l += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1h

                    for cp in statesdm[ccharge]:
                        if not kh.is_included(c, cp, ccharge):
                            continue
                        phi0ccp = kh.get_phi0_element(c, cp, ccharge)

                        cpb = si.get_ind_dm1(cp, b, bcharge)
                        fct2 = phi1fct[l, cpb, 1]
                        fct2h = phi1fct_energy[l, cpb, 1]

                        phi1[l, cb] += Tba[l, cp, b]*phi0ccp*fct2
                        current_l += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2
                        energy_current_l += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2h

                    current[l] += -2*current_l.imag
                    energy_current[l] += -2*energy_current_l.imag

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
