"""Module containing python functions, which generate first order 1vN kernel."""

import numpy as np
import itertools

from ...wrappers.mytypes import complexnp
from ...wrappers.mytypes import doublenp

from ...specfunc.specfunc import func_1vN
from ..aprclass import Approach


# ---------------------------------------------------------------------------------------------------
# 1 von Neumann approach
# ---------------------------------------------------------------------------------------------------
class Approach1vN(Approach):

    kerntype = 'py1vN'

    def prepare_arrays(self):
        Approach.prepare_arrays(self)
        nleads, ndm1 = self.si.nleads, self.si.ndm1
        self.phi1fct = np.zeros((nleads, ndm1, 2), dtype=complexnp)
        self.phi1fct_energy = np.zeros((nleads, ndm1, 2), dtype=complexnp)
        self.phi1 = np.zeros((nleads, ndm1), dtype=complexnp)
        ndm0r = self.si.ndm0r #simon
        self.Lpm = np.zeros((2, ndm0r, ndm0r), dtype = doublenp) #simon
        self.kernel_handler.set_lpm(self.Lpm) #simon
        self.current_noise = np.zeros(2) #simon
        self.energy_current_noise = np.zeros(1) #simon
        # create additional vectors/matrices: 1d array with noise at all leads, matrix with derivative (liouvillian/parts)
        # make sure kernel handler knows where to find new kernel (point to it)

    def clean_arrays(self):
        Approach.clean_arrays(self)
        self.phi1fct.fill(0.0)
        self.phi1fct_energy.fill(0.0)
        self.phi1.fill(0.0)
        self.Lpm.fill(0.0) #
        self.current_noise.fill(0.0) #
        self.energy_current_noise.fill(0.0) #
        # create additional vectors/matrices: 1d array with noise at all leads, matrix with derivative (liouvillian/parts)

    def generate_fct(self):
        """
        Make factors used for generating 1vN, Redfield master equation kernels.

        Parameters
        ----------
        phi1fct : array
            (Modifies) Factors used for generating 1vN, Redfield master equation kernels.
        phi1fct_energy : array
            (Modifies) Factors used to calculate energy and heat currents in 1vN, Redfield approaches.
        """
        E, si = self.qd.Ea, self.si,
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        itype, limit = self.funcp.itype, self.funcp.dqawc_limit
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        phi1fct = self.phi1fct
        phi1fct_energy = self.phi1fct_energy
        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                Ecb = E[c]-E[b]
                for l in range(nleads):
                    rez = func_1vN(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, limit)
                    phi1fct[l, cb, 0] = rez[0]
                    phi1fct[l, cb, 1] = rez[1]
                    phi1fct_energy[l, cb, 0] = rez[2]
                    phi1fct_energy[l, cb, 1] = rez[3]

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
                bpa = si.get_ind_dm1(bp, a, acharge)
                bap = si.get_ind_dm1(b, ap, acharge)
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += (+ Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bap, 0])
                    if l in countingleads:
                        kh.set_matrix_element_lpm(Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate() - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bap, 0], 0, b, bp, bcharge, a, ap, acharge)
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    bpa = si.get_ind_dm1(bp, a, acharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bpa, 1].conjugate()
                for c in statesdm[ccharge]:
                    cbp = si.get_ind_dm1(c, bp, bcharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    ba = si.get_ind_dm1(b, a, acharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, ba, 1]
                for c in statesdm[ccharge]:
                    cb = si.get_ind_dm1(c, b, bcharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cb, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                cbp = si.get_ind_dm1(c, bp, bcharge)
                cpb = si.get_ind_dm1(cp, b, bcharge)
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += (+ Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cbp, 1]
                                - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                    if l in countingleads:
                        kh.set_matrix_element_lpm(Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cbp, 1] - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate(), 1, b, bp, bcharge, c, cp, ccharge)
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------
           
    def generate_current(self):
        self.generate_current_std()
        self.generate_current_noise()
        
    def generate_current_std(self):
        """
        Calculates currents using 1vN approach.

        Parameters
        ----------
        phi1 : array
            (Modifies) Values of first order density matrix elements
            stored in nleads by ndm1 numpy array.
        current : array
            (Modifies) Values of the current having nleads entries.
        energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
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
                    current_l, energy_current_l, phi1_l = 0, 0, 0

                    fct1 = phi1fct[l, cb, 0]
                    fct2 = phi1fct[l, cb, 1]
                    fct1h = phi1fct_energy[l, cb, 0]
                    fct2h = phi1fct_energy[l, cb, 1]

                    for bp in si.statesdm[bcharge]:
                        if not kh.is_included(bp, b, bcharge):
                            continue
                        phi0bpb = kh.get_phi0_element(bp, b, bcharge)

                        phi1_l += Tba[l, c, bp]*phi0bpb*fct1
                        current_l += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1
                        energy_current_l += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1h

                    for cp in statesdm[ccharge]:
                        if not kh.is_included(c, cp, ccharge):
                            continue
                        phi0ccp = kh.get_phi0_element(c, cp, ccharge)

                        phi1_l += Tba[l, cp, b]*phi0ccp*fct2
                        current_l += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2
                        energy_current_l += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2h

                    phi1[l, cb] += phi1_l
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
        eps = 1e-4
        R   = Q @ np.linalg.inv(1j*eps + kern) @ Q 
        
        # current and noise
        Jp  = 1j*Lp - 1j*Lm
        Jpp = -Lp - Lm
        c = -1j*(O @ Jp @ P)
        s = -O @ (Jpp - 2*(Jp @ R @ Jp)) @ P
        self.current_noise[0] = c.real.item()
        self.current_noise[1] = s.real.item()
# ---------------------------------------------------------------------------------------------------
