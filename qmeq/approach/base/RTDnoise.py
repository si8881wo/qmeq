"""Module containing RTD Approach."""

from itertools import product

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

from ...specfunc.specfunc import integralD
from ...specfunc.specfunc import integralX
from ...specfunc.specfunc import integralD_lpm
from ...specfunc.specfunc import integralX_lpm
from ...specfunc.specfunc import phi
from ...specfunc.specfunc import phi_lpm
from ...specfunc.specfunc import fermi_func
from ...specfunc.specfunc import delta_phi
from ...specfunc.specfunc import BW_Ozaki
from ...specfunc.specfunc import func_pauli
from ...specfunc.specfunc import fermi_lpm
from ...specfunc.specfunc import diff_phi
from .RTD import ApproachPyRTD
from ..kernel_handler import KernelHandlerRTDnoise

class ApproachPyRTDnoise(ApproachPyRTD):

    kerntype = 'pyRTDnoise'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lpm_h = 1e-10

    def restart(self):
        ApproachPyRTD.restart(self)
        
        self.Lpm = None
        self.current_noise = None
        self.lpm_imaginary_2nd = None

    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandlerRTDnoise(self.si)

    def prepare_arrays(self):
        ApproachPyRTD.prepare_arrays(self)

        nleads, ndm1 = self.si.nleads, self.si.ndm1
        kern_size = self.get_kern_size()

        self.paulifct_eps = np.zeros((nleads, ndm1, 2), dtype=doublenp)
        
        self.Lpm = np.zeros((3*5, kern_size, kern_size), dtype=complexnp)
        self.kernel_handler.set_lpm(self.Lpm)
        self.current_noise = np.zeros(2, dtype = complexnp)
        
        self.lpm_imaginary_2nd = True

    def clean_arrays(self):
        ApproachPyRTD.clean_arrays(self)
        
        self.Lpm.fill(0.0)
        self.current_noise.fill(0.0)
        
        self.lpm_imaginary_2nd = True
            
    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        """
        Solves the master equation.

        Parameters
        ----------
        qdq : bool
            Diagonalise many-body quantum dot Hamiltonian
            and express the lead matrix Tba in the eigenbasis.
        rotateq : bool
            Rotate the many-body tunneling matrix Tba.
        masterq : bool
            Solve the master equation.
        currentq : bool
            Calculate the current.
        """
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.rotate()
        #
        if masterq:
            self.prepare_kern()
            self.generate_fct()
            #self.generate_fct_lpm()
            if not self.funcp.mfreeq:
                self.generate_kern()
                self.solve_kern()
            else:
                self.solve_matrix_free()
            if currentq:
                self.generate_current_()
    
    def solve_first(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        """
        Solves the master equation.

        Parameters
        ----------
        qdq : bool
            Diagonalise many-body quantum dot Hamiltonian
            and express the lead matrix Tba in the eigenbasis.
        rotateq : bool
            Rotate the many-body tunneling matrix Tba.
        masterq : bool
            Solve the master equation.
        currentq : bool
            Calculate the current.
        """
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.rotate()
        #
        if masterq:
            self.prepare_kern()
            self.generate_fct()
            #self.generate_fct_lpm()
            if not self.funcp.mfreeq:
                self.generate_kern_first()
                self.solve_kern()
            else:
                self.solve_matrix_free()
            if currentq:
                self.generate_current_()

    def generate_kern_first(self):
        r""" Generates all kernels including tunnel processes of orders :math:`t^2` and :math:`t^4`.

        The total kernel used to solve for :math:`\phi_0` is :math:`W =\sum_r W^r= W_{dd}^{(1)} + W_{dd}^{(2)}
        + W_{dn}^{(1)} (L_{nn})^{-1} W_{nd}^{(1)}`. The last term is ignored if `off_diag_corrections` is False.

        Parameters
        ----------
        self.kern : ndarray
            (Modifies) The total Kernel for the diagonal density matrix. Has npauli * npauli entries.
        self.Wdd :  ndarray
            (Modifies) The lead-resolved Kernel for the diagonal density matrix. Has
            nleads * npauli * npauli entries.

        """
        si, kh = self.si, self.kernel_handler
        ncharge, statesdm = si.ncharge, si.statesdm
        self.off_diag_corrections = self.funcp.off_diag_corrections
        h = self.lpm_h

        if True:#(not np.all(np.isclose(self.leads.tlst, self.leads.tlst[0]))) or np.any(abs(self.leads.Tba.imag)>0):
            self.set_Ozaki_params()

        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                if not kh.is_unique(b, b, bcharge):
                    continue
                self.generate_row_1st_order_kernel(b, bcharge)
                self.generate_row_1st_order_kernel_lpm(b, bcharge)#simon
                #self.generate_col_diag_kern_2nd_order(b, bcharge)
                #self.generate_col_diag_kern_2nd_order_lpm(b, bcharge)
                self.generate_row_1st_energy_kernel(b, bcharge)
                #self.generate_row_2nd_energy_kernel(b, bcharge)

                if self.off_diag_corrections:
                    self.generate_col_nondiag_kern_1st_order_nd(b, bcharge)

        kern_size = self.get_kern_size()
        self.kern[:kern_size, :kern_size] += np.sum(self.Wdd, 0)

        if self.off_diag_corrections:
            for bcharge in range(ncharge):
                for b in statesdm[bcharge]:
                    for bp in statesdm[bcharge]:
                        if b == bp:
                            continue
                        self.generate_col_nondiag_kern_1st_order_dn(b, bp, bcharge)
                        self.generate_row_inverse_Liouvillian(b, bp, bcharge)
            self.add_off_diag_corrections()

    def generate_kern(self):
        r""" Generates all kernels including tunnel processes of orders :math:`t^2` and :math:`t^4`.

        The total kernel used to solve for :math:`\phi_0` is :math:`W =\sum_r W^r= W_{dd}^{(1)} + W_{dd}^{(2)}
        + W_{dn}^{(1)} (L_{nn})^{-1} W_{nd}^{(1)}`. The last term is ignored if `off_diag_corrections` is False.

        Parameters
        ----------
        self.kern : ndarray
            (Modifies) The total Kernel for the diagonal density matrix. Has npauli * npauli entries.
        self.Wdd :  ndarray
            (Modifies) The lead-resolved Kernel for the diagonal density matrix. Has
            nleads * npauli * npauli entries.

        """
        si, kh = self.si, self.kernel_handler
        ncharge, statesdm = si.ncharge, si.statesdm
        self.off_diag_corrections = self.funcp.off_diag_corrections
        h = self.lpm_h

        if True:#(not np.all(np.isclose(self.leads.tlst, self.leads.tlst[0]))) or np.any(abs(self.leads.Tba.imag)>0):
            self.set_Ozaki_params()

        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                if not kh.is_unique(b, b, bcharge):
                    continue
                self.generate_row_1st_order_kernel(b, bcharge)
                self.generate_row_1st_order_kernel_lpm(b, bcharge)#simon
                self.generate_col_diag_kern_2nd_order(b, bcharge)
                self.generate_col_diag_kern_2nd_order_lpm(b, bcharge)
                self.generate_row_1st_energy_kernel(b, bcharge)
                self.generate_row_2nd_energy_kernel(b, bcharge)

                if self.off_diag_corrections:
                    self.generate_col_nondiag_kern_1st_order_nd(b, bcharge)

        kern_size = self.get_kern_size()
        self.kern[:kern_size, :kern_size] += np.sum(self.Wdd, 0)

        if self.off_diag_corrections:
            for bcharge in range(ncharge):
                for b in statesdm[bcharge]:
                    for bp in statesdm[bcharge]:
                        if b == bp:
                            continue
                        self.generate_col_nondiag_kern_1st_order_dn(b, bp, bcharge)
                        self.generate_row_inverse_Liouvillian(b, bp, bcharge)
            self.add_off_diag_corrections()

    def generate_current_(self):
        self.generate_current()
        self.generate_current_noise()
        
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
        kern = sum(self.Lpm[0:5])
        Lm1, Lp1 , Lm2, Lp2 = self.Lpm[1:5]
        Lm1p, Lp1p , Lm2p, Lp2p = self.Lpm[6:10]
        Lm1p2, Lp1p2, Lm2p2, Lp2p2 = self.Lpm[11:15]
        Jdottemp = sum(self.Lpm[5:15])
        h = self.lpm_h
        #print(Jdottemp,'\n',Jdottemp-kern,'\n',(Jdottemp-kern)/1e-10,'\n',np.isclose(Jdottemp,kern,atol=1e-16,rtol=1e-16),'\n',np.isclose(self.kern,kern),'\n')
        
        # auxilliary quantities
        # right eigenvector
        P = phi0[...,None]
        # left eigenvector
        O = np.ones(np.size(P))[None,...]
        # projector
        Q = (np.eye(np.size(P)) - P @ O)
        # pseudoinverse
        eps = 1e-6
        R = Q @ np.linalg.pinv(kern) @ Q # Q @ np.linalg.inv(eps*np.eye(np.size(P)) + kern) @ Q
        # derivatives of noise kernel
        Jp = 1j*(Lp1 - Lm1 + 2*Lp2 - 2*Lm2)
        Jpp = -Lp1 - Lm1 - 4*Lp2 - 4*Lm2
        Jdot = Jdottemp
        Jdotp = 1j*(Lp1p+Lp1p2 - Lm1p-Lm1p2 + 2*Lp2p+2*Lp2p2 - 2*Lm2p-2*Lm2p2)
        # current and noise
        c = -1j*(O @ Jp @ P)
        s = -(O @ (Jpp - 2*(Jp @ R @ Jp)) @ P) + 2 * c *(O @ (Jdotp - (Jp @ R @ (Jdot))) @ P)
        #print('S_m = ',-O @ (Jpp - 2*(Jp @ R @ Jp)) @ P)
        #print(2*c * O @ (Jdotp - Jp @ R @ Jdot) @ P)
        self.current_noise[0] = c.item()
        self.current_noise[1] = s.item()

    # def generate_fct_lpm(self,h=1e-6):
        # """
        # Make factors used for generating the first order diagonal kernel :math:`W_{dd}^{(1)}`.

        # Parameters
        # ----------
        # paulifct : array
            # (Modifies) Factors used for generating Pauli master equation kernel.
        # """
        # E, Tba, si = self.qd.Ea, self.leads.Tba, self.si
        # mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        # ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        # itype = self.funcp.itype
        # paulifct_eps = self.paulifct_eps
        # for charge in range(ncharge-1):
            # ccharge = charge+1
            # bcharge = charge
            # for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                # cb = si.get_ind_dm1(c, b, bcharge)
                # Ecb = E[c]-E[b]+h
                # for l in range(nleads):
                    # xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                    # rez = func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    # paulifct_eps[l, cb, 0] = xcb*rez[0]
                    # paulifct_eps[l, cb, 1] = xcb*rez[1]
   
    def generate_row_1st_order_kernel_lpm(self, b, bcharge):
        """Generates a row in the first order diagonal kernel :math:`W_{dd}^{(1)}`.

        Parameters
        ----------
        b : int
            the final state (row)

        bcharge : int
            charge of state b

        self.Wdd : ndarray
            (Modifies) The kernel connecting diagional density-matrix elements. This Kernel
            has npauli * npauli entries.
        """
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm
        Lpm = self.Lpm #simon
        countingleads = self.funcp.countingleads #simon
        itype = self.funcp.itype
        pi = np.pi
        
        (E, Tba, tleads, mulst, tlst, dlst) = (self.qd.Ea, self.leads.Tba, self.leads.tleads_array,
                                                   self.leads.mulst, self.leads.tlst, self.leads.dlst)
        
        acharge = bcharge-1
        ccharge = bcharge+1
        ## possibly scaling of phi_pm factor by pi
        bb = si.get_ind_dm0(b, b, bcharge)
        for a in statesdm[acharge]: # careful: 1) eta=-xi from emary 2) a initial state, this adds an electron -> L+
            aa = si.get_ind_dm0(a, a, acharge)
            ba = si.get_ind_dm1(b, a, acharge)
            dE = E[b] - E[a]
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], Tba[l, a, b] * Tba[l, a, b].conj()
                # p0=1,p1=-1,eta=1
                lamb_p = dE - mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=-1,p1=1,eta=-1
                lamb_m = -dE + mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # epsilon derivative
                phi_eps = diff_phi(lamb_p/Tr)
                if l in countingleads:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m), 2, bb, aa) # set kernel element
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 7, bb, aa) # set epsilon derivative
                else:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m), 0, bb, aa)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 5, bb, aa)
        for c in statesdm[ccharge]: # carefull: 1) eta=-xi from emary 2) c initial state, this takes out an electron -> L-
            cc = si.get_ind_dm0(c, c, ccharge)
            cb = si.get_ind_dm1(c, b, bcharge)
            dE = E[c] - E[b]
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], Tba[l, c, b] * Tba[l, c, b].conj()
                # p0=-1,p1=1,eta=1
                lamb_m = dE - mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=1,p1=-1,eta=-1
                lamb_p = -dE + mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # epsilon derivative
                phi_eps = diff_phi(lamb_p/Tr)
                if l in countingleads:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m),1,bb,cc)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 6, bb, cc)
                else:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m),0,bb,cc)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 5, bb, cc)
        # diagonals
        # loop over intermediatate states
        for bm in statesdm[acharge]:# acharge=bcharge-1
            dE = E[b] - E[bm]
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], Tba[l, bm, b] * Tba[l, bm, b].conj()
                # p0=-1,p1=-1,eta=1
                lamb_m = dE - mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=1,p1=1,eta=-1
                lamb_p = -dE + mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # epsilon derivative
                phi_eps = diff_phi(lamb_p/Tr)
                kh.set_matrix_element_lpm_pauli(-gamma/2*(fermi_m+fermi_p), 0, bb, bb)
                kh.set_matrix_element_lpm_pauli(-2j*gamma*phi_eps, 5, bb, bb)
        for bp in statesdm[ccharge]:# ccharge=bcharge+1
            dE = E[bp] - E[b]
            for l in range(nleads):       
                mu, Tr, gamma = mulst[l], tlst[l], Tba[l, b, bp] * Tba[l, b, bp].conj() 
                # p0=1,p1=1,eta=1
                lamb_p = dE - mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=-1,p1=-1,eta=-1
                lamb_m = -dE + mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # epsilon derivative
                phi_eps = diff_phi(lamb_p/Tr)
                kh.set_matrix_element_lpm_pauli(-gamma/2*(fermi_p+fermi_m), 0, bb, bb)
                kh.set_matrix_element_lpm_pauli(-2j*gamma*phi_eps, 5, bb, bb)

    def generate_col_diag_kern_2nd_order_lpm(self, a0, charge):
        """Partly generates a column in the second order kernel for the diagonal density matrix :math:`W_{dd}^{(2)}`.
        Due to symmetries among the diagrammatic contributions for different matrix elements also contributions to
        other columns are generated. Assumes that the wide band limit is valid.

        Parameters
        ----------
        a0 : int
            initial state. Sets the column

        charge : int
            charge of state a0

        self.Wdd : ndarray
            (Modifies) diagonal lead-resolved kernel.

        self.Lpm : ndarray
            (Modifies) noise kernels.

        """
        # Evaluating the matrix elements requires summing over three pairs of states |a_+><a_-|,
        # two leads, two electron-hole indices and four propagator signs. This is done as follows:
        # one starts with a loop over the initial states. Then, nested loops iterate over the
        # possible intermediate states of the diagram (which is a fairly small number due
        # to several delta functions arising when evaluating the diagrams). Eventually, the final state
        # is reached and a contribution to the matrix element, specified by the initial and the final
        # state, is found (this only provides a contribution since the intermediate states were not
        # fully looped over yet.) Calculating the full matrix elements requires calling this function
        # for every column in the kernel.
        #
        # Symmetries between diagram contributions are used to avoid looping over some Keldysh- and
        # electron-hole indices. Specifically p1 = 1, p4 = 1 and eta1 = 1 are fixed.
        #
        # For the tunnel matrix elements the following rules apply:
        # 1) t = Tba(i->f) if charge_i < charge_f else t = Tba(f->i).conj()
        # 2) t_n = t_n.conj() if p_n == -1
        #
        # Variable names follow Leijnse et al PRB 78, 235424 (2008). Specifically:
        # - aNp/aNm: states
        # - eta : electron-hole index
        # - r : lead index
        # - p : Keldysh sign
        # - z : energy difference

        statesdm, Tba, E = self.si.statesdm, self.leads.Tba, self.qd.Ea
        tlst, mulst, dlst = self.leads.tlst, self.leads.mulst, self.leads.dlst
        kh = self.kernel_handler
        nleads = self.si.nleads
        b_and_R = self.Ozaki_poles_and_residues
        
        countingleads = self.funcp.countingleads

        t_cutoff1 = 0.0
        t_cutoff2 = 1e-10*max(tlst)
        t_cutoff3 = 1e-20*max(tlst)**2
        indx0 = self.si.get_ind_dm0(a0, a0, charge)
        #eps = 0.0
        h = self.lpm_h
        lpm_imaginary_2nd = self.lpm_imaginary_2nd
        
        # debug variable for mirror rule
        #mirrorp = 0.
        #mirrorm = 0.
        
        #eta0=1,p0=p3=1
        for r0, r1 in product(range(nleads), range(nleads)):
            r0_c, r1_c = int(r0 in countingleads), int(r1 in countingleads)
            T1, T2 = tlst[r0], tlst[r1]
            mu1, mu2 = mulst[r0], mulst[r1]
            D = np.abs(dlst[r0, 1]) + np.abs(dlst[r0, 0])
            #N1 = (N0, N0 + 1), a1- = a0
            for a1p in statesdm[charge+1]:
                t = Tba[r0, a1p, a0]
                if abs(t) == t_cutoff1:
                    continue
                indx1 = self.si.get_ind_dm0(a1p, a1p, charge + 1)
                E1 = E[a1p] - E[a0]
                #eta1 = 1
                #p1 = 1
                #N2 = (N0, N0 + 2), a2m = a0
                for a2p in statesdm[charge+2]:
                    #p2 = 1
                    t1 = t * Tba[r1, a2p, a1p]
                    if abs(t1) < t_cutoff2:
                        continue
                    E2 = E[a2p] - E[a0]
                    #N3 = (N0, N0 + 1 ), a3- = a2-
                    for a3p in statesdm[charge+1]:
                        #charge4 = charge + 0, a4 = a0
                        t2D = t1 * Tba[r1, a2p, a3p].conj() * Tba[r0, a3p, a0].conj()
                        t2X = t1 * Tba[r0, a2p, a3p].conj() * Tba[r1, a3p, a0].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 1, a0, charge)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,1,r0_c, r1_c,'d')
                            #mirrorp += tempD
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge + 1, a0, charge)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,1,r0_c, r1_c,'x')
                            #mirrorp += tempX
                    #p2 = -1
                    #N3 = ( N0 +1, N0 + 2), a3+ = a2+
                    for a3m in statesdm[charge+1]:
                        #charge4 = charge + 1, a4 = a3m
                        t2D = t1 * Tba[r1, a3m, a0].conj() * Tba[r0, a2p, a3m].conj()
                        t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a2p, a3m].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a2p, charge + 2, a3m, charge + 1)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a2p, charge + 2, a3m, charge + 1,1,1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a2p, charge + 2, a3m, charge + 1,1,1,1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a2p, charge + 2, a3m, charge + 1,1,1,1,-1,r0_c, r1_c,'d')
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a2p, charge + 2, a3m, charge+1)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a2p, charge + 2, a3m, charge + 1,1,1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a2p, charge + 2, a3m, charge + 1,1,1,1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a2p, charge + 2, a3m, charge + 1,1,1,1,-1,r0_c, r1_c,'x')
                #p1 = -1
                #N2 = ( N0 - 1, N0 + 1 ), a2+ = a1+
                for a2m in statesdm[charge-1]:
                    t1 = t * Tba[r1, a0, a2m]
                    if abs(t1) < t_cutoff2:
                        continue
                    E2 = E[a1p] - E[a2m]
                    #p2 = 1
                    #N3 = ( N0 - 1 , N0 ), a3- = a2-
                    for a3p in statesdm[charge]:
                        #charge4 = charge - 1, a0 = a2m
                        t2D = t1 * Tba[r1, a1p, a3p].conj() * Tba[r0, a3p, a2m].conj()
                        t2X = t1 * Tba[r0, a1p, a3p].conj() * Tba[r1, a3p, a2m].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(-1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge, a2m, charge - 1)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a3p, charge, a2m, charge - 1,1,-1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a3p, charge, a2m, charge - 1,1,1,-1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge, a2m, charge - 1,1,1,-1,1,r0_c, r1_c,'d')
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(-1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge, a2m, charge-1)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a3p, charge, a2m, charge - 1,1,-1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a3p, charge, a2m, charge - 1,1,1,-1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge, a2m, charge - 1,1,1,-1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0 , N0 + 1), a3+ = a2+
                    for a3m in statesdm[charge]:
                        #charge4 = charge + 0, a4 = a3m
                        t2D = t1 * Tba[r1, a3m, a2m].conj() * Tba[r0, a1p, a3m].conj()
                        t2X = t1 * Tba[r0, a3m, a2m].conj() * Tba[r1, a1p, a3m].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            #tempD =  t2D * integralD(-1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a1p, charge + 1, a3m, charge,1,-1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a1p, charge + 1, a3m, charge,1,1,-1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a1p, charge + 1, a3m, charge,1,1,-1,-1,r0_c, r1_c,'d')
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(-1, 1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a1p, charge + 1, a3m, charge,1,-1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a1p, charge + 1, a3m, charge,1,1,-1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a1p, charge + 1, a3m, charge,1,1,-1,-1,r0_c, r1_c,'x')
                #eta1 = -1
                #p1 = 1
                #N2 = (N0, N0), a2- = a0
                for a2p in statesdm[charge]:
                    E2 = E[a2p] - E[a0]
                    t1 = t * Tba[r1, a1p, a2p].conj()
                    if abs(t1) < t_cutoff2:
                        continue
                    #p2 = 1
                    #N3 = ( N0, N0 +1), a3- = a0
                    for a3p in statesdm[charge+1]:
                        #charge4 = charge, a4 = a0
                        t2D = t1 * Tba[r1, a3p, a2p] * Tba[r0, a3p, a0].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 1, a0, charge)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a3p, charge + 1, a0, charge,-1,1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a3p, charge + 1, a0, charge,1,-1,1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge + 1, a0, charge,1,-1,1,1,r0_c, r1_c,'d')
                    #N3 = (N0, N0-1), a3- = a0
                    for a3p in statesdm[charge-1]:
                        #charge4 = charge, a4 = a0
                        t2X = t1 * Tba[r0, a2p, a3p].conj() * Tba[r1, a0, a3p]
                        E3 = E[a3p] - E[a0]
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge - 1, a0, charge)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a3p, charge - 1, a0, charge,-1,1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a3p, charge - 1, a0, charge,1,-1,1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge - 1, a0, charge,1,-1,1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0-1, N0 ), a3+ = a2+
                    for a3m in statesdm[charge-1]:
                        #charge4 = charge - 1, a4 = a3m
                        t2D = t1 * Tba[r1, a0, a3m] * Tba[r0, a2p, a3m].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a2p, charge, a3m, charge - 1)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a2p, charge, a3m, charge - 1,-1,1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a2p, charge, a3m, charge - 1,1,-1,1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a2p, charge, a3m, charge - 1,1,-1,1,-1,r0_c, r1_c,'d')
                    #N3 = (N0 + 1, N0)
                    for a3m in statesdm[charge+1]:
                        #charge4 = charge + 1, a4 = a3m
                        t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a3m, a2p]
                        E3 = E[a2p] - E[a3m]
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a2p, charge, a3m, charge + 1)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a2p, charge, a3m, charge + 1,-1,1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a2p, charge, a3m, charge + 1,1,-1,1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a2p, charge, a3m, charge + 1,1,-1,1,-1,r0_c, r1_c,'x')
                #p1 = -1
                #N2 = ( N0 + 1  , N0 + 1), a2+ = a1+
                for a2m in statesdm[charge+1]:
                    E2 = E[a1p] - E[a2m]
                    t1 = t * Tba[r1, a2m, a0].conj()
                    #p2 = 1
                    # N3 = (N0 + 1, N0 + 2), a3- = a2-
                    for a3p in statesdm[charge+2]:
                        #charge4 = charge + 1, a4 = a2m
                        t2D = t1 * Tba[r1, a3p, a1p] * Tba[r0, a3p, a2m].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(-1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 2, a2m, charge + 1)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a3p, charge + 2, a2m, charge + 1,-1,-1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a3p, charge + 2, a2m, charge + 1,1,-1,-1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge + 2, a2m, charge + 1,1,-1,-1,1,r0_c, r1_c,'d')
                    #N3 = ( N0 + 1, N0 )
                    for a3p in statesdm[charge]:
                        #charge4 = charge + 1, a4 = a2m
                        t2X = t1 * Tba[r0, a1p, a3p].conj() * Tba[r1, a2m, a3p]
                        E3 = E[a3p] - E[a2m]
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(-1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge, a2m, charge + 1)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a3p, charge, a2m, charge + 1,1,-1,-1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge, a2m, charge + 1,1,-1,-1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0, N0+1), a3+ = a2+
                    for a3m in statesdm[charge]:
                        #charge4 = charge, a4 = a3m
                        t2D = t1 * Tba[r1, a2m, a3m] * Tba[r0, a1p, a3m].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            #tempD = t2D * integralD(-1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2D.imag)>t_cutoff3)
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                            kh.add_element_2nd_order_noise(tempD, indx0, indx1, a1p, charge + 1, a3m, charge,-1,-1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_dot(tempD_dotp, indx0, indx1, a1p, charge + 1, a3m, charge,1,-1,-1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a1p, charge + 1, a3m, charge,1,-1,-1,-1,r0_c, r1_c,'d')
                    #N3 = ( N0 + 2, N0 + 1 ), a3+ = a2+
                    for a3m in statesdm[charge+2]:
                        #charge4 = charge + 2, a4 = a3m
                        t2X = t1 * Tba[r0, a3m, a2m].conj() * Tba[r1, a3m, a1p]
                        E3 = E[a1p] - E[a3m]
                        if abs(t2X) > t_cutoff3:
                            #tempX = -t2X * integralX(-1, -1, E1, E2, E3, T1, T2, 1*mu1, mu2, D, b_and_R, abs(t2X.imag)>t_cutoff3)
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, 1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a1p, charge + 1, a3m, charge + 2)
                            kh.add_element_2nd_order_noise(tempX, indx0, indx1, a1p, charge + 1, a3m, charge + 2,-1,-1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_dot(tempX_dotp, indx0, indx1, a1p, charge + 1, a3m, charge + 2,1,-1,-1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a1p, charge + 1, a3m, charge + 2,1,-1,-1,-1,r0_c, r1_c,'x')
        #flip eta index!
        #p0=p3=1,eta0=-1 (other cases for p0,p3 still added through symmetry (eta screws it))
        for r0, r1 in product(range(nleads), range(nleads)):
            r0_c, r1_c = int(r0 in countingleads), int(r1 in countingleads)
            T1, T2 = tlst[r0], tlst[r1]
            mu1, mu2 = mulst[r0], mulst[r1]
            D = np.abs(dlst[r0, 1]) + np.abs(dlst[r0, 0])
            #N1 = (N0, N0 - 1), a1- = a0
            for a1p in statesdm[charge-1]:
                t = Tba[r0, a0, a1p]
                if abs(t) == t_cutoff1:
                    continue
                indx1 = self.si.get_ind_dm0(a1p, a1p, charge - 1)
                E1 = E[a1p] - E[a0]
                #eta1 = 1
                #p1 = 1
                #N2 = (N0, N0), a2m = a0
                for a2p in statesdm[charge]:
                    #p2 = 1
                    t1 = t * Tba[r1, a2p, a1p]
                    if abs(t1) < t_cutoff2:
                        continue
                    E2 = E[a2p] - E[a0]
                    #N3 = (N0, N0 - 1 ), a3- = a2-
                    for a3p in statesdm[charge-1]:
                        #charge4 = charge + 0, a4 = a0
                        t2D = t1 * Tba[r1, a2p, a3p].conj() * Tba[r0, a0, a3p].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 1, a0, charge)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a3p, charge - 1, a0, charge,1,1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a3p, charge - 1, a0, charge,-1,1,1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge - 1, a0, charge,-1,1,1,1,r0_c, r1_c,'d')
                    #N3 = (N0, N0 + 1 ), a3- = a2-
                    for a3p in statesdm[charge+1]:
                        #charge4 = charge + 0, a4 = a0
                        t2X = t1 * Tba[r0, a3p, a2p].conj() * Tba[r1, a3p, a0].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge + 1, a0, charge)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a3p, charge + 1, a0, charge,1,1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a3p, charge + 1, a0, charge,-1,1,1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge + 1, a0, charge,-1,1,1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0 + 1 , N0)
                    for a3m in statesdm[charge+1]:
                        #charge4 = charge + 1, a4 = a3m
                        t2D = t1 * Tba[r1, a3m, a0].conj() * Tba[r0, a3m, a2p].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a2p, charge + 2, a3m, charge + 1)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a2p, charge, a3m, charge + 1,1,1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a2p, charge, a3m, charge + 1,-1,1,1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a2p, charge, a3m, charge + 1,-1,1,1,-1,r0_c, r1_c,'d')
                    #N3 = ( N0 - 1, N0)
                    for a3m in statesdm[charge-1]:
                        #charge4 = charge - 1, a4 = a3m
                        t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a2p, a3m].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a2p, charge + 2, a3m, charge+1)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a2p, charge, a3m, charge - 1,1,1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a2p, charge, a3m, charge - 1,-1,1,1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a2p, charge, a3m, charge - 1,-1,1,1,-1,r0_c, r1_c,'x')
                #p1 = -1
                #N2 = ( N0 - 1, N0 - 1 ), a2+ = a1+
                for a2m in statesdm[charge-1]:
                    t1 = t * Tba[r1, a0, a2m]
                    if abs(t1) < t_cutoff2:
                        continue
                    E2 = E[a1p] - E[a2m]
                    #p2 = 1
                    #N3 = ( N0 - 1 , N0 - 2), a3- = a2-
                    for a3p in statesdm[charge-2]:
                        #charge4 = charge - 1, a4 = a2m
                        t2D = t1 * Tba[r1, a1p, a3p].conj() * Tba[r0, a2m, a3p].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge, a2m, charge - 1)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a3p, charge - 2, a2m, charge - 1,1,-1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a3p, charge - 2, a2m, charge - 1,-1,1,-1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge - 2, a2m, charge - 1,-1,1,-1,1,r0_c, r1_c,'d')
                    #N3 = ( N0 - 1 , N0 ), a3- = a2-
                    for a3p in statesdm[charge]:
                        #charge4 = charge - 1, a4 = a2m
                        t2X = t1 * Tba[r0, a3p, a1p].conj() * Tba[r1, a3p, a2m].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge, a2m, charge-1)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a3p, charge, a2m, charge - 1,1,-1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a3p, charge, a2m, charge - 1,-1,1,-1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge, a2m, charge - 1,-1,1,-1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0 , N0 - 1), a3+ = a2+
                    for a3m in statesdm[charge]:
                        #charge4 = charge + 0, a4 = a3m
                        t2D = t1 * Tba[r1, a3m, a2m].conj() * Tba[r0, a3m, a1p].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a1p, charge - 1, a3m, charge,1,-1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a1p, charge - 1, a3m, charge,-1,1,-1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a1p, charge - 1, a3m, charge,-1,1,-1,-1,r0_c, r1_c,'d')
                    #N3 = ( N0 - 2 , N0 - 1), a3+ = a2+
                    for a3m in statesdm[charge-2]:
                        #charge4 = charge + 0, a4 = a3m
                        t2X = t1 * Tba[r0, a2m, a3m].conj() * Tba[r1, a1p, a3m].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, 1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, 1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a1p, charge - 1, a3m, charge - 2,1,-1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a1p, charge - 1, a3m, charge - 2,-1,1,-1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a1p, charge - 1, a3m, charge - 2,-1,1,-1,-1,r0_c, r1_c,'x')
                #eta1 = -1
                #p1 = 1
                #N2 = (N0, N0 - 2), a2- = a0
                for a2p in statesdm[charge-2]:
                    E2 = E[a2p] - E[a0]
                    t1 = t * Tba[r1, a1p, a2p]
                    if abs(t1) < t_cutoff2:
                        continue
                    #p2 = 1
                    #N3 = ( N0, N0 -1), a3- = a0
                    for a3p in statesdm[charge-1]:
                        #charge4 = charge, a4 = a0
                        t2D = t1 * Tba[r1, a3p, a2p].conj() * Tba[r0, a0, a3p].conj()
                        t2X = t1 * Tba[r0, a3p, a2p].conj() * Tba[r1, a0, a3p].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 1, a0, charge)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a3p, charge - 1, a0, charge,-1,1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a3p, charge - 1, a0, charge,-1,-1,1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge - 1, a0, charge,-1,-1,1,1,r0_c, r1_c,'d')
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge - 1, a0, charge)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a3p, charge - 1, a0, charge,-1,1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a3p, charge - 1, a0, charge,-1,-1,1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge - 1, a0, charge,-1,-1,1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0-1, N0 - 2 ), a3+ = a2+
                    for a3m in statesdm[charge-1]:
                        #charge4 = charge - 1, a4 = a3m
                        t2D = t1 * Tba[r1, a0, a3m].conj() * Tba[r0, a3m, a2p].conj()
                        t2X = t1 * Tba[r0, a0, a3m].conj() * Tba[r1, a3m, a2p].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, 1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a2p, charge, a3m, charge - 1)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a2p, charge - 2, a3m, charge - 1,-1,1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a2p, charge - 2, a3m, charge - 1,-1,-1,1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a2p, charge - 2, a3m, charge - 1,-1,-1,1,-1,r0_c, r1_c,'d')
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, 1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a2p, charge, a3m, charge + 1)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a2p, charge - 2, a3m, charge - 1,-1,1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a2p, charge - 2, a3m, charge - 1,-1,-1,1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a2p, charge - 2, a3m, charge - 1,-1,-1,1,-1,r0_c, r1_c,'x')
                #p1 = -1
                #N2 = ( N0 + 1  , N0 - 1), a2+ = a1+
                for a2m in statesdm[charge+1]:
                    E2 = E[a1p] - E[a2m]
                    t1 = t * Tba[r1, a2m, a0]
                    #p2 = 1
                    # N3 = (N0 + 1, N0), a3- = a2-
                    for a3p in statesdm[charge]:
                        #charge4 = charge + 1, a4 = a2m
                        t2D = t1 * Tba[r1, a3p, a1p].conj() * Tba[r0, a2m, a3p].conj()
                        t2X = t1 * Tba[r0, a3p, a1p].conj() * Tba[r1, a2m, a3p].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 2, a2m, charge + 1)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,-1,1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,-1,1,r0_c, r1_c,'d')
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge, a2m, charge + 1)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,-1,1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a3p, charge, a2m, charge + 1,-1,-1,-1,1,r0_c, r1_c,'x')
                    #p2 = -1
                    #N3 = ( N0, N0-1), a3+ = a2+
                    for a3m in statesdm[charge]:
                        #charge4 = charge, a4 = a3m
                        t2D = t1 * Tba[r1, a2m, a3m].conj() * Tba[r0, a3m, a1p].conj()
                        t2X = t1 * Tba[r0, a2m, a3m].conj() * Tba[r1, a3m, a1p].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempD_dotp = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempD_dotm = t2D * integralD_lpm(lpm_imaginary_2nd, -1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                            kh.add_element_2nd_order_noise_m(tempD, indx0, indx1, a1p, charge - 1, a3m, charge,-1,-1,-1,r0_c, r1_c,'d')
#                            kh.add_element_2nd_order_noise_m_dot(tempD_dotp, indx0, indx1, a1p, charge + 1, a3m, charge,-1,-1,-1,-1,r0_c, r1_c,'d')
                            kh.add_element_2nd_order_noise_dot_2((tempD_dotp-tempD)/h, indx0, indx1, a1p, charge + 1, a3m, charge,-1,-1,-1,-1,r0_c, r1_c,'d')
                            #mirrorm += tempD
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, True)
                            tempX_dotp = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, -1, E1-h, E2-h, E3-h, T1, T2, mu1, mu2, D, b_and_R, True)
#                            #tempX_dotm = -t2X * integralX_lpm(lpm_imaginary_2nd, -1, -1, -1, E1+h, E2+h, E3+h, T1, T2, mu1, mu2, D, b_and_R, True)
                            #kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a1p, charge + 1, a3m, charge + 2)
                            kh.add_element_2nd_order_noise_m(tempX, indx0, indx1, a1p, charge - 1, a3m, charge,-1,-1,-1,r0_c, r1_c,'x')
#                            kh.add_element_2nd_order_noise_m_dot(tempX_dotp, indx0, indx1, a1p, charge - 1, a3m, charge,-1,-1,-1,-1,r0_c, r1_c,'x')
                            kh.add_element_2nd_order_noise_dot_2((tempX_dotp-tempX)/h, indx0, indx1, a1p, charge - 1, a3m, charge,-1,-1,-1,-1,r0_c, r1_c,'x')
                            #mirrorm += tempX
                            
        #print(mirrorp,mirrorm)
