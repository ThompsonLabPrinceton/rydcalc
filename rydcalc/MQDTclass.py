import numpy as np
import scipy as scp
import sys
from .constants import *

class mqdt_class:
    def __init__(self, channels, rot_order,
                 rot_angles, eig_defects, nulims, Uiabar, atom,odd_powers=False,e_scale_low_lim = False):
        self.channels = channels
        self.numchannels = len(channels)
        self.rotchannels = rot_order
        self.rotangles = rot_angles
        self.muvec = eig_defects

        #self.Vrot = self.Vrotfunc()
        self.Uiabar = Uiabar#self.Uiabarfunc()

        #self.Uia = np.dot(self.Uiabar, self.Vrot)

        self.ionizationlimits_invcm=[]
        for ch in channels:
            self.ionizationlimits_invcm.append(0.01*ch.core.Ei_Hz/cs.c)

        [self.nulima,self.nulimb] = nulims
        self.i = atom.I
        self.RydConst_invcm = 0.01*atom.RydConstHz / cs.c

        self.mateps = sys.float_info.epsilon

        self.atom = atom

        self.odd_powers = odd_powers #odd powers in quantum defects expansion for relativistic corrections
        self.e_scale_low_lim = e_scale_low_lim # if True, energy-expansion in terms of nu with respect to lowest ionization limit, if false energy-expansion in terms of nu with respect to series threshold 

    def getCofactorMatrix(self,A):
        ''' Returns the cofactor matrix of matrix A using SVD. See e.g.
        Linear Algebra Its Appl. 283 (998) 5-64.
        '''
        U, sigma, Vt = np.linalg.svd(A)
        N = len(sigma)
        g = np.tile(sigma, N)
        g[::(N + 1)] = 1
        G = np.diag(np.product(np.reshape(g, (N, N)), 1))
        return np.linalg.det(U)*np.linalg.det(Vt.T)*U @ G @ Vt

    def sinfunc(self, x, A, phi, b):
        return A * np.sin(2 * np.pi * x + phi) + b

    def customhalley(self,fwithdiff, x0, eps=10 ** (-16), tol=10 ** (-20), maxfunctioncalls=500):

        ''' Root finding algorithm based on Halley's method. It iteratively approximates a root of f(x) from an initial guess x0
        using f(x_n), f'(x_n) and f"(x_n) by

        x_(n+1) = x_n - ( 2*f(x_n)*f'(x_n) ) / (2*(f'(x_n))**2 - f(x_n)*f"(x_n)).

        If f"() == False, the function uses Newton's method

        x_(n+1) = x_n - f(x_n) / f'(x_n).

        '''

        x = x0

        functioncalls = 1
        while True:
            (fx, fpx, fpx2) = fwithdiff(x)

            if fpx2 == False:
                xnew = x - fx / fpx
            else:
                xnew = x - (2.0 * fx * fpx) / (2.0 * fpx ** 2 - fx * fpx2)

            if fx < tol and abs(xnew - x) <= eps:
                solution = [x, True, functioncalls]
                break

            if functioncalls > maxfunctioncalls:
                solution = [x, False, functioncalls]
                break

            x = xnew
            functioncalls += 1

        return solution

    def nux(self, Ia, Ib, nub):
        ''' Converts effective principal quantum nub with respect to threshold Ib
        to effective principal quantum number nua.
        '''
        return ((Ia - Ib) / self.RydConst_invcm + 1 / nub ** 2) ** (-1 / 2)

    def rotmat(self, i, j, thetaij):
        ''' Returns rotation matrix around channels i and j with angle thetaij.
        '''
        rotmat = np.identity(self.numchannels)
        rotmat[i - 1, i - 1] = np.cos(thetaij)
        rotmat[j - 1, j - 1] = np.cos(thetaij)
        rotmat[i - 1, j - 1] = -np.sin(thetaij)
        rotmat[j - 1, i - 1] = np.sin(thetaij)
        return rotmat

    def Vrotfunc(self,nub):
        ''' Performs consecutive rotations specified in the MQDT model in rot_order with angles specified in rot_angles.
        '''
        th=self.rotangles[0][0]

        for k,ii in enumerate(self.rotangles[0][1:],1):
            th+=ii/(nub**(2*k))

        Vrot = self.rotmat(self.rotchannels[0][0], self.rotchannels[0][1], th)

        for j,i in enumerate(self.rotchannels[1:],1):

            th = self.rotangles[j][0]
            for k,ii in enumerate(self.rotangles[j][1:],1):
                th += ii / (nub ** (2 * k))

            Vrot = np.dot(Vrot, self.rotmat(i[0], i[1], th))

        return Vrot

    def nufunc(self, nu, mu):
        ''' Generates matrix sin(pi*(nu_i+mu_a)) from vectors nu and mu.
        '''
        numat = np.zeros((self.numchannels, self.numchannels))
        for i in range(self.numchannels):
            for j in range(self.numchannels):
                numat[i, j] = np.sin(np.pi * (nu[i] + mu[j]))

        return numat

    def nudifffunc(self,nu,nudiff,mu,mudiff):
        ''' Generates derivative of matrix sin(pi*(nu_i+mu_a)), from vectors nu and mu and their derivatives.
        '''
        nudiffmat = np.zeros((self.numchannels, self.numchannels))

        for i in range(self.numchannels):
            for j in range(self.numchannels):
                nudiffmat[i, j] = np.pi*np.cos(np.pi * (nu[i] + mu[j])) * (nudiff[i] + mudiff[j])

        return nudiffmat

    def nudiff2func(self,nu,nudiff,nudiff2,mu,mudiff,mudiff2):
        ''' Generates second derivative of matrix sin(pi*(nu_i+mu_a)), from vectors nu and mu and their first and second derivatives.
        '''
        nudiff2mat = np.zeros((self.numchannels, self.numchannels))

        for i in range(self.numchannels):
            for j in range(self.numchannels):
                nudiff2mat[i, j] = -(np.pi**2)*np.sin(np.pi*(nu[i]+mu[j]))*(mudiff[j]+nudiff[i])**2+np.pi*np.cos(np.pi * (nu[i] + mu[j]))*(mudiff2[j]+nudiff2[i])

        return nudiff2mat

    def mqdtmodel(self, nuvec):
        ''' For a given set of effective quantum numbers with respect to two ionization limits a and b, this function returns the value of

        det (U_ia*sin(pi*(nu_i+mu_a)))

        '''
        [nua, nub] = nuvec

        nu=[]
        for i in np.arange(self.numchannels):
            if i in self.nulima:
                nu.append(nua)
            elif i in self.nulimb:
                nu.append(nub)
            else:
                nu.append(self.nux(self.ionizationlimits_invcm[i],self.ionizationlimits_invcm[self.nulima[0]],nua))

        mu = []

        for i in np.arange(self.numchannels):
            mue=0


            if self.odd_powers == True:
                for k,muk in enumerate(self.muvec[i]):
                    if k==0:
                        mue += muk
                    else:
                        mue += muk/self.nux(self.ionizationlimits_invcm[i],self.ionizationlimits_invcm[self.nulima[0]],nua)**(k)
            elif self.e_scale_low_lim == True:
                for k,muk in enumerate(self.muvec[i]):
                    if k==0:
                        mue += muk
                    else:
                        mue += muk/nub**(k*2)
            else:
                for k,muk in enumerate(self.muvec[i]):
                    if k==0:
                        mue += muk
                    else:
                        mue += muk/self.nux(self.ionizationlimits_invcm[i],self.ionizationlimits_invcm[self.nulima[0]],nua)**(k*2)

            mu.append(mue)

        self.Uia = np.dot(self.Uiabar, self.Vrotfunc(nub))

        nmat = self.nufunc(nu, mu)

        return np.linalg.det(np.multiply(self.Uia, nmat))


    def diff2mqdtmodel(self,nub):

        '''
        Calculates the determinant
         
         det|U_{ialpha} sin(pi(mu_alpha(E)+nu_i(E)))|
         
         along Ia-Rk/nua**2==Ib-Rk/nub**2, and it's first two derivatives. The derivates of the determinant are obtained by application of Jacobi's formula
         
         (1) d/dt (det A(t)) = tr (adj(A(t)) dA(t)/dt).
         
         and further for invertible matrices A(t), 
         
         (2) d/dt (det A(t)) = det(A(t))*tr(inv(A(t))*dA(t)/dt)
         
         (3) d^2 / dt^2 (det A(t)) = det (A(t)) *(tr(inv(A(t))*(dA/dt))^2 - tr((inv(A(t))*(dA/dt))^2-inv(A(t))*(d^2A/dt^2))) )
         
        '''

        nu = []
        nudiff = []
        nudiff2 =[]

        for i in np.arange(self.numchannels):
            if i in self.nulimb:
                nu.append(nub)
                nudiff.append(1)
                nudiff2.append(0)
            else:
                nu.append(self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub))
                nudiff.append(1 / ((nub ** 3) * ((1 / nub ** 2) + (self.ionizationlimits_invcm[i] - self.ionizationlimits_invcm[self.nulimb[0]]) / self.RydConst_invcm) ** (3 / 2)))
                nudiff2.append((3/((nub**6)*(1/nub**2 + (self.ionizationlimits_invcm[i] - self.ionizationlimits_invcm[self.nulimb[0]]) / self.RydConst_invcm)**(5/2))-3/((nub**4)*(1/nub**2 + (self.ionizationlimits_invcm[i] - self.ionizationlimits_invcm[self.nulimb[0]]) / self.RydConst_invcm)**(3/2))))

        mu = []
        mudiff = []
        mudiff2 = []

        for i in np.arange(self.numchannels):
            mue = 0
            muediff = 0
            muediff2 = 0

            for k,muk in enumerate(self.muvec[i]):
                if k == 0:
                    mue += muk
                else:
                    if self.odd_powers == True:
                        mue += muk / self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub) ** (k)
                        muediff += (- muk * (k) *(1/nub**2+(self.ionizationlimits_invcm[i]-self.ionizationlimits_invcm[self.nulimb[0]])/self.RydConst_invcm)**(k/2-1))/nub**3
                        muediff2 +=  (2*(k/2-1)*k*muk*(1/nub**2+(self.ionizationlimits_invcm[i]-self.ionizationlimits_invcm[self.nulimb[0]])/self.RydConst_invcm)**(k/2-2))/nub**6 + (3*k*muk*(1/nub**2+(self.ionizationlimits_invcm[i]-self.ionizationlimits_invcm[self.nulimb[0]])/self.RydConst_invcm)**(k/2-1))/nub**4
                    elif self.e_scale_low_lim == True:
                        mue += muk / nub ** (k * 2)
                        muediff += -2*k*muk/nub**(2*k+1)
                        muediff2 += 2*(1 + 2*k)*k*muk/nub**(2 + 2*k)
                    else:
                        mue += muk / self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub) ** (k * 2)
                        muediff += (- muk * (2*k) *(1/nub**2+(self.ionizationlimits_invcm[i]-self.ionizationlimits_invcm[self.nulimb[0]])/self.RydConst_invcm)**(k-1))/nub**3
                        muediff2 +=  (4*(k-1)*k*muk*(1/nub**2+(self.ionizationlimits_invcm[i]-self.ionizationlimits_invcm[self.nulimb[0]])/self.RydConst_invcm)**(k-2))/nub**6 + (6*k*muk*(1/nub**2+(self.ionizationlimits_invcm[i]-self.ionizationlimits_invcm[self.nulimb[0]])/self.RydConst_invcm)**(k-1))/nub**4

            mu.append(mue)
            mudiff.append(muediff)
            mudiff2.append(muediff2)

        self.Uia = np.dot(self.Uiabar, self.Vrotfunc(nub))

        ndiffmat = self.nudifffunc(nu,nudiff, mu,mudiff)
        ndiff2mat = self.nudiff2func(nu,nudiff,nudiff2,mu,mudiff,mudiff2)
        numat = self.nufunc(nu, mu)

        Umat = np.multiply(self.Uia, numat)
        detUmat = np.linalg.det(Umat)

        diffUmat = np.multiply(self.Uia, ndiffmat)

        # Check if matrix is singular. If not, use Eq. (2) and (3), if (close to) singular use Eq. (1) and don't return second derivative
        if np.linalg.cond(Umat) < 1 / self.mateps:
            invUmat = np.linalg.inv(Umat)
            diff2Umat = np.multiply(self.Uia, ndiff2mat)
            inVdiff = np.dot(invUmat, diffUmat)
            trinVdiff = np.trace(inVdiff)
            return [detUmat, detUmat*trinVdiff, detUmat*(trinVdiff**2 - np.trace(np.dot(inVdiff,inVdiff)-np.dot(invUmat,diff2Umat)))]
        else:
            adjUmat = self.getCofactorMatrix(Umat).T
            return [detUmat, np.trace(np.dot(adjUmat, diffUmat)), False]

    def lufano(self, na):

        if len(self.nulimb)==1:

            nuavec = [0, 0.5]

            fitvals = np.array([])

            for i in nuavec:
                fitvals = np.append(fitvals, self.mqdtmodel([na,i]))

            y0 = fitvals[0]
            y12 = fitvals[1]

            phi = np.arctan(y0 / y12)

            return [np.mod(1 - phi / np.pi, 1)]

        elif len(self.nulimb)==2:
            nubvec = np.arange(0.1, 1.1, 1 / 5)

            fitvals = np.array([])

            for i in nubvec:
                fitvals = np.append(fitvals, self.mqdtmodel([na,i]))

            params, params_covariance = scp.optimize.curve_fit(self.sinfunc, nubvec, fitvals, p0=[-0.0001, 1.24, 0.0001])

            A = params[0]
            phi = params[1]
            b = params[2]

            root = [np.mod((-phi - np.arcsin(b / A)) / (2 * np.pi), 1),np.mod((-phi + np.pi + np.arcsin(b / A)) / (2 * np.pi), 1)]

            return root
        else:
            print("Implement three channels converging to Ia")


    def boundstatesminimizer(self,nub):
        # solve for determinant along function Ea = Eb = Ia - R/nua**2 = Ib - R/nub**2
        [A,B,C]=self.diff2mqdtmodel(nub)

        return A,B,C

    def boundstates(self,nubguess,accuracy = 10):
        ''' calculates the theoretical bound state of the mqdt model close to guess of an effective quantum number with respect to the higher (Ia,Ib) ionization limit.
        Currently uses Halley's method (like Newton's method, but with additional information from second derivative) without brackets around the root. 
        Accuracy 8 corresponds to 1 MHz at nub = 4, 8 kHz at nub = 20, and 0.065 kHz at nub = 100.
        For bracketed root search use scp.optimize.brentq'''

        sol = self.customhalley(self.boundstatesminimizer, x0=nubguess,eps=10**(-10),tol=10**(-accuracy),maxfunctioncalls=100000)

        if sol[1]==True:
            return np.round(sol[0], decimals=accuracy )
        else:
            #print("I did not converge")
            return 1

        #return np.round(sol, decimals=accuracy - 1)


    def boundstatesinrange(self,range,accuracy=10):
        '''Finds bound states in given range with given accuracy in nub. For nub converging to first ionization limit.
        '''

        nutheorb = []
        nutheora = []

        for nubguess in np.arange(range[0], range[1], 0.01):
            nutheorb.append(np.round(self.boundstates(nubguess), decimals=accuracy))

        values = np.sort(nutheorb)
        diffs = np.diff(values)
        # Where the difference exceeds the threshold, we split
        split_indices = np.where(diffs > 1e-8)[0] + 1
        split = np.split(values, split_indices)
        nutheorb = np.array([g.mean() for g in split])
        #self.channels[np.argmax(np.array(self.channelcontributions(item)[0])**2)].l
    
        nutheorb = [item for item in nutheorb if (maxind := np.argmax(np.array(self.channelcontributions(item)[0])**2)) is not None and self.nux(self.ionizationlimits_invcm[maxind],self.ionizationlimits_invcm[self.nulimb[0]],item) >= self.channels[np.argmax(np.array(self.channelcontributions(item)[0])**2)].l] # filter out bound states with l > nu_i,



        for i in nutheorb:
            nutheora.append(self.nux(self.ionizationlimits_invcm[self.nulima[0]], self.ionizationlimits_invcm[self.nulimb[0]], i))

        return [np.sort(nutheora), np.sort(nutheorb)]

    def channelcontributions(self,nub):

        ''' Calculates the channel contributions A_i for a given bound state with effective principal quantum number nub,
        using Eq. (5) and Eq. (24) from 0. ROBAUX, and M. AYMAR, Comp. Phys. Commun. 25, 223—236 (1982).
        The A_i's are normalized following Lee and Lu (Eq. (A5) from , Phys. Rev. A 8, 1241 (1973))
        '''

        def Uiafunc(nub):
            return np.dot(self.Uiabar, self.Vrotfunc(nub))

        self.Uia = Uiafunc(nub)

        nu = []
        for i in np.arange(self.numchannels):
            if i in self.nulimb:
                nu.append(nub)
            else:
                nu.append(self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub))

        mu = []
        for i in np.arange(self.numchannels):
            mue = 0
            for k,muk in enumerate(self.muvec[i]):
                if k == 0:
                    mue += muk
                else:
                    if self.odd_powers == True:
                        mue += muk / self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub) ** (k)
                    elif self.e_scale_low_lim == True:
                        mue += muk / nub ** (k * 2)
                    else:
                        mue += muk / self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub) ** (k * 2)
            mu.append(mue)

        dmu = []

        for i in np.arange(self.numchannels):
            dmue = 0
            for k,muk in enumerate(self.muvec[i]):
                if k != 0:
                    if self.odd_powers == True:
                        dmue += - k * muk / self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub) ** ((k/2-1) * 2)
                    elif self.e_scale_low_lim == True:
                        dmue += - 2*k * muk / nub ** ((k-1) * 2)
                    else:
                        dmue += - 2*k * muk / self.nux(self.ionizationlimits_invcm[i], self.ionizationlimits_invcm[self.nulimb[0]], nub) ** ((k-1) * 2)
            dmu.append(dmue)

        eps = 10**(-6)
        dUia_dnu = (Uiafunc(nub+eps/2)-Uiafunc(nub-eps/2)) / eps
        dUia_dE = (nub**3)*dUia_dnu

        nmat = self.nufunc(nu, mu)
        Fialpha = np.multiply(self.Uia, nmat)

        cofacjalpha = self.getCofactorMatrix(Fialpha)

        Aalpha = []
        # Eq. (5) from 0. ROBAUX, and M. AYMAR, Comp. Phys. Commun. 25, 223—236 (1982)
        for i in np.arange(self.numchannels):
                Aalpha.append(cofacjalpha[0, i] / np.sqrt(np.sum(cofacjalpha[0, :] ** 2)))

        Ai = np.array([])

        # Eq. (24) from 0. ROBAUX, and M. AYMAR, Comp. Phys. Commun. 25, 223—236 (1982)
        for i in np.arange(self.numchannels):
            sumalpha = 0
            for alpha in np.arange(self.numchannels):
                sumalpha += (-1)**(self.channels[i].l+1) * nu[i]**(3 / 2) * self.Uia[i, alpha] * np.cos(np.pi * (nu[i] + mu[alpha])) * Aalpha[alpha]

            Ai=np.append(Ai,sumalpha)

        # simple normalization of channels contributions to np.sum(Ai ** 2) == 1, neglects energy dependence of mu and Uia
        #Ai_norm = Ai / np.sqrt(np.sum(Ai ** 2))


        # Eq. (A5) from Lee and Lu, Phys. Rev. A 8, 1241 (1973)
        Nsq = 0

        for i in np.arange(self.numchannels):
            Ni = 0
            for alpha in np.arange(self.numchannels):

                Ni+=(self.Uia[i,alpha]*np.cos(np.pi*(nu[i]+mu[alpha]))*Aalpha[alpha])
            Nsq += (nu[i]**3)*Ni**2

        for alpha in np.arange(self.numchannels):
            Nsq += dmu[alpha]*Aalpha[alpha]**2

        for i in np.arange(self.numchannels):
            for alpha in np.arange(self.numchannels):
                for beta in np.arange(self.numchannels):
                    Nsq += (1/np.pi)*dUia_dE[i,alpha]*self.Uia[i,beta]*np.sin(np.pi*(mu[alpha]-mu[beta]))*Aalpha[alpha]*Aalpha[beta]

        Ai_norm = Ai / np.sqrt(Nsq)

        # channel contributions in alpha channels

        Aalpha_norm = np.array([])

        for alpha in np.arange(self.numchannels):
            Aalpha = 0
            for i in np.arange(self.numchannels):
                Aalpha += self.Uiabar[i, alpha] * Ai_norm[i]

            Aalpha_norm = np.append(Aalpha_norm, Aalpha)

        return [Ai_norm,Aalpha_norm]


class mqdt_class_rydberg_ritz(mqdt_class):
    ''' Adaptation of the MQDT class for single-channel channel quantum defect theory.
    '''

    def __init__(self, channels,deltas,atom,HFlimit = None,odd_powers = False):
        self.channels = [channels]
        self.deltas=deltas
        self.HFlimit = HFlimit
        self.odd_powers = odd_powers    

        self.atom = atom

        self.nulima = [0]
        self.nulimb = [0]


        self.ionizationlimits_invcm=[]

        self.ionizationlimits_invcm.append(0.01 * channels.core.Ei_Hz / cs.c)

        if HFlimit == "upper":
            self.nulima = [1]
            self.nulimb = [0]
            self.ionizationlimits_invcm.append(0.01 *( channels.core.Ei_Hz  - self.atom.ion_hyperfine_6s_Hz)/ cs.c)
        elif HFlimit == "lower":
            self.nulima = [0]
            self.nulimb = [1]
            self.ionizationlimits_invcm.append(0.01 * (channels.core.Ei_Hz + self.atom.ion_hyperfine_6s_Hz) / cs.c)

        self.RydConst_invcm = 0.01*atom.RydConstHz / cs.c



    def boundstates(self, nubguess,accuracy=10):

        if self.HFlimit == "upper" or self.HFlimit == None:
            #searchrange = [np.ceil(nubguess - 1.5), np.floor(nubguess + 1.5)]
            searchrange = [nubguess,nubguess]
        elif self.HFlimit == "lower":
            nuaguess = self.nux(0,0.01*self.atom.ion_hyperfine_6s_Hz/ cs.c,nubguess)
            #print(nuaguess)
            #searchrange = [np.ceil(nuaguess - 1.5), np.floor(nuaguess + 1.5)]
            searchrange = [ nuaguess,nuaguess]
        else:
            print("Unspecified HF limit!")

        approxdelta = 0
        approxnu = (searchrange[0]+searchrange[1])/2

        if self.odd_powers == True:
            for k,di in enumerate(self.deltas):
                approxdelta += di / (approxnu) ** (k)
        else:
            for k,di in enumerate(self.deltas):
                approxdelta += di / (approxnu) ** (2 * k)

        n=np.round(approxnu+approxdelta)

        nutheor=n

        if self.odd_powers == True:
            for k,di in enumerate(self.deltas):
                nutheor += -  di / (n) ** (k)
        else:   
            for k,di in enumerate(self.deltas):
                nutheor += -  di / (n-self.deltas[0]) ** (2 * k)

        if self.HFlimit == "upper" or self.HFlimit == None:
            return np.round(nutheor, decimals=accuracy-1)
        elif self.HFlimit == "lower":
            return np.round(self.nux(self.ionizationlimits_invcm[self.nulimb[0]],self.ionizationlimits_invcm[self.nulima[0]],nutheor), decimals=accuracy-1)


    def channelcontributions(self, nub):
        ''' Channel contributions for single channel model set to be 1.0.
        '''
        Ai = [1.0]
        Aalpha = [1.0]
        return [Ai,Aalpha]
    
class mqdt_class_wrapper(mqdt_class):
    ''' Wrapper class for the MQDT class. Allows for simultaneous treatment of un-coupled channels with idential quantum numbers.    '''

    def __init__(self, classlist):

        self.classlist = classlist

        # Check if all classes have the same atom attribute
        if not all(cls.atom == classlist[0].atom for cls in classlist):
            raise ValueError("All classes in classlist must have the same atom attribute.")

        self.atom = classlist[0].atom

        self.channels = []

        for i in self.classlist:
            self.channels.extend(i.channels)

        self.numchannels = len(self.channels)

        self.nulima = [0]
        self.nulimb = [0]

        self.RydConst_invcm = 0.01*self.atom.RydConstHz / cs.c

        self.ionizationlimits_invcm=[]
        for i in self.classlist:
            self.ionizationlimits_invcm.extend(i.ionizationlimits_invcm)
                

    def boundstates(self, nubguess,accuracy=10):

        ''' Returns the closest bound state to nubguess for all channels in the class list. 
        '''
        channelnub = [] 

        for i in self.classlist:
            channelnub.append(i.boundstates(nubguess, accuracy))

        channelsel = np.argmin(np.abs(np.array(channelnub)-nubguess)) # finds the channel with the closest bound state to nubguess

        return np.round(channelnub[channelsel], decimals=accuracy )
    
    def boundstatesinrange(self, range, accuracy=10):

        ''' Returns the bound state in range for all channels in the class list. 
        '''
        nua = np.array([])
        nub = np.array([])

        for i in self.classlist:
            [nuachan,nubchan] = i.boundstatesinrange(range, accuracy)
            nua = np.append(nua,nuachan)
            nub = np.append(nub,nubchan)

        return [nua,nub]
    


    def channelcontributions(self, nub):
        ''' Channel contributions for wrapped MQDT classes. 
        '''

        
        channelnub = [] 

        for i in self.classlist:
            channelnub.append(i.boundstates(nub))

        channelsel = np.argmin(np.abs(np.array(channelnub)-nub)) # finds the channel with the closest bound state to nubguess

        Ai = []
        Aalpha = []

        for count,i in enumerate(self.classlist):
            if count == channelsel:
                [Aichan,Aalphachan] = i.channelcontributions(nub)
                Ai.extend(Aichan)
                Aalpha.extend(Aalphachan)
            else:
                Ai.extend([0]*len(i.channels))
                Aalpha.extend([0]*len(i.channels))


        return [Ai,Aalpha]
    

