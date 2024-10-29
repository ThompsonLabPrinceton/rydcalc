from .alkaline import *
from .MQDTclass import *
from .utils import model_params
import json

import csv, importlib.resources

class Ytterbium174(AlkalineAtom):

    I = 0
    Z = 70

    name = '174Yb'

    mass = 173.9388664

    model_pot = model_potential(0, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [1e-3] * 4,
                                Z, include_so=True, use_model=False)

    gI = 0

    Elim_cm = 50443.070417
    Elim_THz = Elim_cm * (cs.c * 100 * 1e-12)

    RydConstHz = cs.physical_constants["Rydberg constant times c in Hz"][0] * \
                 (1 - cs.physical_constants["electron mass"][0] / (mass * cs.physical_constants["atomic mass constant"][0]))


    def __init__(self, params=None, **kwargs):

        self.citations = ['Peper2024Spectroscopy', 'Aymar1984three', 'Meggers1970First', 'Camus1980Highly', 'Camus1969spectre', 'Meggers1970First', 'Wyart1979Extended', 'Aymar1980Highly', 'Camus1980Highly', 'Aymar1984three', 'Martin1978Atomic', 'BiRu1991The', 'Maeda1992Optical', 'zerne1996lande', 'Ali1999Two', 'Lehec2017PhD', 'Lehec2018Laser']

        my_params = {
                     }

        if params is not None:
            my_params.update(params)

        self.p = model_params(my_params)


        self.mqdt_models = []
        self.channels = []

        self.Elim_cm = 50443.070393
        self.Elim_THz = self.Elim_cm*cs.c*10**(-10) + self.p.value('174Yb_Elim_offset_MHz', 0, 1) * 1e-6



        # For 1S0 MQDT model

        mqdt_1S0 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                   config='6s1/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((1 / 2, 1, 3 / 2, 0, 0), Ei_Hz=(80835.39 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='6p3/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 0, 0), Ei_Hz=(77504.98 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='6p1/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot)
                    ]}

        mqdt_1S0.update({
            'channels': [
                channel(mqdt_1S0['cores'][0], (1 / 2, 0, 1 / 2), tt='slj'),
                channel(mqdt_1S0['cores'][1], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_1S0['cores'][2], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_1S0['cores'][3], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_1S0['cores'][4], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_1S0['cores'][5], (1 / 2, 1, 1 / 2), tt='slj', no_me=True)
            ]})

        # From LS-jj transformation.
        Uiabar1S0 = np.identity(6)
        Uiabar1S0[2, 2] = -np.sqrt(2 / 3)
        Uiabar1S0[4, 4] = np.sqrt(2 / 3)
        Uiabar1S0[2, 4] = np.sqrt(1 / 3)
        Uiabar1S0[4, 2] = np.sqrt(1 / 3)

        self.p.set_prefix('174Yb_1s0')

        MQDT_1S0 = mqdt_class(channels=mqdt_1S0['channels'],
                              eig_defects=[[self.p.value('mu0',0.355097325), self.p.value('mu0_1',0.278368431)], [self.p.value('mu1',0.204537279)],
                                           [self.p.value('mu2',0.116394359)], [self.p.value('mu3',0.295432196)], [self.p.value('mu4',0.25765161)],
                                           [self.p.value('mu5',0.155807042)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [3, 4], [3, 5], [1, 6]],
                              rot_angles=[[self.p.value('th12',0.126548585)], [self.p.value('th13',0.300107437)], [self.p.value('th14',0.0570338119)],
                                          [self.p.value('th34',0.114398046)],[self.p.value('th35',0.0986437454)], [self.p.value('th16',0.142482098)]],
                              Uiabar=Uiabar1S0, nulims=[[2], [0]],atom=self)

        self.mqdt_models.append({'L': 0, 'F': 0, 'model': MQDT_1S0})
        self.channels.extend(mqdt_1S0['channels'])

        self.p.set_prefix('174Yb')

        # 3S1 Rydberg Ritz formula
        QDT_3S1 = mqdt_class_rydberg_ritz(channels=mqdt_1S0['channels'][0],
                                          deltas=[self.p.value('3s1_rr_%d'%it,val) for it,val in enumerate([4.4382, 4, -10000, 8 * 10 ** 6, -3 * 10 ** 9])],atom=self)


        self.mqdt_models.append({'L': 0, 'F': 1, 'model': QDT_3S1})

        # For 3P0 MQDT model
        mqdt_3P0 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                   config='6s1/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                    ]}

        mqdt_3P0.update({
            'channels': [
                channel(mqdt_3P0['cores'][0], (1 / 2, 1, 1 / 2), tt='slj'),
                channel(mqdt_3P0['cores'][1], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
            ]})

        self.p.set_prefix('174Yb')

        MQDT_3P0 = mqdt_class(channels=mqdt_3P0['channels'],
                              eig_defects=[[self.p.value('3p0_mu0',0.95356884), self.p.value('3p0_mu0_1',-0.28602498)], [self.p.value('3p0_mu1',0.19845903)]],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('3p0_th12',0.16328854)]],
                              Uiabar=np.identity(2), nulims=[[1],[0]],atom=self)

        self.mqdt_models.append({'L': 1, 'F': 0, 'model': MQDT_3P0})
        self.channels.extend(mqdt_3P0['channels'])


        # For 1,3P1 MQDT model
        mqdt_13P1 = {'cores': [
                         core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                    config='6s1/2', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (a)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (b)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (c)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (d)', potential=self.model_pot),
                     ]}

        mqdt_13P1.update({
            'channels': [
                channel(mqdt_13P1['cores'][0], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_13P1['cores'][0], (1 / 2, 1, 1 / 2), tt='slj'),
                channel(mqdt_13P1['cores'][1], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_13P1['cores'][2], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_13P1['cores'][3], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_13P1['cores'][4], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
            ]})

        # From LS-jj transformation from F. ROBICHEAUX et al. PRA (2018), Eq. (11)
        Uiabar13P1 = np.identity(6)
        Uiabar13P1[0, 0] = np.sqrt(2 / 3)
        Uiabar13P1[0, 1] = 1 / np.sqrt(3)
        Uiabar13P1[1, 0] = -1 / np.sqrt(3)
        Uiabar13P1[1, 1] = np.sqrt(2 / 3)

        self.p.set_prefix('174Yb')

        MQDT_13P1 = mqdt_class(channels=mqdt_13P1['channels'],  # self.p.value('13p1_th13',-0.058),#
                               eig_defects=[[self.p.value('13p1_mu0', 0.922710983), self.p.value('13p1_mu0_1', 2.60362571)],
                                            [self.p.value('13p1_mu1', 0.982087193), self.p.value('13p1_mu1_1', -5.4562725)], [self.p.value('13p1_mu2', 0.228517204)],
                                            [self.p.value('13p1_mu3', 0.206077587)], [self.p.value('13p1_mu4', 0.193527511)], [self.p.value('13p1_mu5', 0.181530935)]],
                               rot_order=[[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6]],
                               rot_angles=[[self.p.value('13p1_th12_0', -8.41087098e-02), self.p.value('13p1_th12_2', 1.20375554e+02), self.p.value('13p1_th12_4', -9.31423120e+03)], [self.p.value('13p1_th13', -0.073181559)], [self.p.value('13p1_th14', -0.06651977)], [self.p.value('13p1_th15', -0.0221098858)], [self.p.value('13p1_th16', -0.104516976)], [self.p.value('13p1_th23', 0.0247704758)], [self.p.value('13p1_th24', 0.0576580705)], [self.p.value('13p1_th25', 0.0860627643)], [self.p.value('13p1_th26', 0.0499436344)]],
                               Uiabar=Uiabar13P1, nulims=[[2, 3, 4, 5], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 1, 'F': 1, 'model': MQDT_13P1})
        self.channels.extend(mqdt_13P1['channels'])

        mqdt_3P2 = {'cores': [
                    core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                config='6s1/2', potential=self.model_pot),
                    core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                config='4f135d6s (a)', potential=self.model_pot),
                    core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                config='4f135d6s (b)', potential=self.model_pot),
                    core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                config='4f135d6s (c)', potential=self.model_pot),
                ]}

        mqdt_3P2.update({
            'channels': [
                channel(mqdt_3P2['cores'][0], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_3P2['cores'][1], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_3P2['cores'][2], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_3P2['cores'][3], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
            ]})

        # From LS-jj transformation from F. ROBICHEAUX et al. PRA (2018), Eq. (11)
        Uiabar3P2 = np.identity(4)

        self.p.set_prefix('174Yb')

        MQDT_3P2 = mqdt_class(channels=mqdt_3P2['channels'],  # self.p.value('13p1_th13',-0.058),
                                    eig_defects=[[self.p.value('3p2_mu0', 0.925121305), self.p.value('3p2_mu0_1', -2.73247165), self.p.value('3p2_mu0_2', 74.664989)],
                                                  [self.p.value('3p2_mu1', 0.230133261)],
                                                 [self.p.value('3p2_mu2', 0.209638118)],
                                                 [self.p.value('3p2_mu3', 0.186228192)]],
                                    rot_order=[[1, 2], [1, 3],[1,4]],
                                    rot_angles=[[self.p.value('3p2_th12', 0.0706666127)], [self.p.value('3p2_th13', 0.0232711158)], [self.p.value('3p2_th14', -0.0292153659)]],
                                    Uiabar=Uiabar3P2, nulims=[[1,2,3], [0]], atom=self)

        self.mqdt_models.append({'L': 1, 'F': 2, 'model': MQDT_3P2})

        # For 1,3D2 MQDT model
        mqdt_13D2 = {'cores': [
                         core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                    config='6s1/2', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (a)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (b)', potential=self.model_pot),
                         core_state((1 / 2, 1, 1 / 2, 0, 1 / 2), Ei_Hz=(79725.35 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (c)', potential=self.model_pot),
                     ]}

        mqdt_13D2.update({
            'channels': [
                channel(mqdt_13D2['cores'][0], (1 / 2, 2, 5 / 2), tt='slj'),
                channel(mqdt_13D2['cores'][0], (1 / 2, 2, 3 / 2), tt='slj'),
                channel(mqdt_13D2['cores'][1], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_13D2['cores'][2], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_13D2['cores'][3], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
            ]})

        # From LS-jj transformation from F. ROBICHEAUX et al. PRA (2018), Eq. (11)
        Uiabar13D2 = np.identity(5)
        Uiabar13D2[0, 0] = np.sqrt(3 / 5)
        Uiabar13D2[0, 1] = np.sqrt(2 / 5)
        Uiabar13D2[1, 0] = - np.sqrt(2 / 5)
        Uiabar13D2[1, 1] = np.sqrt(3 / 5)

        self.p.set_prefix('174Yb')

        MQDT_13D2 = mqdt_class(channels=mqdt_13D2['channels'],
                               eig_defects=[[self.p.value('13d2_mu0',0.729500971), self.p.value('13d2_mu0_1',-0.0284447537)],
                                            [self.p.value('13d2_mu1',0.75229161), self.p.value('13d2_mu1_1',0.0967044398)],
                                            [self.p.value('13d2_mu2', 0.196120406)], [self.p.value('13d2_mu3',0.233821165)],[self.p.value('13d2_mu4',0.152890218)]],
                               rot_order=[[1,2],[1, 3], [1, 4], [2, 4], [1, 5], [2, 5]],
                               rot_angles=[[self.p.value('13d2_th12_0',0.21157531),self.p.value('13d2_th12_2',-15.38440215)],[self.p.value('13d2_th13',0.00522534111)], [self.p.value('13d2_th14',0.0398754262)],
                                           [self.p.value('13d2_th24',-0.00720265975)], [self.p.value('13d2_th15',0.104784389)], [self.p.value('13d2_th25',0.0721775002)]],
                               Uiabar=Uiabar13D2, nulims=[[4],[0, 1]],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 2, 'model': MQDT_13D2})
        self.channels.extend(mqdt_13D2['channels'])

        # 3D1 Rydberg Ritz formula
        QDT_3D1 = mqdt_class_rydberg_ritz(channels=mqdt_13D2['channels'][1],
                                          deltas=[self.p.value('3d1_rr_0',2.75258093), self.p.value('3d1_rr_1',0.382628525,1), self.p.value('3d1_rr_2',-483.120633,100)],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 1, 'model': QDT_3D1})

        # 3D3 Rydberg Ritz formula
        QDT_3D3 = mqdt_class_rydberg_ritz(channels=mqdt_13D2['channels'][0],
                                          deltas=[self.p.value('3d3_rr_0',2.72895315), self.p.value('3d3_rr_1',-0.20653489,1), self.p.value('3d3_rr_2',220.484722,100)],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 3, 'model': QDT_3D3})

        super().__init__(**kwargs)


    def get_state(self, qn, tt='vlfm', energy_exp_Hz=None, energy_only=False):
        """
        Retrieves the state for a given set of quantum numbers (nu, l, f, m).

        Note that nu is used as a guess to find an exact nu from the relevant MQDT model. The energy is calculated from the
        exact nu corresponding to the bound state, while st.v is rounded to two decimal places to serve as a unique but not overly complex
        label for the state.

        Parameters:
            qn (list): A list containing the quantum numbers [nu, l, f, m].
            tt (str): The type of calculation to perform. Default is 'vlfm'.
            energy_exp_Hz (float): The experimental energy in Hz with respect to the lowest ionization limit. Default is None.
            energy_only (bool): If True, only the energy is calculated without finding channel contributions. Default is False. Useful for spectrum fitting.

        Returns:
            state_mqdt: An instance of the state_mqdt class representing the state information.

        Raises:
            ValueError: If the MQDT model for the given quantum numbers is not found.
        """
        if tt == 'vlfm' and len(qn) == 4:

            n = qn[0]
            v = qn[0]
            l = qn[1]
            f = qn[2]
            m = qn[3]

            # NB: l >= v is not exactly right...
            if l < 0 or l >= v or np.abs(m) > f:
                return None

        elif tt == 'NIST':
            st = self.get_state_nist(qn, tt='nsljm')
            return st
        else:
            print("tt=", tt, " not supported by H.get_state")

        # choose MQDT model
        try:
            solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
        except:
            #raise ValueError(f"Could not find MQDT model for qn={qn}")
            return None
            #continue

        Ia = solver.ionizationlimits_invcm[solver.nulima[0]]
        Ib = solver.ionizationlimits_invcm[solver.nulimb[0]]

        # calculate experimental effective quantum number
        if energy_exp_Hz is not None:
            nuexp = ((- 0.01 * energy_exp_Hz / cs.c) / solver.RydConst_invcm) ** (-1 / 2)
        else:
            nuexp = v

        nub = solver.boundstates(nuexp)
        nuapprox = round(nub * 100) / 100

        # calculate energy of state
        E_rel_Hz = (-solver.RydConst_invcm / nub ** 2 + Ib) * 100 * cs.c

        if energy_only:
            [coeffs_i, coeffs_alpha] = [len(solver.channels) * [0],len(solver.channels) * [0]]
        else:
            [coeffs_i, coeffs_alpha] = solver.channelcontributions(nub)

        # define sate
        st = state_mqdt(self, (nuapprox, (-1) ** l, f, m), coeffs_i, coeffs_alpha, solver.channels, energy_Hz=E_rel_Hz, tt='vpfm')
        st.pretty_str =  "|%s:%.2f,L=%d,F=%.1f,%.1f>" % (self.name, nuapprox, l, f, m)

        # effective quantum numbers with respect to two ionization limits Ia and Ib
        st.nua = solver.nux(Ia,Ib,nub)
        st.nub = nub
        st.v_exact = nub

        return st

    def get_state_nist(self, qn, tt='nsljm'):

        if tt == 'nsljm':
            # this is what we use to specify states near the ground state, that are LS coupled

            n = qn[0]
            s = qn[1]
            l = qn[2]
            j = qn[3]
            m = qn[4]

            if l < 0 or l >= n or np.abs(m) > j:
                return None

            pretty_str = "|%s:%d,S=%d,L=%d,j=%d,%.1f>" % (self.name, n, s, l, j, m)



            # defining core states
            mqdt_LS = {'cores': [core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                    config='6s1/2', potential=self.model_pot),]}

            # generate channels by iterating over two core hyperfine states and Rydberg quantum numbers
            mqdt_LS.update({'channels': [channel(mqdt_LS['cores'][0], (1 / 2, l, j), tt='slj') for j in np.arange(np.abs(l-1/2),l+1/2+0.1) ]})

            datadir = importlib.resources.files('rydcalc')

            with open(datadir.joinpath('Yb174_NIST.txt'), 'r') as json_file:
                nist_data = json.load(json_file)

            nist_data = nist_data[1:] # drop references

            dat = list(filter(lambda x: x['n']== n and x['l']== l and x['S']== s and x['J'] == j, nist_data))

            if len(dat) == 0:
                return None

            dat = dat[0]

            # we are going to express this in terms of our mqdt_LS system, which will cover all of the 3PJ states (some will have zero weight)

            energy_Hz = (dat['E_cm'] - self.Elim_cm) * 100 * cs.c

            Ais = []

            for ch in mqdt_LS['channels']:
                # now go through the frame transformations in 10.1103/PhysRevA.97.022508 Eq. 11,

                # Eq 11
                # print((ch.core.s, ch.s, s, ch.core.l, ch.l, l, ch.core.j, ch.j, j))
                ls_to_jj = np.sqrt(2 * s + 1) * np.sqrt(2 * l + 1) * np.sqrt(2 * ch.core.j + 1) * np.sqrt(2 * ch.j + 1) * wigner_9j(ch.core.s, ch.s, s, ch.core.l, ch.l, l, ch.core.j, ch.j, j)

                # print(jj_to_f,ls_to_jj)
                Ais.append(ls_to_jj)

            Aalphas  = []

            st = state_mqdt(self, (n, s, l, j, j, m), Ais, Aalphas, mqdt_LS['channels'], energy_Hz=energy_Hz, tt='nsljfm')
            st.pretty_str = pretty_str
            return st

        else:
            print("tt=", tt, " not supported by H.get_state")

    def get_nearby(self, st, include_opts={}, energy_only = False):
        """ generate a list of quantum number tuples specifying nearby states for sb.fill().
        include_opts can override options in terms of what states are included.

        It's a little messy to decide which options should be handled here vs. in single_basis
        decision for now is to have all quantum numbers here but selection rules/energy cuts
        in single_basis to avoid duplication of code.

        In contrast to get_nearby, this function actually returns a list of states """

        ret = []

        o = {'dn': 2, 'dl': 2, 'dm': 1, 'ds': 0}

        for k, v in include_opts.items():
            o[k] = v

        if 'df' not in o.keys():
            o['df'] = o['dl']

        # calculate experimental effective quantum number
        nu0 = st.nub

        for l in np.arange(st.channels[0].l - o['dl'], st.channels[0].l + o['dl'] + 1):
            if l < 0:
                continue
            for f in np.arange(st.f - o['df'], st.f + o['df'] + 1):
                if f < 0 or f>l+1 or f<l-1:
                    continue

                try:
                    # choose MQDT model
                    solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
                except:
                    continue

                boundstatesinrange = solver.boundstatesinrange([nu0 - o['dn'], nu0 + o['dn']])

                # TODO: the code below here is duplicative of what is in Yb.get_state, and we would ideally like to merge them
                for nua,nub in zip(boundstatesinrange[0],boundstatesinrange[1]):

                    if nub <= 0:
                        continue # reject occasional garbage from solver

                    # calculate energy of new state
                    E_rel_Hz = (-solver.RydConst_invcm / nub ** 2 + solver.ionizationlimits_invcm[solver.nulimb[0]]) * 100 * cs.c

                    if energy_only:
                        [coeffs_i, coeffs_alpha] = [len(solver.channels) * [0], len(solver.channels) * [0]]
                    else:
                        [coeffs_i, coeffs_alpha] = solver.channelcontributions(nub)

                    nuapprox = round(nub * 100) / 100
                    t = np.argmax(np.array(coeffs_i) ** 2)

                    for m in np.arange(st.m - o['dm'], st.m + o['dm'] + 1):

                        if (-f) <= m <= f:
                            # define sate
                            st_new = state_mqdt(self, (nuapprox, (-1) ** l, f, m), coeffs_i,coeffs_alpha, solver.channels, energy_Hz=E_rel_Hz, tt='vpfm')

                            st_new.pretty_str = "|%s:%.2f,L=%d,F=%.1f,%.1f>" % (self.name, nuapprox, l, f, m)

                            # effective quantum numbers with respect to two ionization limits Ia and Ib
                            st_new.nua = nua
                            st_new.nub = nub
                            st.v_exact = nub

                            ret.append(st_new)

        return ret
