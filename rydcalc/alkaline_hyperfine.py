from .alkaline import *
from .alkaline_data import *
from .MQDTclass import *
from .utils import model_params
import json

import csv, importlib.resources

class Ytterbium171(AlkalineAtom):
    """
        Properites of ytterbium 171 atoms with MQDT models
    """

    name = '171Yb'
    dipole_data_file = 'yb171_dipole_matrix_elements.npy'
    dipole_db_file = 'yb171_dipole.db'

    mass = 170.9363302
    Z = 70
    I = 1 / 2
    gI = 0.49367
    # muI = +0.49367 *muN;

    RydConstHz = cs.physical_constants["Rydberg constant times c in Hz"][0] * \
                 (1 - cs.physical_constants["electron mass"][0] / (mass * cs.physical_constants["atomic mass constant"][0]))

    model_pot = model_potential(0, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [1e-3] * 4,
                                Z, include_so=True, use_model=False)

    ion_hyperfine_6s_Hz = 12642812124.2

    Yb174 = Ytterbium174(use_db=False, cpp_numerov=True)


    
    def __init__(self,params=None,**kwargs):

        self.citations = ['Peper2024Spectroscopy', 'Majewski1985diploma']
        
        my_params = {

        }

        
        if params is not None:
            my_params.update(params)
        
        self.p = model_params(my_params)
        
        self.mqdt_models = []
        self.channels = []

        self.Elim_THz = 1512.24645536 + self.p.value('Elim_offset_MHz',0,1)*1e-6
        self.Elim_cm = self.Elim_THz / (cs.c * 100 * 1e-12)

        # For S F=1/2 MQDT model

        UiaFbar_S12 = np.identity(7)
        UiaFbar_S12[0, 0] = 1 / 2
        UiaFbar_S12[6, 6] = -1 / 2
        UiaFbar_S12[0, 6] = np.sqrt(3) / 2
        UiaFbar_S12[6, 0] = np.sqrt(3) / 2
        UiaFbar_S12[2, 2] = -np.sqrt(2 / 3)
        UiaFbar_S12[4, 4] = np.sqrt(2 / 3)
        UiaFbar_S12[2, 4] = np.sqrt(1 / 3)
        UiaFbar_S12[4, 2] = np.sqrt(1 / 3)

        # MQDT_S12 = mqdt_class_kmatrix(channels = [], k = [])
        # This includes 1S0 MQDT and 3S1 QDT model for 174Yb

        mqdt_s12 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=0)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((1 / 2, 1, 3 / 2, 1 / 2, 1), Ei_Hz=(80835.39 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='6p3/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 1 / 2, 0), Ei_Hz=(77504.98 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='6p1/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot),
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot)

                    ]}

        mqdt_s12.update({
            'channels': [
                channel(mqdt_s12['cores'][0], (1 / 2, 0, 1 / 2), tt='slj'),
                channel(mqdt_s12['cores'][1], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_s12['cores'][2], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_s12['cores'][3], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_s12['cores'][4], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_s12['cores'][5], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_s12['cores'][6], (1 / 2, 0, 1 / 2), tt='slj')
            ]})

        self.p.set_prefix('171YbS12')

        MQDT_S12 = mqdt_class(channels=mqdt_s12['channels'],
                              eig_defects=[[self.p.value('1S0_mu0',0.357519763),self.p.value('1S0_mu0_1',0.298712849)],
                                           [self.p.value('1S0_mu1',0.203907536)], [self.p.value('1S0_mu2',0.116803536)],
                                           [self.p.value('1S0_mu3',0.286731074)], [self.p.value('1S0_mu4',0.248113946)],
                                           [self.p.value('1S0_mu5',0.148678953)],
                                           [self.p.value('3s1_rr_%d'%it,val) for it,val in enumerate([0.438426851,3.91762642, -10612.6828, 8017432.38, -2582622910.0])]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [3, 4], [3, 5], [1, 6]],
                              rot_angles=[[self.p.value('1S0_th12',0.131810463)], [self.p.value('1S0_th13',0.297612147)],
                                          [self.p.value('1S0_th14',0.055508821)],
                                          [self.p.value('1S0_th34',0.101030515)],
                                          [self.p.value('1S0_th35',0.102911159)],
                                          [self.p.value('1S0_th16',0.137723736)]],
                              Uiabar=UiaFbar_S12, nulims=[[0], [6]],atom=self)

        self.mqdt_models.append({'L': 0, 'F': 1 / 2, 'model': MQDT_S12})
        self.channels.extend(mqdt_s12['channels'])


        # For S F=3/2 QDT model
        QDT_S32 = mqdt_class_rydberg_ritz(channels=mqdt_s12['channels'][-1],
                                          deltas=[self.p.value('3s1_rr_%d'%it,val) for it,val in enumerate([0.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0])],atom=self,HFlimit="upper")
        self.mqdt_models.append({'L': 0, 'F': 3 / 2, 'model': QDT_S32})

        # For P F=1/2 QDT model
        mqdt_p12 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (d)', potential=self.model_pot),
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=0)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (e)', potential=self.model_pot),
                    ]}

        mqdt_p12.update({
            'channels': [
                channel(mqdt_p12['cores'][0], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_p12['cores'][0], (1 / 2, 1, 1 / 2), tt='slj'),
                channel(mqdt_p12['cores'][1], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p12['cores'][2], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p12['cores'][3], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p12['cores'][4], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p12['cores'][5], (1 / 2, 1, 1 / 2), tt='slj'),
                channel(mqdt_p12['cores'][6], (1 / 2, 2, 3 / 2), tt='slj', no_me=True)
            ]})

        UiaFbar_P12 = np.identity(8)
        UiaFbar_P12[0, 0] = - np.sqrt(2 / 3)
        UiaFbar_P12[0, 1] = -1 / np.sqrt(3)
        UiaFbar_P12[0, 6] = 0
        UiaFbar_P12[1, 0] = 1 / (np.sqrt(3)) / 2
        UiaFbar_P12[1, 1] = -np.sqrt(1 / 6)
        UiaFbar_P12[1, 6] = np.sqrt(3) / 2
        UiaFbar_P12[6, 0] = -1 / 2
        UiaFbar_P12[6, 1] = 1 / np.sqrt(2)
        UiaFbar_P12[6, 6] = 1 / 2

        self.p.set_prefix('171YbP12')

        # this includes 1,3P1 and 3P0 MQDT models of 174Yb
        MQDT_P12 = mqdt_class(channels=mqdt_p12['channels'],
                              eig_defects=[[self.p.value('13p1_mu0',0.9217065854179427), self.p.value('13p1_mu0_1',2.5656945891503833)],
                                           [self.p.value('13p1_mu1', 0.97963857957678), self.p.value('13p1_mu1_1',-5.239904223907001)], [self.p.value('13p1_mu2', 0.22882871998916327)],
                                           [self.p.value('13p1_mu3',0.20548481835495658)],[self.p.value('13p1_mu4', 0.19352862937052895)],[self.p.value('13p1_mu5', 0.18138500038581007)],
                                           [self.p.value('3p0_mu0', 0.9530712824876894), self.p.value('3p0_mu0_1',0.13102524736110277)], [self.p.value('3p0_mu1',0.19844592849024123)]],
                              rot_order=[[1, 2], [2, 7] ,[1,3],  [1, 4], [1,5], [1,6],[2, 3], [2, 4], [2, 5],[2,6],[7,8]],
                              rot_angles=[[self.p.value('13p1_th12_0', -0.0871272272787181),self.p.value('13p1_th12_2', 135.4000088099952),self.p.value('13p1_th12_4', -12985.016241140967)],[self.p.value('13p1_3P0_th27',-0.0014301745962215227)],[self.p.value('13p1_th13',-0.0739040597918768)], [self.p.value('13p1_th14',-0.06363266785280258)],[self.p.value('13p1_th15', -0.02192456902997307)],[self.p.value('13p1_th16',-0.10667881013177766)], [self.p.value('13p1_th23',0.032556998997603115)], [self.p.value('13p1_th24', 0.05410514219677972)], [self.p.value('13p1_th25', 0.08612767171943969)],[self.p.value('13p1_th26',0.05380448698875848)], [self.p.value('3p0_th12',0.16304361917499854)]],
                              Uiabar=UiaFbar_P12, nulims=[[6],[0,1],],atom=self)

        self.mqdt_models.append({'L': 1, 'F': 1 / 2, 'model': MQDT_P12})
        self.channels.extend(mqdt_p12['channels'])

        mqdt_p32 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (a)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (b)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (c)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (d)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (e)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (f)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (g)', potential=self.model_pot),
        ]}

        mqdt_p32.update({
            'channels': [
                channel(mqdt_p32['cores'][0], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_p32['cores'][0], (1 / 2, 1, 1 / 2), tt='slj'),
                channel(mqdt_p32['cores'][1], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p32['cores'][2], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p32['cores'][3], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p32['cores'][4], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p32['cores'][5], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_p32['cores'][6], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p32['cores'][7], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p32['cores'][8], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
            ]})

        UiaFbar_P32 = np.identity(10)
        UiaFbar_P32[0, 0] = np.sqrt(5 / 3) / 2
        UiaFbar_P32[0, 1] = np.sqrt(5 / 6) / 2
        UiaFbar_P32[0, 6] = -np.sqrt(3 / 2) / 2
        UiaFbar_P32[1, 0] = -1 / np.sqrt(3)
        UiaFbar_P32[1, 1] = np.sqrt(2 / 3)
        UiaFbar_P32[1, 6] = 0
        UiaFbar_P32[6, 0] = 1 / 2
        UiaFbar_P32[6, 1] = 1 / (2 * np.sqrt(2))
        UiaFbar_P32[6, 6] = np.sqrt(5 / 2) / 2

        self.p.set_prefix('171YbP32')



        # this includes 1,3P1 MQDT and 3P2 MQDT models of 174Yb,
        MQDT_P32 = mqdt_class(channels=mqdt_p32['channels'],
                              eig_defects=[[self.p.value('13p1_mu0',0.9217065854179427), self.p.value('13p1_mu0_1',2.5656945891503833)],
                                           [self.p.value('13p1_mu1', 0.97963857957678), self.p.value('13p1_mu1_1',-5.239904223907001)], [self.p.value('13p1_mu2', 0.22882871998916327)],
                                           [self.p.value('13p1_mu3',0.20548481835495658)],[self.p.value('13p1_mu4', 0.19352862937052895)],[self.p.value('13p1_mu5', 0.18138500038581007)],
                                           [self.p.value('3p2_mu0', 0.9248257361868282), self.p.value('3p2_mu0_1', -3.5424816435901345), self.p.value('3p2_mu0_2', 81.53346865313813, 100)], [self.p.value('3p2_mu1', 0.2368669026171927,0.005)],
                                                 [self.p.value('3p2_mu2', 0.22105588290875544,0.005)],
                                                 [self.p.value('3p2_mu3', 0.18559937607058488,0.005)],
                                                 ],
                              rot_order=[[1, 2], [1, 3], [1, 4], [1, 5],[1, 6], [2, 3], [2, 4], [2, 5], [2,6],[7,8],[7,9],[7,10]],
                              rot_angles=[[self.p.value('13p1_th12_0', -0.0871272272787181),self.p.value('13p1_th12_2', 135.4000088099952),self.p.value('13p1_th12_4', -12985.016241140967)],[self.p.value('13p1_th13',-0.0739040597918768)], [self.p.value('13p1_th14',-0.06363266785280258)],[self.p.value('13p1_th15', -0.02192456902997307)],[self.p.value('13p1_th16',-0.10667881013177766)], [self.p.value('13p1_th23',0.032556998997603115)], [self.p.value('13p1_th24', 0.05410514219677972)], [self.p.value('13p1_th25', 0.08612767171943969)],[self.p.value('13p1_th26',0.05380448698875848)],[self.p.value('3p2_th12', 0.07142668453155682,0.1)], [self.p.value('3p2_th13', 0.027464109684890103,0.03)], [self.p.value('3p2_th14', -0.02974186194134832,0.03)]],
                              Uiabar=UiaFbar_P32, nulims=[[6],[0,1]],atom=self)

        self.mqdt_models.append({'L': 1, 'F': 3 / 2, 'model': MQDT_P32})
        self.channels.extend(mqdt_p32['channels'])

        mqdt_p52 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (a)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (b)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                       config='4f135d6s (c)', potential=self.model_pot),
        ]}

        mqdt_p52.update({
            'channels': [
                channel(mqdt_p52['cores'][0], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_p52['cores'][1], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p52['cores'][2], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_p52['cores'][3], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
            ]})

        # From LS-jj transformation from F. ROBICHEAUX et al. PRA (2018), Eq. (11)
        UiaFbar_P52 = np.identity(4)

        self.p.set_prefix('171YbP52')

        # For P F=5/2 QDT model. Which is purely Fc=1, q.d. taken from 171Yb F=3/2
        MQDT_P52 = mqdt_class(channels=mqdt_p52['channels'],
                              eig_defects=[[self.p.value('3p2_mu0', 0.9248257361868282), self.p.value('3p2_mu0_1', -3.5424816435901345), self.p.value('3p2_mu0_2', 81.53346865313813, 100)], [self.p.value('3p2_mu1', 0.2368669026171927,0.005)],
                                                 [self.p.value('3p2_mu2', 0.22105588290875544,0.005)],
                                                 [self.p.value('3p2_mu3', 0.18559937607058488,0.005)]],
                              rot_order=[[1, 2], [1, 3], [1, 4]],
                              rot_angles=[[self.p.value('3p2_th12', 0.07142668453155682,0.1)], [self.p.value('3p2_th13', 0.027464109684890103,0.03)], [self.p.value('3p2_th14', -0.02974186194134832,0.03)]],
                              Uiabar=UiaFbar_P52, nulims=[[1, 2, 3], [0]], atom=self)


        self.mqdt_models.append({'L': 1, 'F': 5 / 2, 'model': MQDT_P52})

        # For D F=3/2
        mqdt_d32 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 1 / 2, 0), Ei_Hz=(79725.35 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot),
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=0)', potential=self.model_pot),
                    ]}

        mqdt_d32.update({
            'channels': [
                channel(mqdt_d32['cores'][0], (1 / 2, 2, 5 / 2), tt='slj'),
                channel(mqdt_d32['cores'][0], (1 / 2, 2, 3 / 2), tt='slj'),
                channel(mqdt_d32['cores'][1], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_d32['cores'][2], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_d32['cores'][3], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_d32['cores'][4], (1 / 2, 2, 3 / 2), tt='slj'),
            ]})

        UiaFbar_D32 = np.identity(6)
        UiaFbar_D32[0, 0] = - np.sqrt(3 / 5)
        UiaFbar_D32[0, 1] = - np.sqrt(2 / 5)
        UiaFbar_D32[0, 5] = 0
        UiaFbar_D32[1, 0] = np.sqrt(3 / 5) / 2
        UiaFbar_D32[1, 1] = - 3 / (2 * np.sqrt(10))
        UiaFbar_D32[1, 5] = np.sqrt(5 / 2) / 2
        UiaFbar_D32[5, 0] = - 1 / 2
        UiaFbar_D32[5, 1] = np.sqrt(3 / 2) / 2
        UiaFbar_D32[5, 5] = np.sqrt(3 / 2) / 2

        self.p.set_prefix('171YbD32')
        
        # this includes 1,3D2 MQDT and 3D1 QDT models of 174Yb. Introduced S-T mixing angle
        MQDT_D32 = mqdt_class(channels=mqdt_d32['channels'],
                              eig_defects=[[self.p.value('13d2_mu0',0.730537124), self.p.value('13d2_mu0_1',-0.000186828866)],
                                           [self.p.value('13d2_mu1',0.751591782), self.p.value('13d2_mu1_1',-0.00114049637)],
                                           [self.p.value('13d2_mu2',0.196120394)], [self.p.value('13d2_mu3',0.233742396)],
                                           [self.p.value('13d2_mu4',0.152905343)], [self.p.value('3d1_rr_0',2.75258093), self.p.value('3d1_rr_1',0.382628525,1), self.p.value('3d1_rr_2',-483.120633,100)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [2, 4], [1, 5], [2, 5]],
                              rot_angles=[[self.p.value('13d2_th12',0.205496654)], [self.p.value('13d2_th13',0.00522401624)], [self.p.value('13d2_th14',0.0409502343)], [self.p.value('13d2_th24',-0.00378075773)], [self.p.value('13d2_th15',0.108563952)], [self.p.value('13d2_th25',0.0665700438)]],
                              Uiabar=UiaFbar_D32, nulims=[[5],[0, 1]],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 3 / 2, 'model': MQDT_D32})
        self.channels.extend(mqdt_d32['channels'])

        mqdt_d52 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 1 / 2, 0), Ei_Hz=(79725.35 - 50443.07) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot),
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=0)', potential=self.model_pot),
                    ]}

        mqdt_d52.update({
            'channels': [
                channel(mqdt_d52['cores'][0], (1 / 2, 2, 5 / 2), tt='slj'),
                channel(mqdt_d52['cores'][0], (1 / 2, 2, 3 / 2), tt='slj'),
                channel(mqdt_d52['cores'][1], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_d52['cores'][2], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_d52['cores'][3], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_d52['cores'][4], (1 / 2, 2, 5 / 2), tt='slj'),
            ]})

        UiaFbar_D52 = np.identity(6)
        UiaFbar_D52[0, 0] = np.sqrt(7 / 5) / 2
        UiaFbar_D52[0, 1] = np.sqrt(7 / 30)
        UiaFbar_D52[0, 5] = - np.sqrt(5 / 3) / 2
        UiaFbar_D52[1, 0] = - np.sqrt(2 / 5)
        UiaFbar_D52[1, 1] = np.sqrt(3 / 5)
        UiaFbar_D52[1, 5] = 0
        UiaFbar_D52[5, 0] = 1 / 2
        UiaFbar_D52[5, 1] = 1 / np.sqrt(6)
        UiaFbar_D52[5, 5] = np.sqrt(7 / 3) / 2

        self.p.set_prefix('171YbD52')

        # this includes 1,3D2 MQDT and 3D3 QDT models of 174Yb. Introduced S-T mixing angle
        MQDT_D52 = mqdt_class(channels=mqdt_d52['channels'],
                              eig_defects=[[self.p.value('13d2_mu0',0.730537124), self.p.value('13d2_mu0_1',-0.000186828866)],
                                           [self.p.value('13d2_mu1',0.751591782), self.p.value('13d2_mu1_1',-0.00114049637)],
                                           [self.p.value('13d2_mu2',0.196120394)], [self.p.value('13d2_mu3',0.233742396)],
                                           [self.p.value('13d2_mu4',0.152905343)],  [self.p.value('3d3_rr_0',2.72895315), self.p.value('3d3_rr_1',-0.20653489,1), self.p.value('3d3_rr_2',220.484722,100)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [2, 4], [1, 5], [2, 5]],
                              rot_angles=[[self.p.value('13d2_th12', 0.205496654)], [self.p.value('13d2_th13', 0.00522401624)], [self.p.value('13d2_th14', 0.0409502343)], [self.p.value('13d2_th24', -0.00378075773)], [self.p.value('13d2_th15', 0.108563952)], [self.p.value('13d2_th25', 0.0665700438)]],
                              Uiabar=UiaFbar_D52, nulims=[[5],[0, 1]],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 5 / 2, 'model': MQDT_D52})
        self.channels.extend(mqdt_d52['channels'])

        # For D F=1/2 QDT model. Which is purely Fc=1, q.d. taken from 3D1 fit 171Yb F=3/2
        QDT_D12 = mqdt_class_rydberg_ritz(channels=mqdt_d32['channels'][1],
                                          deltas=[2.75258093, 0.382628525,-483.120633], atom=self, HFlimit = "upper")
        self.mqdt_models.append({'L': 2, 'F': 1 / 2, 'model': QDT_D12})

        # For D F=7/2 QDT model. Which is purely Fc=1, q.d. taken from 3D3 fit 171Yb F=3/2
        QDT_D72 = mqdt_class_rydberg_ritz(channels=mqdt_d32['channels'][0],
                                          deltas=[2.72895315, -0.20653489,220.484722], atom=self,HFlimit = "upper")
        self.mqdt_models.append({'L': 2, 'F': 7 / 2, 'model': QDT_D72})

        
        super().__init__(**kwargs)



    def get_state(self, qn, tt='vlfm', energy_exp_Hz=None, energy_only=False):
        """ IF energy_only= True, just find energy but do not find channel contributions. Useful for spectrum fitting """
        #energyexpHz is the binding energy with respect to the upper hyperfine threshold
        if tt == 'vlfm' and len(qn) == 4:

            n = qn[0]
            v = qn[0]
            l = qn[1]
            f = qn[2]
            m = qn[3]

            if l < 0 or l >= v or np.abs(m) > f or round(f-m) != (f-m):
                return None
        elif tt == 'NIST':
            st = self.get_state_nist(qn, tt='nsljfm')
            return st

        else:
            print("tt=", tt, " not supported by H.get_state")

        # choose MQDT model
        try:
            solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
        except:
            return None

        # calculate experimental effective quantum number
        if energy_exp_Hz is not None:
            nuexp = ((- 0.01 * energy_exp_Hz / cs.c) / solver.RydConst_invcm) ** (-1 / 2)
        else:
            nuexp = v

        nutheor = solver.boundstates(nuexp)
        nuapprox = round(nutheor * 100) / 100

        nua = solver.nux(solver.ionizationlimits_invcm[solver.nulima[0]], solver.ionizationlimits_invcm[solver.nulimb[0]], nutheor)

        # calculate energy of state
        E_rel_Hz = (-solver.RydConst_invcm / nutheor ** 2 + solver.ionizationlimits_invcm[solver.nulimb[0]]) * 100 * cs.c

        if energy_only:
            [coeffs_i, coeffs_alpha] = [len(solver.channels)*[0], len(solver.channels)*[0]]
        else:
            [coeffs_i, coeffs_alpha] = solver.channelcontributions(nutheor)
            #print(coeffs)

        # define sate
        st = state_mqdt(self, (nuapprox, (-1) ** l, f, m), coeffs_i, coeffs_alpha, solver.channels, energy_Hz=E_rel_Hz, tt='vpfm')
        st.pretty_str = "|%s:%.2f,L=%d,F=%.1f,%.1f>" % (self.name, nuapprox, l, f, m)
        st.short_str = "|%.2f,%d,%.1f,%.1f>" % (nuapprox, l, f, m)

        # effective quantum numbers with respect to two ionization limits Ia and Ib
        st.nua = nua
        st.nub = nutheor
        st.v_exact = nutheor

        return st


    def get_state_nist(self, qn, tt='nsljfm'):

        if tt == 'nsljfm':
            # this is what we use to specify states near the ground state, that are LS coupled

            n = qn[0]
            s = qn[1]
            l = qn[2]
            j = qn[3]
            f = qn[4]
            m = qn[5]

            if l < 0 or l >= n or np.abs(m) > f:
                return None

            pretty_str = "|%s:%d,S=%d,L=%d,j=%d,F=%.1f,%.1f>" % (self.name, n, s, l, j, f, m)



            # defining core states
            mqdt_LS = {'cores': [core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                              config='6s1/2 (Fc=0)', potential=self.model_pot),
                                   core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                              config='6s1/2 (Fc=1)', potential=self.model_pot)]}

            # generate channels by iterating over two core hyperfine states and Rydberg quantum numbers
            mqdt_LS.update({'channels': [channel(mqdt_LS['cores'][i], (1 / 2, l, j), tt='slj') for i in [0,1] for j in np.arange(np.abs(l-1/2),l+1/2+0.1) ]})

            datadir = importlib.resources.files('rydcalc')

            with open(datadir.joinpath('Yb171_NIST.txt'), 'r') as json_file:
                nist_data = json.load(json_file)

            nist_data = nist_data[1:] # drop references

            dat = list(filter(lambda x: x['n']== n and x['l']== l and x['S']== s and x['J'] == j, nist_data))

            if len(dat) == 0:
                return None

            dat = dat[0]

            # we are going to express this in terms of our mqdt_LS system, which will cover all of the 3PJ states (some will have zero weight)

            energy_Hz = (dat['E_cm'] - self.Elim_cm) * 100 * cs.c
            # Steck Rb notes Eq. 16
            energy_Hz += 0.5 * dat['A_GHz'] * 1e9 * (f * (f + 1) - self.I * (self.I + 1) - j * (j + 1))

            coeffs_i = []

            for ch in mqdt_LS['channels']:
                # now go through the frame transformations in 10.1103/PhysRevA.97.022508 Eq. 11, 13.

                # Eq 13
                jj_to_f = (-1) ** (ch.j + ch.core.f + ch.core.i + j) * np.sqrt(2 * j + 1) * np.sqrt(2 * ch.core.f + 1) * wigner_6j(ch.j, ch.core.j, j, ch.core.i, f, ch.core.f)

                # Eq 11
                # print((ch.core.s, ch.s, s, ch.core.l, ch.l, l, ch.core.j, ch.j, j))
                ls_to_jj = np.sqrt(2 * s + 1) * np.sqrt(2 * l + 1) * np.sqrt(2 * ch.core.j + 1) * np.sqrt(2 * ch.j + 1) * wigner_9j(ch.core.s, ch.s, s, ch.core.l, ch.l, l, ch.core.j, ch.j, j)

                # print(jj_to_f,ls_to_jj)
                coeffs_i.append(jj_to_f * ls_to_jj)

            coeffs_alpha  = []

            st = state_mqdt(self, (n, s, l, j, f, m), coeffs_i, coeffs_alpha, mqdt_LS['channels'], energy_Hz=energy_Hz, tt='nsljfm')
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

        # get effective quantum number of target state
        nu0 = st.nub

        for l in np.arange(st.channels[0].l - o['dl'], st.channels[0].l + o['dl'] + 1):
            if l<0:
                continue
            for f in np.arange(st.f - o['df'], st.f + o['df'] + 1):
                if f < 0:
                    continue

                try:
                    # choose MQDT model
                    solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
                except:
                    continue

                boundstatesinrange = solver.boundstatesinrange([nu0 - o['dn'], nu0 + o['dn']])

                for nua,nub in zip(boundstatesinrange[0],boundstatesinrange[1]):

                    # calculate energy of new state
                    E_rel_Hz = (-solver.RydConst_invcm / nub ** 2 + solver.ionizationlimits_invcm[solver.nulimb[0]]) * 100 * cs.c

                    if energy_only:
                        [coeffs_i, coeffs_alpha] = [len(solver.channels) * [0], len(solver.channels) * [0]]
                    else:
                        [coeffs_i, coeffs_alpha] = solver.channelcontributions(nub)
                        # print(coeffs)

                    nuapprox = round(nub * 100) / 100
                    t = np.argmax(np.array(coeffs_i)**2)

                    for m in np.arange(st.m - o['dm'], st.m + o['dm'] + 1):

                        if  (-f) <= m <=f:

                            # define sate
                            st_new = state_mqdt(self, (nuapprox, (-1) ** l, f, m), coeffs_i, coeffs_alpha, solver.channels, energy_Hz=E_rel_Hz, tt='npfm')

                            st_new.pretty_str =  "|%s:%.2f,L=%d,F=%.1f,%.1f>" % (self.name, nuapprox, l, f, m)
                            st_new.short_str = "|%.2f,%d,%.1f,%.1f>" % (nuapprox, l, f, m)
                            # effective quantum numbers with respect to two ionization limits Ia and Ib
                            st_new.nua = nua
                            st_new.nub = nub
                            st.nu_exact = nub

                            if st_new.nub>0:
                                ret.append(st_new)

        return ret

    def energy_from_3P0_Hz(self, st):
        """ Compute energy relative to 6s6p 3p0 state.

        Energy of

        518295836590863.61 Hz

        is from:

        Pizzocaro et al 2020 Metrologia 57 035007, 10.1088/1681-7575/ab50e8
        """

        return st.get_energy_Hz() + self.Elim_THz * 1e12 - 518295836590863.61
    