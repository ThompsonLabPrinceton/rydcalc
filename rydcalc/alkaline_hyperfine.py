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
    ion_hyperfine_6p12_Hz = 2.1*10**9

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

        # Center-of-mass energy and fine structure splitting of lowest p and d states of the Yb+ ion for calculation of high-l fine structure
        self.deltaEp_m = 100*(30392.23 - 27061.82)
        self.Ep_m = 100*((4/6)*30392.23 + (2/6)*27061.82)

        self.deltaEd_m = 100*(24332.69 - 22960.80)
        self.Ed_m = 100*((3/5)*24332.69 + (2/5)*22960.80)

        # For S F=1/2 MQDT model

        UiaFbar_S12 = np.identity(7)
        UiaFbar_S12[0, 0] = 1 / 2
        UiaFbar_S12[6, 6] = -1 / 2
        UiaFbar_S12[0, 6] = np.sqrt(3) / 2
        UiaFbar_S12[6, 0] = np.sqrt(3) / 2
        UiaFbar_S12[2, 2] = np.sqrt(2 / 3)
        UiaFbar_S12[4, 4] = np.sqrt(2 / 3)
        UiaFbar_S12[2, 4] = -np.sqrt(1 / 3)
        UiaFbar_S12[4, 2] = np.sqrt(1 / 3)

        # This includes 1S0 MQDT and 3S1 QDT model for 174Yb

        mqdt_s12 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=0)', potential=self.model_pot,alpha_d_a03 = 60.51, alpha_q_a05 = 672),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((1 / 2, 1, 3 / 2, 1 / 2, 1), Ei_Hz=(80835.39 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='6p3/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 1 / 2, 0), Ei_Hz=(77504.98 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='6p1/2', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot),
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot,alpha_d_a03 = 60.51, alpha_q_a05 = 672)

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
                              eig_defects=[[self.p.value('1S0_mu0',0.357488757),self.p.value('1S0_mu0_1',0.163255076)],
                                           [self.p.value('1S0_mu1',0.203917828)], [self.p.value('1S0_mu2',0.116813499)],
                                           [self.p.value('1S0_mu3',0.287210377)], [self.p.value('1S0_mu4',0.247550262)],
                                           [self.p.value('1S0_mu5',0.148686263)],
                                           [self.p.value('3s1_rr_%d'%it,val) for it,val in enumerate([0.438803844,3.54561559, -10673.2496, 7702824.55, -2255973430.0])]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [3, 4], [3, 5], [1, 6]],
                              rot_angles=[[self.p.value('1S0_th12',0.13179534)], [self.p.value('1S0_th13',0.29748039)],
                                          [self.p.value('1S0_th14',0.0553920359)],
                                          [self.p.value('1S0_th34',0.100843905)],
                                          [self.p.value('1S0_th35',0.10317753)],
                                          [self.p.value('1S0_th16',0.137709223)]],
                              Uiabar=UiaFbar_S12, nulims=[[0], [6]],atom=self)

        self.mqdt_models.append({'L': 0, 'F': 1 / 2, 'model': MQDT_S12})
        self.channels.extend(mqdt_s12['channels'])


        # For S F=3/2 QDT model
        QDT_S32 = mqdt_class_rydberg_ritz(channels=mqdt_s12['channels'][-1],
                                          deltas=[self.p.value('3s1_rr_%d'%it,val) for it,val in enumerate([0.438803844,3.54561559, -10673.2496, 7702824.55, -2255973430.0])],atom=self,HFlimit="upper")
        self.mqdt_models.append({'L': 0, 'F': 3 / 2, 'model': QDT_S32})

        # For P F=1/2 QDT model
        mqdt_p12 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (c)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (d)', potential=self.model_pot),
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=0)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
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

        # this includes 1,3P1 and 3P0 MQDT models of 174Yb#
        MQDT_P12 = mqdt_class(channels=mqdt_p12['channels'],
                              eig_defects=[[self.p.value('13p1_mu0',0.922046003), self.p.value('13p1_mu0_1',2.17932092)],
                                           [self.p.value('13p1_mu1', 0.981076381), self.p.value('13p1_mu1_1',-4.48995034)], [self.p.value('13p1_mu2', 0.229019212)],
                                           [self.p.value('13p1_mu3',0.205999445)],[self.p.value('13p1_mu4', 0.19352756)],[self.p.value('13p1_mu5', 0.181271472)],
                                           [self.p.value('3p0_mu0', 0.953288122), self.p.value('3p0_mu0_1',-0.0298396847)], [self.p.value('3p0_mu1',0.198451139)]],
                              rot_order=[[1, 2], [2, 7] ,[1,3],  [1, 4], [1,5], [1,6],[2, 3], [2, 4], [2, 5],[2,6],[7,8]],
                              rot_angles=[[self.p.value('13p1_th12_0', -0.100292816),self.p.value('13p1_th12_2', 149.140925),self.p.value('13p1_th12_4', -13487.7692)],[self.p.value('13p1_3P0_th27',-0.0016592076)],[self.p.value('13p1_th13',-0.0727917308)], [self.p.value('13p1_th14',-0.0669120237)],[self.p.value('13p1_th15',-0.0221321759)],[self.p.value('13p1_th16', -0.107302569)], [self.p.value('13p1_th23',0.0396527798)], [self.p.value('13p1_th24', 0.0596597186)], [self.p.value('13p1_th25', 0.0861416497)],[self.p.value('13p1_th26',0.0565415641)], [self.p.value('3p0_th12',0.163175562)]],
                              Uiabar=UiaFbar_P12, nulims=[[6],[0,1],],atom=self)

        self.mqdt_models.append({'L': 1, 'F': 1 / 2, 'model': MQDT_P12})
        self.channels.extend(mqdt_p12['channels'])

        # For P and F F=3/2 MQDT model

        mqdt_p32 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (a)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (b)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (c)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (d)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (e)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (f)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
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
                channel(mqdt_p32['cores'][0], (1 / 2, 3, 5 / 2), tt='slj'),
            ]})

        UiaFbar_P32 = np.identity(11)
        UiaFbar_P32[0, 0] = np.sqrt(5 / 3) / 2
        UiaFbar_P32[0, 1] = np.sqrt(5 / 6) / 2
        UiaFbar_P32[0, 6] = -np.sqrt(3 / 2) / 2
        UiaFbar_P32[1, 0] = -1 / np.sqrt(3)
        UiaFbar_P32[1, 1] = np.sqrt(2 / 3)
        UiaFbar_P32[1, 6] = 0
        UiaFbar_P32[6, 0] = 1 / 2
        UiaFbar_P32[6, 1] = 1 / (2 * np.sqrt(2))
        UiaFbar_P32[6, 6] = np.sqrt(5 / 2) / 2
        UiaFbar_P32[-1, -1] = -1

        self.p.set_prefix('171YbP32')


        # this includes 1,3P1 MQDT, 3P2, and 3F2 MQDT models of 174Yb,
        MQDT_P32 = mqdt_class(channels=mqdt_p32['channels'],
                              eig_defects=[[self.p.value('13p1_mu0',0.922046003), self.p.value('13p1_mu0_1',2.17932092)],
                                           [self.p.value('13p1_mu1', 0.981076381), self.p.value('13p1_mu1_1',-4.48995034)], [self.p.value('13p1_mu2', 0.229019212)],
                                           [self.p.value('13p1_mu3',0.205999445)],[self.p.value('13p1_mu4', 0.19352756)],[self.p.value('13p1_mu5', 0.181271472)],
                                           [self.p.value('3p2_mu0', 0.926908753), self.p.value('3p2_mu0_1', -4.6228148), self.p.value('3p2_mu0_2', 453.565911, 100)], [self.p.value('3p2_mu1', 0.223859222,0.005)],
                                                 [self.p.value('3p2_mu2', 0.240057861,0.005)],
                                                 [self.p.value('3p2_mu3', 0.184824687,0.005)], [self.p.value('3f2_rr_0', 0.0718810048),self.p.value('3f2_rr_1',-1.08216233),self.p.value('3f2_rr_2',-38.2507093)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [1, 5],[1, 6], [2, 3], [2, 4], [2, 5], [2,6],[7,8],[7,9],[7,10],[7,11]],
                              rot_angles=[[self.p.value('13p1_th12_0', -0.100292816),self.p.value('13p1_th12_2', 149.140925),self.p.value('13p1_th12_4', -13487.7692)],[self.p.value('13p1_th13',-0.0727917308)], [self.p.value('13p1_th14',-0.0669120237)],[self.p.value('13p1_th15',-0.0221321759)],[self.p.value('13p1_th16', -0.107302569)], [self.p.value('13p1_th23',0.0396527798)], [self.p.value('13p1_th24', 0.0596597186)], [self.p.value('13p1_th25', 0.0861416497)],[self.p.value('13p1_th26',0.0565415641)],[self.p.value('3p2_th12', 0.0657620037,0.1)], [self.p.value('3p2_th13', 0.0215995148,0.03)], [self.p.value('3p2_th14',-0.028252844,0.03)], [self.p.value('3f2_th', 0.0184267375)]],
                              Uiabar=UiaFbar_P32, nulims=[[6],[0,1,10]],atom=self)

        self.mqdt_models.append({'L': 1, 'F': 3 / 2, 'model': MQDT_P32})
        self.channels.extend(mqdt_p32['channels'])

        #NOTE P F=5/2 is defined with the F F=5/2 model below 

        # For D F=3/2 QDT model
        
        mqdt_d32 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 1 / 2, 0), Ei_Hz=(79725.35 - self.Elim_cm) * cs.c * 100, tt='sljif',
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
                              eig_defects=[[self.p.value('13d2_mu0',0.730541589), self.p.value('13d2_mu0_1',-0.0967938662)],
                                           [self.p.value('13d2_mu1',0.751542685), self.p.value('13d2_mu1_1',0.00038836127)],
                                           [self.p.value('13d2_mu2',0.195864083)], [self.p.value('13d2_mu3',0.235944408)],
                                           [self.p.value('13d2_mu4',0.147483609)], [self.p.value('3d1_rr_0',2.75334159), self.p.value('3d1_rr_1',-1.80187083,1), self.p.value('3d1_rr_2',986.918851,100)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [2, 4], [1, 5], [2, 5]],
                              rot_angles=[[self.p.value('13d2_th12_0', 0.220048245),self.p.value('13d2_th12_2', -14.9486)], [self.p.value('13d2_th13',0.00427599)], [self.p.value('13d2_th14',0.0381563093)], [self.p.value('13d2_th24',-0.00700797918)], [self.p.value('13d2_th15',0.109380331)], [self.p.value('13d2_th25',0.0635544456)]],
                              Uiabar=UiaFbar_D32, nulims=[[5],[0, 1]],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 3 / 2, 'model': MQDT_D32})
        self.channels.extend(mqdt_d32['channels'])

        mqdt_d52 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                                   config='6s1/2 (Fc=1)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (a)', potential=self.model_pot),
                        core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                   config='4f135d6s (b)', potential=self.model_pot),
                        core_state((1 / 2, 1, 1 / 2, 1 / 2, 0), Ei_Hz=(79725.35 - self.Elim_cm) * cs.c * 100, tt='sljif',
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
                              eig_defects=[[self.p.value('13d2_mu0',0.730541589), self.p.value('13d2_mu0_1',-0.0967938662)],
                                           [self.p.value('13d2_mu1',0.751542685), self.p.value('13d2_mu1_1',0.00038836127)],
                                           [self.p.value('13d2_mu2',0.195864083)], [self.p.value('13d2_mu3',0.235944408)],
                                           [self.p.value('13d2_mu4',0.147483609)],  [self.p.value('3d3_rr_0',2.72865616), self.p.value('3d3_rr_1',0.793185994,1), self.p.value('3d3_rr_2',-523.858959,100)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [2, 4], [1, 5], [2, 5]],
                              rot_angles=[[self.p.value('13d2_th12_0', 0.220048245),self.p.value('13d2_th12_2', -14.9486)], [self.p.value('13d2_th13',0.00427599)], [self.p.value('13d2_th14',0.0381563093)], [self.p.value('13d2_th24',-0.00700797918)], [self.p.value('13d2_th15',0.109380331)], [self.p.value('13d2_th25',0.0635544456)]],
                              Uiabar=UiaFbar_D52, nulims=[[5],[0, 1]],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 5 / 2, 'model': MQDT_D52})
        self.channels.extend(mqdt_d52['channels'])

        # For D F=1/2 QDT model. Which is purely Fc=1, q.d. taken from 3D1 fit 171Yb F=3/2
        QDT_D12 = mqdt_class_rydberg_ritz(channels=mqdt_d32['channels'][1],
                                          deltas=[2.75334159, -1.80187083,986.918851], atom=self, HFlimit = "upper")
        self.mqdt_models.append({'L': 2, 'F': 1 / 2, 'model': QDT_D12})

        # For D F=7/2 QDT model. Which is purely Fc=1, q.d. taken from 3D3 fit 171Yb F=3/2
        QDT_D72 = mqdt_class_rydberg_ritz(channels=mqdt_d32['channels'][0],
                                          deltas=[2.72865616, 0.793185994,-523.858959], atom=self,HFlimit = "upper")
        self.mqdt_models.append({'L': 2, 'F': 7 / 2, 'model': QDT_D72})

        mqdt_f52 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (a)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (b)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (c)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (d)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (e)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (a)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (b)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (c)', potential=self.model_pot),
        ]}

        mqdt_f52.update({
            'channels': [
                channel(mqdt_f52['cores'][0], (1 / 2, 3, 7 / 2), tt='slj'),
                channel(mqdt_f52['cores'][0], (1 / 2, 3, 5 / 2), tt='slj'),
                channel(mqdt_f52['cores'][1], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][2], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][3], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][4], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][5], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][0], (1 / 2, 1, 3 / 2), tt='slj'),
                channel(mqdt_f52['cores'][7], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][8], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][9], (1 / 2, 2, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f52['cores'][6], (1 / 2, 3, 5 / 2), tt='slj'),
            ]})

        UiaFbar_F52 = np.identity(12)
        UiaFbar_F52[0, 0] = -0.75592895
        UiaFbar_F52[0, 1] = -0.65465367
        UiaFbar_F52[0, 11] = 0
        UiaFbar_F52[1, 0] = 0.42257713
        UiaFbar_F52[1, 1] = -0.48795004
        UiaFbar_F52[1, 11] = 0.76376262
        UiaFbar_F52[11, 0] = -1 / 2
        UiaFbar_F52[11, 1] =  0.57735027
        UiaFbar_F52[11, 11] = 0.64549722

        self.p.set_prefix('171YbF52')        

        MQDT_F52 = mqdt_class(channels=mqdt_f52['channels'],
                              eig_defects=[[self.p.value('13f3_mu0',0.27707535), self.p.value('13f3_mu0_1',-13.2829133)], [self.p.value('13f3_mu1',0.0719827367), self.p.value('13f3_mu1_1',-0.741492064)], [self.p.value('13f3_mu2',0.250829254)],[self.p.value('13f3_mu3',0.227042016)],[self.p.value('13f3_mu4',0.17576256)],[self.p.value('13f3_mu5',0.19654694)],[self.p.value('13f3_mu6',0.214351625)],
                                           [self.p.value('3p2_mu0', 0.926908753), self.p.value('3p2_mu0_1', -4.6228148), self.p.value('3p2_mu0_2', 453.565911, 100)], [self.p.value('3p2_mu1', 0.223859222,0.005)],
                                            [self.p.value('3p2_mu2', 0.240057861,0.005)],[self.p.value('3p2_mu3', 0.184824687,0.005)], 
                                            [self.p.value('3f2_rr_0', 0.0718810048),self.p.value('3f2_rr_1',-1.08216233),self.p.value('3f2_rr_2',-38.2507093)]],
                              rot_order=[[1,2],[1, 3],[1,4],[1,5],[1,6],[1,7],[2, 3],[2,4],[2,5],[2,6],[2,7],[8, 9], [8, 10], [8, 11],[8,12]],
                              rot_angles=[[self.p.value('13f3_th12_0', -0.0208481417),self.p.value('13f3_th12_2', 0.239045493)],[self.p.value('13f3_th13',-0.0593782568)],[self.p.value('13f3_th14',-0.0755947274)],[self.p.value('13f3_th15',0.122678758)],[self.p.value('13f3_th16',-0.0400418576)],[self.p.value('13f3_th17',0.0646373252)], [self.p.value('13f3_th23',-0.0684538786)],[self.p.value('13f3_th24', 0.0352130279)],[self.p.value('13f3_th25',-0.0326572035)],[self.p.value('13f3_th26',-0.050215299)],[self.p.value('13f3_th27',0.0453892695)],[self.p.value('3p2_th12', 0.0657620037,0.1)], [self.p.value('3p2_th13', 0.0215995148,0.03)], [self.p.value('3p2_th14',-0.028252844,0.03)], [self.p.value('3f2_th', 0.0184267375)]],
                              Uiabar=UiaFbar_F52, nulims=[[11], [0, 1, 7]], atom=self)

        self.mqdt_models.append({'L': 1, 'F': 5 / 2, 'model': MQDT_F52})
        self.channels.extend(mqdt_f52['channels'])

        mqdt_f72 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (a)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (b)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (c)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (d)', potential=self.model_pot),
            core_state((-1, -1, 1 / 2, 1 / 2, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                       config='4f135d6s (e)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_f72.update({
            'channels': [
                channel(mqdt_f72['cores'][0], (1 / 2, 3, 7 / 2), tt='slj'),
                channel(mqdt_f72['cores'][0], (1 / 2, 3, 5 / 2), tt='slj'),
                channel(mqdt_f72['cores'][1], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f72['cores'][2], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f72['cores'][3], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f72['cores'][4], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f72['cores'][5], (1 / 2, 1, 3 / 2), tt='slj', no_me=True),
                channel(mqdt_f72['cores'][6], (1 / 2, 3, 7 / 2), tt='slj'),
            ]})

        UiaFbar_F72 = np.identity(8)
        UiaFbar_F72[0, 0] = 0.56694671
        UiaFbar_F72[0, 1] = 0.49099025
        UiaFbar_F72[0, 7] = -0.66143783
        UiaFbar_F72[1, 0] = -0.65465367
        UiaFbar_F72[1, 1] = 0.75592895
        UiaFbar_F72[1, 7] = 0
        UiaFbar_F72[7, 0] = 1 / 2
        UiaFbar_F72[7, 1] = 0.4330127
        UiaFbar_F72[7, 7] = 3/4

        self.p.set_prefix('171YbF72')

        # this includes 1,3F3 MQDT of 174Yb from Lehec'c thesis and a 3F4 QDT model guess from the P F=3/2 residuals
        MQDT_F72 = mqdt_class(channels=mqdt_f72['channels'],
                              eig_defects=[[self.p.value('13f3_mu0',0.27707535), self.p.value('13f3_mu0_1',-13.2829133)], [self.p.value('13f3_mu1',0.0719827367), self.p.value('13f3_mu1_1',-0.741492064)], [self.p.value('13f3_mu2',0.250829254)],[self.p.value('13f3_mu3',0.227042016)],[self.p.value('13f3_mu4',0.17576256)],[self.p.value('13f3_mu5',0.19654694)],[self.p.value('13f3_mu6',0.214351625)], [self.p.value('3f4_rr_0',  0.0834066138), self.p.value('3f4_rr_2', -1.11349283), self.p.value('3f4_rr_4', -1539.63739)], ],
                              rot_order=[[1,2],[1, 3],[1,4],[1,5],[1,6],[1,7],[2, 3],[2,4],[2,5],[2,6],[2,7],],
                              rot_angles=[[self.p.value('13f3_th12_0', -0.0208481417),self.p.value('13f3_th12_2', 0.239045493)],[self.p.value('13f3_th13',-0.0593782568)],[self.p.value('13f3_th14',-0.0755947274)],[self.p.value('13f3_th15',0.122678758)],[self.p.value('13f3_th16',-0.0400418576)],[self.p.value('13f3_th17',0.0646373252)], [self.p.value('13f3_th23',-0.0684538786)],[self.p.value('13f3_th24', 0.0352130279)],[self.p.value('13f3_th25',-0.0326572035)],[self.p.value('13f3_th26',-0.050215299)],[self.p.value('13f3_th27',0.0453892695)],],
                              Uiabar=UiaFbar_F72, nulims=[[7], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 3, 'F': 7 / 2, 'model': MQDT_F72})
        self.channels.extend(mqdt_f72['channels'])


        self.p.set_prefix('171YbF92')

        # Guess for F F=9/2 QDT model. Which is purely Fc=1
        QDT_F92 = mqdt_class_rydberg_ritz(channels=mqdt_f52['channels'][0],
                                          deltas=[self.p.value('3f4_rr_0', 0.0834066138), self.p.value('3f4_rr_2', -1.11349283), self.p.value('3f4_rr_4', -1539.63739)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 3, 'F': 9 / 2, 'model': QDT_F92})

        # Guess for G F=7/2

        self.p.set_prefix('171YbG72')

        mqdt_g72 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_g72.update({
            'channels': [
                channel(mqdt_g72['cores'][0], (1 / 2, 4, 9 / 2), tt='slj'),
                channel(mqdt_g72['cores'][0], (1 / 2, 4, 7 / 2), tt='slj'),
                channel(mqdt_g72['cores'][1], (1 / 2, 4, 7 / 2), tt='slj'),
            ]})

        UiaFbar_G72 = np.identity(3)
        UiaFbar_G72[0, 0] = -1
        UiaFbar_G72[0, 1] = 0
        UiaFbar_G72[0, 2] = 0
        UiaFbar_G72[1, 0] = 0
        UiaFbar_G72[1, 1] = -0.66143783
        UiaFbar_G72[1, 2] = 0.75
        UiaFbar_G72[2, 0] = 0
        UiaFbar_G72[2, 1] = 0.75
        UiaFbar_G72[2, 2] = 0.66143783

        # this includes 1,3F3 MQDT of 174Yb from Lehec'c thesis and a 3F2 QDT model guess from the P F=3/2 residuals
        MQDT_G72 = mqdt_class(channels=mqdt_g72['channels'],
                              eig_defects=[[self.p.value('1g4_rr_0', 0.02627822),self.p.value('1g4_rr_1', -0.13288481)],
                                           [self.p.value('3g4_rr_0', 0.02547095),self.p.value('3g4_rr_1', -0.11668088)],[self.p.value('3g3_rr_0', 0.02611627),self.p.value('3g3_rr_1', -0.13079392)], ],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('13g4_th12', -0.08222676)],],
                              Uiabar=UiaFbar_G72, nulims=[[2], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 4, 'F': 7 / 2, 'model': MQDT_G72})
        self.channels.extend(mqdt_g72['channels'])

        # Guess for G F=9/2

        self.p.set_prefix('171YbG92')

        mqdt_g92 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_g92.update({
            'channels': [
                channel(mqdt_g92['cores'][0], (1 / 2, 4, 9 / 2), tt='slj'),
                channel(mqdt_g92['cores'][0], (1 / 2, 4, 7 / 2), tt='slj'),
                channel(mqdt_g92['cores'][1], (1 / 2, 4, 9 / 2), tt='slj'),
            ]})

        UiaFbar_G92 = np.identity(3)
        UiaFbar_G92[0, 0] = 0.74161985
        UiaFbar_G92[0, 1] = 0
        UiaFbar_G92[0, 2] = -0.67082039
        UiaFbar_G92[1, 0] = 0
        UiaFbar_G92[1, 1] = 1
        UiaFbar_G92[1, 2] = 0
        UiaFbar_G92[2, 0] = 0.67082039
        UiaFbar_G92[2, 1] = 0
        UiaFbar_G92[2, 2] = 0.74161985

        # this includes 1,3F3 MQDT of 174Yb from Lehec'c thesis and a 3F2 QDT model guess from the P F=3/2 residuals
        MQDT_G92 = mqdt_class(channels=mqdt_g92['channels'],
                              eig_defects=[[self.p.value('1g4_rr_0', 0.02627822),self.p.value('1g4_rr_1', -0.13288481)],
                                           [self.p.value('3g4_rr_0', 0.02547095),self.p.value('3g4_rr_1', -0.11668088)], [self.p.value('3g5_rr_0', 0.02536987),self.p.value('3g5_rr_1', -0.2007948)], ],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('13g4_th12', -0.08222676)], ],
                              Uiabar=UiaFbar_G92, nulims=[[2], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 4, 'F': 9 / 2, 'model': MQDT_G92})
        self.channels.extend(mqdt_g92['channels'])

        self.p.set_prefix('171YbG52')

        # Guess for G F=5/2 QDT model. Which is purely Fc=1
        QDT_G52 = mqdt_class_rydberg_ritz(channels=mqdt_g72['channels'][1],
                                          deltas=[self.p.value('3g3_rr_0', 0.02611627),self.p.value('3g3_rr_1', -0.13079392)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 4, 'F': 5 / 2, 'model': QDT_G52})

        self.p.set_prefix('171YbG112')

        # Guess for G F=11/2 QDT model. Which is purely Fc=1
        QDT_G112 = mqdt_class_rydberg_ritz(channels=mqdt_g72['channels'][0],
                                          deltas=[self.p.value('3g5_rr_0', 0.02536987),self.p.value('3g5_rr_1', -0.2007948)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 4, 'F': 11 / 2, 'model': QDT_G112})

        # Guess for H F=9/2 NOTE only the 1H5 series has been observed in 174Yb

        self.p.set_prefix('171YbH92')

        mqdt_h92 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_h92.update({
            'channels': [
                channel(mqdt_h92['cores'][0], (1 / 2, 5, 11 / 2), tt='slj'),
                channel(mqdt_h92['cores'][0], (1 / 2, 5, 9 / 2), tt='slj'),
                channel(mqdt_h92['cores'][1], (1 / 2, 5, 9 / 2), tt='slj'),
            ]})

        UiaFbar_H92 = np.identity(3)
        UiaFbar_H92[0, 0] = -1
        UiaFbar_H92[1, 1] = -0.67082039
        UiaFbar_H92[1, 2] = 0.74161985
        UiaFbar_H92[2, 1] = 0.74161985
        UiaFbar_H92[2, 2] = 0.67082039

        # 
        MQDT_H92 = mqdt_class(channels=mqdt_h92['channels'],
                              eig_defects=[[self.p.value('1h5_rr_0', 0.009305),self.p.value('1h5_rr_2', -0.073)],
                                           [self.p.value('3h5_rr_0', 0.009205),self.p.value('3h5_rr_2', -0.073)],[self.p.value('3h4_rr_0', 0.009305),self.p.value('3h4_rr_2', -0.073)], ],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('13h5_th12', 1e-3)],],
                              Uiabar=UiaFbar_H92, nulims=[[2], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 5, 'F': 9 / 2, 'model': MQDT_H92})
        self.channels.extend(mqdt_h92['channels'])

        # Guess for H F=11/2

        self.p.set_prefix('171YbH112')

        mqdt_h112 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_h112.update({
            'channels': [
                channel(mqdt_h112['cores'][0], (1 / 2, 5, 11 / 2), tt='slj'),
                channel(mqdt_h112['cores'][0], (1 / 2, 5, 9 / 2), tt='slj'),
                channel(mqdt_h112['cores'][1], (1 / 2, 5, 11 / 2), tt='slj'),
            ]})

        UiaFbar_H112 = np.identity(3)
        UiaFbar_H112[0, 0] = 0.73598007
        UiaFbar_H112[0, 2] = -0.6770032
        UiaFbar_H112[1, 1] = 1
        UiaFbar_H112[2, 0] = 0.6770032 
        UiaFbar_H112[2, 2] = 0.73598007

        # this includes 1,3F3 MQDT of 174Yb from Lehec'c thesis and a 3F2 QDT model guess from the P F=3/2 residuals
        MQDT_H112 = mqdt_class(channels=mqdt_h112['channels'],
                              eig_defects=[[self.p.value('1h5_rr_0', 0.009305),self.p.value('1h5_rr_2', -0.073)],
                                           [self.p.value('3h5_rr_0', 0.009205),self.p.value('3h5_rr_2', -0.073)], [self.p.value('3h6_rr_0', 0.009305),self.p.value('3h6_rr_2', -0.073)], ],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('13h5_th12', 1e-3)], ],
                              Uiabar=UiaFbar_H112, nulims=[[2], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 5, 'F': 11 / 2, 'model': MQDT_H112})
        self.channels.extend(mqdt_h112['channels'])

        self.p.set_prefix('171YbH72')

        # Guess for H F=7/2 QDT model. Which is purely Fc=1
        QDT_H72 = mqdt_class_rydberg_ritz(channels=mqdt_h92['channels'][1],
                                          deltas=[self.p.value('3h4_rr', 0.009305),self.p.value('3h4_rr_2', -0.073)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 5, 'F': 7 / 2, 'model': QDT_H72})

        self.p.set_prefix('171YbH132')

        # Guess for H F=13/2 QDT model. Which is purely Fc=1
        QDT_H132 = mqdt_class_rydberg_ritz(channels=mqdt_h92['channels'][0],
                                          deltas=[self.p.value('3h6_rr', 0.009305),self.p.value('3h6_rr_2', -0.073)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 5, 'F': 13 / 2, 'model': QDT_H132})

        # Guess for I F=11/2 NOTE only the 1I6 series has been observed in 174Yb

        self.p.set_prefix('171YbI112')

        mqdt_i112 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_i112.update({
            'channels': [
                channel(mqdt_i112['cores'][0], (1 / 2, 6, 13 / 2), tt='slj'),
                channel(mqdt_i112['cores'][0], (1 / 2, 6, 11 / 2), tt='slj'),
                channel(mqdt_i112['cores'][1], (1 / 2, 6, 11 / 2), tt='slj'),
            ]})

        UiaFbar_I112 = np.identity(3)
        UiaFbar_I112[0, 0] = -1
        UiaFbar_I112[1, 1] = -0.6770032
        UiaFbar_I112[1, 2] = 0.73598007
        UiaFbar_I112[2, 1] = 0.73598007
        UiaFbar_I112[2, 2] = 0.6770032

        # 
        MQDT_I112 = mqdt_class(channels=mqdt_i112['channels'],
                              eig_defects=[[self.p.value('1i6_rr_0', 0.004062),self.p.value('1i6_rr_2', -0.128)],
                                           [self.p.value('3i6_rr_0', 0.004052),self.p.value('3i6_rr_2', -0.128)],[self.p.value('3i5_rr_0', 0.004062),self.p.value('3i5_rr_2', -0.128)], ],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('13i6_th12', 1e-3)],],
                              Uiabar=UiaFbar_I112, nulims=[[2], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 6, 'F': 11 / 2, 'model': MQDT_I112})
        self.channels.extend(mqdt_i112['channels'])

        # Guess for I F=13/2

        self.p.set_prefix('171YbI132')

        mqdt_i132 = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        mqdt_i132.update({
            'channels': [
                channel(mqdt_i132['cores'][0], (1 / 2, 6, 13 / 2), tt='slj'),
                channel(mqdt_i132['cores'][0], (1 / 2, 6, 11 / 2), tt='slj'),
                channel(mqdt_i132['cores'][1], (1 / 2, 6, 13 / 2), tt='slj'),
            ]})

        UiaFbar_I132 = np.identity(3)
        UiaFbar_I132[0, 0] = 0.73192505
        UiaFbar_I132[0, 2] = -0.68138514
        UiaFbar_I132[1, 1] = 1
        UiaFbar_I132[2, 0] = 0.68138514
        UiaFbar_I132[2, 2] = 0.73192505

        # this includes 1,3F3 MQDT of 174Yb from Lehec'c thesis and a 3F2 QDT model guess from the P F=3/2 residuals
        MQDT_I132 = mqdt_class(channels=mqdt_i132['channels'],
                              eig_defects=[[self.p.value('1i6_rr_0', 0.004062),self.p.value('1i6_rr_2', -0.128)],
                                           [self.p.value('3i6_rr_0', 0.004052),self.p.value('3i6_rr_2', -0.128)], [self.p.value('3i7_rr_0', 0.004062),self.p.value('3i7_rr_2', -0.128)], ],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('13i6_th12', 1e-3)], ],
                              Uiabar=UiaFbar_I132, nulims=[[2], [0, 1]], atom=self)

        self.mqdt_models.append({'L': 6, 'F': 13 / 2, 'model': MQDT_I132})
        self.channels.extend(mqdt_i132['channels'])

        self.p.set_prefix('171YbI92')

        # Guess for I F=9/2 QDT model. Which is purely Fc=1
        QDT_I92 = mqdt_class_rydberg_ritz(channels=mqdt_i112['channels'][1],
                                          deltas=[self.p.value('3i5_rr', 0.004062),self.p.value('3i5_rr_2', -0.128)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 6, 'F': 9 / 2, 'model': QDT_I92})

        self.p.set_prefix('171YbI152')

        # Guess for I F=15/2 QDT model. Which is purely Fc=1
        QDT_I152 = mqdt_class_rydberg_ritz(channels=mqdt_i112['channels'][0],
                                          deltas=[self.p.value('3i7_rr', 0.004062),self.p.value('3i7_rr_2', -0.128)], atom=self, HFlimit="upper")
        self.mqdt_models.append({'L': 6, 'F': 15 / 2, 'model': QDT_I152})
        
        super().__init__(**kwargs)



    def get_state(self, qn, tt='vlfm', energy_exp_Hz=None, energy_only=False,whittaker_wfct=False):
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
            # If quantum numbers are valid, create a new MQDT model, else return None
            if np.abs(f-l)<=1.51 and isinstance(l, int):
                #print("new model is created")
                self.create_high_l_MQDT(qn)
                solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
            else:
                #raise ValueError(f"Could not find MQDT model for qn={qn}")
                print()
                return None
                #continue  

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
        st.whittaker_wfct = whittaker_wfct

        return st


    def get_state_nist(self, qn, tt='nsljfm',whittaker_wfct=False):

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
            st.whittaker_wfct = whittaker_wfct

            return st

        else:
            print("tt=", tt, " not supported by H.get_state")


    def create_high_l_MQDT(self,qn):
        """
        Creates a high-angular-momentum (high-l) MQDT (Multichannel Quantum Defect Theory) model for alkaline atoms  with nuclear spin (e.g., Yb-171).

        This function is called when a requested state has quantum numbers (l, f) not already present in the MQDT model list,
        typically for l > 4. It constructs the necessary core states and channels, and calculates quantum defect parameters
        including dipole and quadrupole polarizability, relativistic corrections, and spin-orbit coupling effects.
        The model assumes high-l states are jj-coupled.

        Parameters:
            qn (list): Quantum numbers for the state, typically [n, l, f, m], where
                n (float): Principal quantum number or effective quantum number.
                l (int): Orbital angular momentum quantum number.
                f (float): Total angular momentum quantum number.
                m (float): Magnetic quantum number.

        Behavior:
            - Defines core states and channels for the requested high-l value.
            - Calculates quantum defect parameters using atomic constants and polarizabilities.
            - Handles regular and indirect spin-orbit coupling corrections.
            - Adds the constructed MQDT model to self.mqdt_models and extends self.channels.
            - Supports cases where f = l+1/2, f = l-1/2, f = l-1-1/2, and f = l+1+1/2.

        Returns:
            None. The new MQDT model is added to the object's model list for future use.

        Notes:
            - This function is automatically called by get_state if a high-l model is needed.
            - The implementation is based on analytic formulas for quantum defects and coupling, and is suitable for alkaline-earth atoms like Yb.
            - The function includes detailed handling of channel mixing, frame transformations, and odd-power corrections for quantum defects.
            NOTE: this is a function still under development. Not tested for edge cases, such as l>nu for series converging to closely spaced ionization limits
        """

        n = qn[0]
        v = qn[0]
        l = qn[1]
        f = qn[2]
        m = qn[3]

        # QDT high l channels
        self.p.set_prefix('171Yb_'+str(l))


        qdt = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 1), Ei_Hz=0.25 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=1)', potential=self.model_pot),
            core_state((1 / 2, 0, 1 / 2, 1 / 2, 0), Ei_Hz=-0.75 * self.ion_hyperfine_6s_Hz, tt='sljif',
                       config='6s1/2 (Fc=0)', potential=self.model_pot),
        ]}

        qdt.update({
            'channels': [
                channel(qdt['cores'][0], (1 / 2, l, l + 1 / 2), tt='slj'),
                channel(qdt['cores'][0], (1 / 2, l, l - 1 / 2), tt='slj'),
                channel(qdt['cores'][1], (1 / 2, l, l + 1 / 2), tt='slj'),
                channel(qdt['cores'][1], (1 / 2, l, l - 1 / 2), tt='slj'),
            ]})

        # calculate high-l quantum defects, this includes dipole and quadrupole polarizability, relativistic corrections, and "direct" and "indirect" spin-orbit coupling
        # fine structure terms as summarized in Lundeen, ADVANCES IN ATOMIC, MOLECULAR AND OPTICAL PHYSICS, VOL. 52, 161 (2005) (https://www.sciencedirect.com/science/article/pii/S1049250X05520044)
        A4 = 1/(2*(l-1/2)*l*(l+1/2)*(l+1)*(l+3/2))
        A6 = A4 / (4*(l-3/2)*(l-1)*(l+2)*(l+5/2))

        A36 = 8*(l-3/2)*(l-1)*(l-1/2)*(l+3/2)*(l+2)*(l+5/2)
        A38 = 16*(l-5/2)*(l-2)*(l-3/2)*(l-1)*(l-1/2)*(l+3/2)*(l+2)*(l+5/2)*(l+3)*(l+7/2)

        # dipole and quadrupole coefficients in indirect spin-orbit coupling: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.68.022510
        bdprefac = (2/3)*(self.deltaEp_m/self.Ep_m**2)*qdt['cores'][0].alpha_d_a03/cs.alpha**2*cs.physical_constants['hartree-inverse meter relationship'][0]
        bqprefac = (5/6)*(self.deltaEd_m/self.Ed_m**2)*qdt['cores'][0].alpha_q_a05/cs.alpha**2*cs.physical_constants['hartree-inverse meter relationship'][0]

        # collecting terms all spin-orbit terms for the quantum defect
        qd_regularSO = (1/2)*cs.alpha**2/(l*(l+1))
        qd_indirectSO_0 = cs.alpha**2/(l*(l+1))*(1-bdprefac*35/A36-bqprefac*231/A38) 
        qd_indirectSO_2 = cs.alpha**2/(l*(l+1))*(-bdprefac*(25-30*l*(l+1))/A36-bqprefac*(735-315*l*(l+1))/A38)
        qd_indirectSO_4 = cs.alpha**2/(l*(l+1))*(-bdprefac*(3*(l-1)*l*(l+1)*(l+2))/A36-bqprefac*(105*(l-1)*l*(l+1)*(l+2)-315*l*(l+1)+294)/A38)

        # dipole and quadrupole polarizability and relativistic corrections to the quantum defect of high l states
        # the relativistic kinetic energy corection introduces odd (1/n) terms in the energy-dependence expansion of the quantum defect
        mu0 = (3/2)*A4*qdt['cores'][0].alpha_d_a03 + (35/2)*A6*qdt['cores'][0].alpha_q_a05 + cs.alpha**2/(2*(l+1/2))
        mu1 = -(3/8)*cs.alpha**2
        mu2 = (-1/2)*(l*(l+1)*A4*qdt['cores'][0].alpha_d_a03 + 5*(6*l**2+6*l-5)*A6*qdt['cores'][0].alpha_q_a05)
        mu3 = 0
        mu4 = (3/2)*(l-1)*l*(l+1)*(l+2)*A6*qdt['cores'][0].alpha_q_a05

        #print([mu0, mu1, mu2, mu3, mu4])
        if l>4:
            if f == l+1/2:

                # The G channels are closer to being jj coupled thatn they are to being LS coupled, lets treat the high l channels the same way

                jj = [[l+1/2,l],[l+1/2,l+1]]
                jjF = [[l+1/2,1,f],[l+1/2,0,f]]
                Uiabar= self.UiFi(jjF,jj,1/2)

                weightdirect_0 = (2*l)/(4*l + 2)
                weight_indirect_0 = -(2*l+2)/(4*l + 2)

                weightdirect_1 = -(2*l+2)/(4*l + 2)
                weight_indirect_1 = (2*l)/(4*l + 2)

                
                QDT_mLL = mqdt_class_rydberg_ritz(channels=qdt['channels'][1],
                                                deltas=[self.p.value('pLL_mu0',mu0+weightdirect_0*qd_regularSO-weight_indirect_0*qd_indirectSO_0), self.p.value('pLL_mu0_1',mu1), self.p.value('pLL_mu0_2',mu2-weight_indirect_0*qd_indirectSO_2), self.p.value('pLL_mu0_3',mu3), self.p.value('pLL_mu0_4',mu4-weight_indirect_0*qd_indirectSO_4)], atom=self,odd_powers=True)
                
                
                 
                MQDT_pmLL = mqdt_class(channels=[qdt['channels'][0],qdt['channels'][2]],
                                    eig_defects=[[self.p.value('pLL_mu0',mu0+weightdirect_0*qd_regularSO-weight_indirect_0*qd_indirectSO_0), self.p.value('pLL_mu0_1',mu1), self.p.value('pLL_mu0_2',mu2-weight_indirect_0*qd_indirectSO_2), self.p.value('pLL_mu0_3',mu3), self.p.value('pLL_mu0_4',mu4-weight_indirect_0*qd_indirectSO_4)], [self.p.value('mLL_mu1',mu0+weightdirect_1*qd_regularSO-weight_indirect_1*qd_indirectSO_0), self.p.value('mLL_mu1_1',mu1), self.p.value('mLL_mu1_2',mu2-weight_indirect_1*qd_indirectSO_2), self.p.value('mLL_mu1_3',mu3), self.p.value('mLL_mu1_4',mu4-weight_indirect_1*qd_indirectSO_4)]],
                                    rot_order=[[1,2]],
                                    rot_angles=[[self.p.value('pmLL_th12',1e-2)],],# the code has issues with fully uncoupled channels. We introduce a small angle (essentially 0) angle to avoid this issue to be fixed
                                    Uiabar=Uiabar, nulims=[[1],[0]],atom=self,odd_powers=True)        
                        
                MQDT_Lp12 = mqdt_class_wrapper([QDT_mLL,MQDT_pmLL])

                self.mqdt_models.append({'L': l, 'F': f, 'model': MQDT_Lp12})
                self.channels.extend(qdt['channels'])

            elif f==l-1/2:


                weightdirect_0 = (2*l)/(4*l + 2)
                weight_indirect_0 = -(2*l+2)/(4*l + 2)

                weightdirect_1 = -(2*l+2)/(4*l + 2)
                weight_indirect_1 = (2*l)/(4*l + 2)
            

                QDT_pLL = mqdt_class_rydberg_ritz(channels=qdt['channels'][0],
                                                deltas=[self.p.value('mLL_mu1',mu0+weightdirect_1*qd_regularSO-weight_indirect_1*qd_indirectSO_0), self.p.value('mLL_mu1_1',mu1), self.p.value('mLL_mu1_2',mu2-weight_indirect_1*qd_indirectSO_2), self.p.value('mLL_mu1_3',mu3), self.p.value('mLL_mu1_4',mu4-weight_indirect_1*qd_indirectSO_4)], atom=self,odd_powers=True)
 

                jj = [[l-1/2,l],[l-1/2,l-1]]
                jjF = [[l-1/2,1,f],[l-1/2,0,f]]
                Uiabar= self.UiFi(jjF,jj,1/2)

                 
                MQDT_pmLL = mqdt_class(channels=[qdt['channels'][1],qdt['channels'][3]],
                                    eig_defects=[[self.p.value('pLL_mu0',mu0+weightdirect_0*qd_regularSO-weight_indirect_0*qd_indirectSO_0), self.p.value('pLL_mu0_1',mu1), self.p.value('pLL_mu0_2',mu2-weight_indirect_0*qd_indirectSO_2), self.p.value('pLL_mu0_3',mu3), self.p.value('pLL_mu0_4',mu4-weight_indirect_0*qd_indirectSO_4)], [self.p.value('mLL_mu1',mu0+weightdirect_1*qd_regularSO-weight_indirect_1*qd_indirectSO_0), self.p.value('mLL_mu1_1',mu1), self.p.value('mLL_mu1_2',mu2-weight_indirect_1*qd_indirectSO_2), self.p.value('mLL_mu1_3',mu3), self.p.value('mLL_mu1_4',mu4-weight_indirect_1*qd_indirectSO_4)]],
                                    rot_order=[[1,2]],
                                    rot_angles=[[self.p.value('pmLL_th12',1e-2)],],# the code has issues with fully uncoupled channels. We introduce a small angle (essentially 0) angle to avoid this issue to be fixed
                                    Uiabar=Uiabar, nulims=[[1],[0]],atom=self,odd_powers=True)      
                        
                MQDT_Lm12 = mqdt_class_wrapper([QDT_pLL,MQDT_pmLL])

                self.mqdt_models.append({'L': l, 'F': f, 'model': MQDT_Lm12})
                self.channels.extend(qdt['channels'])

                    
            elif f==l-1-1/2:
                weightdirect = (2*l)/(4*l + 2)
                weight_indirect = (2*l)/(4*l + 2)


                QDT_3LLm1 = mqdt_class_rydberg_ritz(channels=qdt['channels'][1],
                                                deltas=[self.p.value('3LLm1_rr_0',mu0+weightdirect*qd_regularSO-weight_indirect*qd_indirectSO_0),self.p.value('3LLm1_rr_1',mu1),self.p.value('3LLm1_rr_2',mu2-weight_indirect*qd_indirectSO_2),self.p.value('3LLm1_rr_3',mu3),self.p.value('3LLm1_rr_4',mu4-weight_indirect*qd_indirectSO_4)], atom=self,odd_powers=True,HFlimit='upper')

                self.mqdt_models.append({'L': l, 'F': f, 'model': QDT_3LLm1})
                self.channels.extend(qdt['channels'])

            elif f==l+1+1/2:

                weightdirect = -(2*l + 2)/(4*l + 2)
                weight_indirect = -(2*l + 2)/(4*l + 2)

                QDT_3LLp1 = mqdt_class_rydberg_ritz(channels=qdt['channels'][0],
                                                deltas=[self.p.value('3LLp1_rr_0',mu0+weightdirect*qd_regularSO-weight_indirect*qd_indirectSO_0),self.p.value('3LLp1_rr_1',mu1),self.p.value('3LLp1_rr_2',mu2-weight_indirect*qd_indirectSO_2),self.p.value('3LLp1_rr_3',mu3),self.p.value('3LLp1_rr_4',mu4-weight_indirect*qd_indirectSO_4)], atom=self,odd_powers=True,HFlimit='upper')

                self.mqdt_models.append({'L': l, 'F': f, 'model': QDT_3LLp1})
                self.channels.extend(qdt['channels'])

        pass

    def get_nearby(self, st, include_opts={}, energy_only = False,whittaker_wfct=False):
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
                    # If quantum numbers are valid, create a new MQDT model, else return None
                    if np.abs(f-l)<=1.51 and l>4:
                        #print("new model is created")
                        self.create_high_l_MQDT([st.nub,l,f,st.m])
                        solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
                    else:
                        #raise ValueError(f"Could not find MQDT model for qn={qn}")
                        continue
                        #continue  

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
                            st_new.whittaker_wfct = whittaker_wfct
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
    
    def UiFi(self,jjF,jj,Ii):

        """
        Constructs the unitary transformation matrix between jj-coupled and hyperfine-coupled basis states.

        This function computes the transformation matrix (UiFi) that relates basis states labeled by
        total angular momentum quantum numbers in the jj-coupling scheme (jj) to those in the hyperfine-coupling
        scheme (jjF), for a given nuclear spin Ii. The transformation elements are calculated using Wigner 6-j symbols.

        Parameters:
            jjF (list): List of basis states in the hyperfine-coupled scheme, each as [j', Fc, F],
                        where j' is the electronic angular momentum, Fc is the core angular momentum,
                        and F is the total angular momentum.
            jj (list): List of basis states in the jj-coupled scheme, each as [j, J],
                       where j is the electronic angular momentum and J is the total angular momentum.
            Ii (float): Nuclear spin quantum number.

        Returns:
            UiFi (ndarray): Unitary transformation matrix of shape (len(jjF), len(jj)), with elements
                            given by the appropriate Wigner 6-j coefficients.

        Notes:
            - The transformation is only computed if jjF and jj have the same length.
            - If the input lists have unequal lengths, a warning is printed and no matrix is returned.
            - This matrix is used for frame transformations between different angular momentum coupling schemes
              in multichannel quantum defect theory (MQDT) calculations for atoms with nuclear spin.
        """
    
        if len(jjF) == len(jj):
            UiFi = np.zeros([len(jjF),len(jj)])
            
            for (n,i) in enumerate(jj):
                for (m,k) in enumerate(jjF):

                    j = i[0]
                    J = i[1]

                    jdash = k[0]
                    Fc = k[1]   
                    F = k[2]
                    Jc = 1/2
            
                    if j == jdash:
                        UiFi[m,n] = (-1)**(j+Fc+Ii+J)*np.sqrt((2*Fc+1)*(2*J+1))*wigner_6j(Ii,Jc,Fc,j,F,J)

            return UiFi
        else:
            print("Unequal lengths jjF-jj vectors")
    
    