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

        self.citations = ['Peper2024Spectroscopy', 'Aymar1984three', 'Meggers1970First', 'Camus1980Highly', 'Camus1969spectre', 'Meggers1970First', 'Wyart1979Extended', 'Aymar1980Highly', 'Camus1980Highly', 'Aymar1984three', 'Martin1978Atomic', 'BiRu1991The', 'Maeda1992Optical', 'zerne1996lande', 'Ali1999Two', 'Lehec2017PhD', 'Lehec2018Laser', 'Niyaz2019Microwave']

        my_params = {
                     }

        if params is not None:
            my_params.update(params)

        self.p = model_params(my_params)


        self.mqdt_models = []
        self.channels = []

        self.Elim_cm = 50443.070393
        self.Elim_THz = self.Elim_cm*cs.c*10**(-10) + self.p.value('174Yb_Elim_offset_MHz', 0, 1) * 1e-6


        # Center-of-mass energy and fine structure splitting of lowest p and d states of the Yb+ ion for calculation of high-l fine structure
        self.deltaEp_m = 100*(30392.23 - 27061.82)
        self.Ep_m = 100*((4/6)*30392.23 + (2/6)*27061.82)

        self.deltaEd_m = 100*(24332.69 - 22960.80)
        self.Ed_m = 100*((3/5)*24332.69 + (2/5)*22960.80)



        # For 1S0 MQDT model

        mqdt_1S0 = {'cores': [
                        core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                   config='6s1/2', potential=self.model_pot,alpha_d_a03 = 60.51, alpha_q_a05 = 672),
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
        Uiabar1S0[2, 2] = np.sqrt(2 / 3)
        Uiabar1S0[4, 4] = np.sqrt(2 / 3)
        Uiabar1S0[2, 4] = -np.sqrt(1 / 3)
        Uiabar1S0[4, 2] = np.sqrt(1 / 3)

        self.p.set_prefix('174Yb_1s0')

        MQDT_1S0 = mqdt_class(channels=mqdt_1S0['channels'],
                              eig_defects=[[self.p.value('mu0',0.355101645), self.p.value('mu0_1',0.277673956)], [self.p.value('mu1',0.204537535)],
                                           [self.p.value('mu2',0.116393648)], [self.p.value('mu3',0.295439966)], [self.p.value('mu4',0.257664798)],
                                           [self.p.value('mu5',0.155797119)]],
                              rot_order=[[1, 2], [1, 3], [1, 4], [3, 4], [3, 5], [1, 6]],
                              rot_angles=[[self.p.value('th12',0.126557575)], [self.p.value('th13',0.300103593)], [self.p.value('th14',0.056987912)],
                                          [self.p.value('th34',0.114312578)],[self.p.value('th35',0.0986363362)], [self.p.value('th16',0.142498543)]],
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
                              eig_defects=[[self.p.value('3p0_mu0',0.953661478), self.p.value('3p0_mu0_1',-0.287531374)], [self.p.value('3p0_mu1',0.198460766)]],
                              rot_order=[[1, 2]],
                              rot_angles=[[self.p.value('3p0_th12',0.163343232)]],
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
                               eig_defects=[[self.p.value('13p1_mu0', 0.922709076), self.p.value('13p1_mu0_1', 2.60055203)],
                                            [self.p.value('13p1_mu1', 0.982084772), self.p.value('13p1_mu1_1', -5.45063476)], [self.p.value('13p1_mu2', 0.228518316)],
                                            [self.p.value('13p1_mu3', 0.206081775)], [self.p.value('13p1_mu4', 0.193527605)], [self.p.value('13p1_mu5', 0.181533031)]],
                               rot_order=[[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6]],
                               rot_angles=[[self.p.value('13p1_th12_0', -8.41087098e-02), self.p.value('13p1_th12_2', 1.20375554e+02), self.p.value('13p1_th12_4', -9.31423120e+03)], [self.p.value('13p1_th13', -0.0731798557)], [self.p.value('13p1_th14', -0.06651879)], [self.p.value('13p1_th15', -0.022121936)], [self.p.value('13p1_th16', -0.104521091)], [self.p.value('13p1_th23', 0.0247746449)], [self.p.value('13p1_th24', 0.0576393392)], [self.p.value('13p1_th25', 0.0860644)], [self.p.value('13p1_th26', 0.0499381827)]],
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
                                    eig_defects=[[self.p.value('3p2_mu0', 0.925150932), self.p.value('3p2_mu0_1', -2.69197178), self.p.value('3p2_mu0_2', 66.7159709)],
                                                  [self.p.value('3p2_mu1', 0.230028034)],
                                                 [self.p.value('3p2_mu2', 0.209224174)],
                                                 [self.p.value('3p2_mu3', 0.186236574)]],
                                    rot_order=[[1, 2], [1, 3],[1,4]],
                                    rot_angles=[[self.p.value('3p2_th12', 0.0706189664)], [self.p.value('3p2_th13', 0.0231221428)], [self.p.value('3p2_th14',  -0.0291730345)]],
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
                               eig_defects=[[self.p.value('13d2_mu0',0.729513646), self.p.value('13d2_mu0_1',-0.0377841183)],
                                            [self.p.value('13d2_mu1',0.752292223), self.p.value('13d2_mu1_1', 0.104072325)],
                                            [self.p.value('13d2_mu2', 0.19612036)], [self.p.value('13d2_mu3',0.233752026)],[self.p.value('13d2_mu4',0.152911249)]],
                               rot_order=[[1,2],[1, 3], [1, 4], [2, 4], [1, 5], [2, 5]],
                               rot_angles=[[self.p.value('13d2_th12_0',0.21157531),self.p.value('13d2_th12_2',-15.38440215)],[self.p.value('13d2_th13',0.00521559431)], [self.p.value('13d2_th14', 0.0398131577)],
                                           [self.p.value('13d2_th24',-0.0071658109)], [self.p.value('13d2_th15',0.10481227)], [self.p.value('13d2_th25',0.0721660042)]],
                               Uiabar=Uiabar13D2, nulims=[[4],[0, 1]],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 2, 'model': MQDT_13D2})
        self.channels.extend(mqdt_13D2['channels'])

        # 3D1 Rydberg Ritz formula
        QDT_3D1 = mqdt_class_rydberg_ritz(channels=mqdt_13D2['channels'][1],
                                          deltas=[self.p.value('3d1_rr_0',2.75258093), self.p.value('3d1_rr_1',0.382628525,1), self.p.value('3d1_rr_2',-483.120633,100)],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 1, 'model': QDT_3D1})

        # 3D3 Rydberg Ritz formula
        QDT_3D3 = mqdt_class_rydberg_ritz(channels=mqdt_13D2['channels'][0],
                                          deltas=[self.p.value('3d3_rr_0',2.72902016), self.p.value('3d3_rr_1',-0.705328923,1), self.p.value('3d3_rr_2',829.238844,100)],atom=self)

        self.mqdt_models.append({'L': 2, 'F': 3, 'model': QDT_3D3})

        # For 1,3F3 MQDT model
        mqdt_13F3 = {'cores': [
                         core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                                    config='6s1/2', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (a)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (b)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (d)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (d)', potential=self.model_pot),
                         core_state((-1, -1, 1 / 2, 0, 0), Ei_Hz=(83967.7 - self.Elim_cm) * cs.c * 100, tt='sljif',
                                    config='4f135d6s (d)', potential=self.model_pot),
                     ]}

        mqdt_13F3.update({
            'channels': [
                channel(mqdt_13F3['cores'][0], (1 / 2, 3, 7 / 2), tt='slj'),
                channel(mqdt_13F3['cores'][0], (1 / 2, 3, 5 / 2), tt='slj'),
                channel(mqdt_13F3['cores'][1], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_13F3['cores'][2], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_13F3['cores'][3], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_13F3['cores'][4], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
                channel(mqdt_13F3['cores'][5], (1 / 2, 1, 1 / 2), tt='slj', no_me=True),
            ]})

        Uiabar13F3 = np.identity(7)
        Uiabar13F3[0, 0] = np.sqrt(4 / 7)
        Uiabar13F3[0, 1] = np.sqrt(3 / 7)
        Uiabar13F3[1, 0] = -np.sqrt(3 / 7)
        Uiabar13F3[1, 1] = np.sqrt(4 / 7)

        MQDT_13F3 = mqdt_class(channels=mqdt_13F3['channels'],
                               eig_defects=[[self.p.value('13f3_mu0',0.276158949), self.p.value('13f3_mu0_1',-12.7258012)], [self.p.value('13f3_mu1',0.0715123712), self.p.value('13f3_mu1_1',-0.768462937)], [self.p.value('13f3_mu2',0.239015576)],[self.p.value('13f3_mu3',0.226770354)],[self.p.value('13f3_mu4',0.175354845)],[self.p.value('13f3_mu5',0.196660618)],[self.p.value('13f3_mu6',0.21069642)],],
                               rot_order=[[1,2],[1, 3],[1,4],[1,5],[1,6],[1,7],[2, 3],[2,4],[2,5],[2,6],[2,7]],
                               rot_angles=[[self.p.value('13f3_th12_0', -0.0209955122),self.p.value('13f3_th12_2', 0.251041249)],[self.p.value('13f3_th13',-0.00411835457)],[self.p.value('13f3_th14',-0.0962784945)],[self.p.value('13f3_th15',0.132826901)],[self.p.value('13f3_th16',-0.0439244317)],[self.p.value('13f3_th17',0.0508460294)], [self.p.value('13f3_th23',-0.0376574252)],[self.p.value('13f3_th24',0.026944623)],[self.p.value('13f3_th25',-0.0148474857)],[self.p.value('13f3_th26',-0.0521244126)],[self.p.value('13f3_th27',0.0349516329)],],
                               Uiabar=Uiabar13F3, nulims=[[2,3,4,5,6],[0, 1]],atom=self)


        self.mqdt_models.append({'L': 3, 'F': 3, 'model': MQDT_13F3})
        self.channels.extend(mqdt_13F3['channels'])

        #3F2 Rydberg Ritz formula
        QDT_3F2 = mqdt_class_rydberg_ritz(channels=mqdt_13F3['channels'][1],
                                         deltas=[self.p.value('3f2_rr_0', 0.0718252326),self.p.value('3f2_rr_1', -1.00091963),self.p.value('3f2_rr_2', -106.291066)], atom=self)

        self.mqdt_models.append({'L': 3, 'F': 2, 'model': QDT_3F2})

        # 3F4 Rydberg Ritz formula
        QDT_3F4 = mqdt_class_rydberg_ritz(channels=mqdt_13F3['channels'][0],
                                          deltas=[self.p.value('3f4_rr_0', 0.0839027969), self.p.value('3f4_rr_2', -2.91009023)], atom=self)

        self.mqdt_models.append({'L': 3, 'F': 4, 'model': QDT_3F4})

        # QDT G channels

        qdt_g = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                       config='6s1/2', potential=self.model_pot),
        ]}

        qdt_g.update({
            'channels': [
                channel(qdt_g ['cores'][0], (1 / 2, 4, 9 / 2), tt='slj'),
                channel(qdt_g ['cores'][0], (1 / 2, 4, 7 / 2), tt='slj'),
            ]})

        # The G channels are closer to being jj coupled thatn they are to being LS coupled
        Uiabar13G4 = np.identity(2)
        
        MQDT_13G4 = mqdt_class(channels=qdt_g['channels'],
                               eig_defects=[[self.p.value('13g4_mu0',0.0262659964), self.p.value('13g4_mu0_1',-0.148808463)], [self.p.value('13g4_mu1',0.0254568575), self.p.value('13g4_mu1_1',-0.134219071)]],
                               rot_order=[[1,2]],
                               rot_angles=[[self.p.value('13g4_th12',-0.089123698)],],
                               Uiabar=Uiabar13G4, nulims=[[1],[0]],atom=self)

        self.mqdt_models.append({'L': 4, 'F': 4, 'model': MQDT_13G4})
        self.channels.extend(qdt_g['channels'])


        # 3G3 Rydberg Ritz formula
        QDT_3G3 = mqdt_class_rydberg_ritz(channels=qdt_g['channels'][1],
                                          deltas=[self.p.value('3g3_rr_0', 0.0260964574),self.p.value('3g3_rr_2',-0.14139526)], atom=self)

        self.mqdt_models.append({'L': 4, 'F': 3, 'model': QDT_3G3})

        # 3G5 Rydberg Ritz formula
        #NOTE: these values are obtained from 171Yb 
        QDT_3G5 = mqdt_class_rydberg_ritz(channels=qdt_g['channels'][0],
                                          deltas=[self.p.value('3g5_rr_0', 0.02536571),self.p.value('3g5_rr_2', -0.18507079)], atom=self)

        self.mqdt_models.append({'L': 4, 'F': 5, 'model': QDT_3G5})

        # QDT H channels

        qdt_h = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                       config='6s1/2', potential=self.model_pot),
        ]}

        qdt_h.update({
            'channels': [
                channel(qdt_h ['cores'][0], (1 / 2, 5, 11 / 2), tt='slj'),
                channel(qdt_h ['cores'][0], (1 / 2, 5, 9 / 2), tt='slj'),
            ]})
        
        # The G channels are closer to being jj coupled thatn they are to being LS coupled, lets treat the H channels as jj coupled

        Uiabar13H5 = np.identity(2)
        # NOTE: get a better estimate for the +,-H5 quantum defect splitting. 13h5_mu1 currently set to 0.009205 to avoid issues with near-degenerate states in MQDT solver ...
        MQDT_13H5 = mqdt_class(channels=qdt_h['channels'],
                               eig_defects=[[self.p.value('13h5_mu0',0.009305), self.p.value('13h5_mu0_1',-0.073)], [self.p.value('13h5_mu1',0.009205), self.p.value('13h5_mu1_1',-0.073)]],
                               rot_order=[[1,2]],
                               rot_angles=[[self.p.value('13h5_th12',1e-3)],],# the code has issues with fully uncoupled channels. We introduce a small angle (essentially 0) angle to avoid this issue. to be fixed ...
                               Uiabar=Uiabar13H5, nulims=[[1],[0]],atom=self)

        self.mqdt_models.append({'L': 5, 'F': 5, 'model': MQDT_13H5})
        self.channels.extend(qdt_h['channels'])


        # 3H4 Rydberg Ritz formula
        # NOTE: get a better estimate for the spin-orbit splitting in H states. Currently set to 1H5 value 
        QDT_3H4 = mqdt_class_rydberg_ritz(channels=qdt_h['channels'][1],
                                          deltas=[self.p.value('3h4_rr_0', 0.009305),self.p.value('3h4_rr_1', -0.073)], atom=self)

        self.mqdt_models.append({'L': 5, 'F': 4, 'model': QDT_3H4})

        # 3H6 Rydberg Ritz formula
        # NOTE: get a better estimate for the spin-orbit splitting in H states. Currently set to 1H5 value 
        QDT_3H6 = mqdt_class_rydberg_ritz(channels=qdt_h['channels'][0],
                                          deltas=[self.p.value('3h6_rr_0', 0.009305),self.p.value('3h6_rr_1', -0.073)], atom=self)

        self.mqdt_models.append({'L': 5, 'F': 6, 'model': QDT_3H6})

        # QDT I channels

        qdt_i = {'cores': [
            core_state((1 / 2, 0, 1 / 2, 0, 1 / 2), Ei_Hz=0, tt='sljif',
                       config='6s1/2', potential=self.model_pot),
        ]}

        qdt_i.update({
            'channels': [
                channel(qdt_i ['cores'][0], (1 / 2, 6, 13 / 2), tt='slj'),
                channel(qdt_i ['cores'][0], (1 / 2, 6, 11 / 2), tt='slj'),
            ]})


        # The G channels are closer to being jj coupled thatn they are to being LS coupled, lets treat the I channels as jj coupled

        Uiabar13I6 = np.identity(2)
        # NOTE: get a better estimate for the +,-I6 quantum defect splitting. 13i6_mu1 currently set to 0.004052 to avoid issues with near-degenerate states in MQDT solver ...

        MQDT_13I6 = mqdt_class(channels=qdt_i['channels'],
                               eig_defects=[[self.p.value('13i6_mu0',0.004062), self.p.value('13i6_mu0_1',-0.128)], [self.p.value('13i6_mu1',0.004052), self.p.value('13i6_mu1_1',-0.128)]],
                               rot_order=[[1,2]],
                               rot_angles=[[self.p.value('13i6_th12',1e-3)],],# the code has issues with fully uncoupled channels. We introduce a small angle (essentially 0) angle to avoid this issue. to be fixed...
                               Uiabar=Uiabar13I6, nulims=[[1],[0]],atom=self)

        self.mqdt_models.append({'L': 6, 'F': 6, 'model': MQDT_13I6})
        self.channels.extend(qdt_i['channels'])
                

        # 3I5 Rydberg Ritz formula, this is a guess
        # NOTE: get a better estimate for the spin-orbit splitting in I states. Currently set to 1I6 value 
        QDT_3I5 = mqdt_class_rydberg_ritz(channels=qdt_i['channels'][1],
                                          deltas=[self.p.value('3i5_rr_0', 0.004062), self.p.value('3i5_rr_1', -0.128)], atom=self)

        self.mqdt_models.append({'L': 6, 'F': 5, 'model': QDT_3I5})

        # 3I7 Rydberg Ritz formula, this is a guess
        # NOTE: get a better estimate for the spin-orbit splitting in I states. Currently set to 1I6 value 
        QDT_3I7 = mqdt_class_rydberg_ritz(channels=qdt_i['channels'][0],
                                          deltas=[self.p.value('3i7_rr_0', 0.004062), self.p.value('3i7_rr_1', -0.128)], atom=self)

        self.mqdt_models.append({'L': 6, 'F': 7, 'model': QDT_3I7})      

        #NOTE higher L states are generated on the fly using the create_high_l_MQDT method.


        super().__init__(**kwargs)


    def get_state(self, qn, tt='vlfm', energy_exp_Hz=None, energy_only=False,whittaker_wfct=False):
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

            whittaker_wf : bool, optional
            If True, computes the wavefunction using the generalized Coulomb Whittaker function (see self.whittaker_wfct).
            If False (default), computes the wavefunction numerically using the Numerov method.
            
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
            # If quantum numbers are valid, create a new MQDT model, else return None
            if np.abs(f-l)<=1 and isinstance(f, int) and isinstance(l, int):
                #print("new model is created")
                self.create_high_l_MQDT(qn)
                solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
            else:
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
        st.whittaker_wfct = whittaker_wfct

        return st

    def get_state_nist(self, qn, tt='nsljm',whittaker_wfct=False):

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
            st.whittaker_wfct = whittaker_wfct

            return st

        else:
            print("tt=", tt, " not supported by H.get_state")


    def create_high_l_MQDT(self,qn):
        """
        Creates a high-angular-momentum (high-l) MQDT (Multichannel Quantum Defect Theory) model for I=0 alkaline atoms.

        This function is called when a requested state has quantum numbers (l, f) not already present in the MQDT model list,
        typically for l > 4. It constructs the necessary core states and channels, and calculates quantum defect parameters
        including dipole and quadrupole polarizability, relativistic corrections, and spin-orbit coupling effects.
        The model assumes high-l states are jj-coupled.

        Parameters:
            qn (list): Quantum numbers for the state, typically [n, l, f, m], where
                n (float): Principal quantum number or effective quantum number.
                l (int): Orbital angular momentum quantum number.
                f (int): Total angular momentum quantum number.
                m (int): Magnetic quantum number.

        Behavior:
            - Defines core states and channels for the requested high-l value.
            - Calculates quantum defect parameters using atomic constants and polarizabilities.
            - Handles regular and indirect spin-orbit coupling corrections.
            - Adds the constructed MQDT model to self.mqdt_models and extends self.channels.
            - Supports cases where f = l, f = l-1, and f = l+1.

        Returns:
            None. The new MQDT model is added to the object's model list for future use.

        Notes:
            - This function is automatically called by get_state if a high-l model is needed.
            - The implementation is based on analytic formulas for quantum defects and coupling, and is suitable for alkaline-earth atoms like Yb.
            - NOTE: still under development
        """

        n = qn[0]
        v = qn[0]
        l = qn[1]
        f = qn[2]
        m = qn[3]

        # QDT high l channels
        self.p.set_prefix('174Yb_'+str(l))

        qdt = {'cores': [
            self.mqdt_models[0]['model'].channels[0].core,
        ]}

        qdt.update({
            'channels': [
                channel(qdt ['cores'][0], (1 / 2, l, l+1/2), tt='slj'),
                channel(qdt ['cores'][0], (1 / 2, l, l-1/2), tt='slj'),
            ]})


        # calculate high-l quantum defects, this includes dipole and quadrupole polarizability, relativistic corrections, and "direct" and "indirect" spin-orbit coupling
        # fine structure terms as summarized in Lundeen, ADVANCES IN ATOMIC, MOLECULAR AND OPTICAL PHYSICS, VOL. 52, 161 (2005) (https://www.sciencedirect.com/science/article/pii/S1049250X05520044)
        A4 = 1/(2*(l-1/2)*l*(l+1/2)*(l+1)*(l+3/2))
        A6 = A4 / (4*(l-3/2)*(l-1)*(l+2)*(l+5/2))

        A36 = 8*(l-3/2)*(l-1)*(l-1/2)*(l+3/2)*(l+2)*(l+5/2)
        A38 = 16*(l-5/2)*(l-2)*(l-3/2)*(l-1)*(l-1/2)*(l+3/2)*(l+2)*(l+5/2)*(l+3)*(l+7/2)

        # dipole and quadrupole coefficients in indirect spin-orbit coupling: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.68.022510
        bdprefac = (2/3)*(self.deltaEp_m/self.Ep_m**2)*qdt['cores'][0].alpha_d_a03/cs.alpha**2*cs.physical_constants['hartree-inverse meter relationship'][0]
        bqprefac = (6/5)*(self.deltaEd_m/self.Ed_m**2)*qdt['cores'][0].alpha_q_a05/cs.alpha**2*cs.physical_constants['hartree-inverse meter relationship'][0]

        # dipole and quadrupole polarizability and relativistic corrections to the quantum defect of high l states
        # the relativistic kinetic energy corection introduces odd (1/n) terms in the energy-dependence expansion of the quantum defect
        mu0 = (3/2)*A4*qdt['cores'][0].alpha_d_a03 + (35/2)*A6*qdt['cores'][0].alpha_q_a05 + cs.alpha**2/(2*(l+1/2))
        mu1 = -(3/8)*cs.alpha**2
        mu2 = (-1/2)*(l*(l+1)*A4*qdt['cores'][0].alpha_d_a03 + 5*(6*l**2+6*l-5)*A6*qdt['cores'][0].alpha_q_a05)
        mu3 = 0
        mu4 = (3/2)*(l-1)*l*(l+1)*(l+2)*A6*qdt['cores'][0].alpha_q_a05
        mu5 = 0
        mu6 = 0

        if l>4:
            if l==f:
                
                fsprefac3 = -(1/2)*cs.alpha**2/((l+1/2)*l*(l+1))
                fsprefac13 = (1/2)*cs.alpha**2/((l+1/2)*np.sqrt(l*(l+1)))

                qd_regularSO_3 = (1/2)*fsprefac3 + fsprefac3
                qd_indirectSO_0_3 = -fsprefac3*(1-bdprefac*35/A36-bqprefac*231/A38) 
                qd_indirectSO_2_3 = -fsprefac3*(-bdprefac*(25-30*l*(l+1))/A36-bqprefac*(735-315*l*(l+1))/A38)
                qd_indirectSO_4_3 = -fsprefac3*(-bdprefac*(3*(l-1)*l*(l+1)*(l+2))/A36-bqprefac*(21*(14 + 5*l*(l+1)*(l**2+l-5)))/A38)
                qd_indirectSO_6_3 = - fsprefac3*(-bqprefac*(-(5*(l-2)*(l-1)*(l)*(l+1)*(l+2)*(l+3)))/A38)

                qd_regularSO_13 = -(1/2)*fsprefac13 
                qd_indirectSO_0_13 = -fsprefac13*(1-bdprefac*35/A36-bqprefac*231/A38) 
                qd_indirectSO_2_13 = -fsprefac13*(-bdprefac*((25-30*l*(l+1)))/A36-bqprefac*(735-315*l*(l+1))/A38)
                qd_indirectSO_4_13 = -fsprefac13*(-bdprefac*(3*(l-1)*l*(l+1)*(l+2))/A36-bqprefac*(21*(14 + 5*l*(l+1)*(l**2+l-5)))/A38)
                qd_indirectSO_6_13 = fsprefac13*(-bqprefac*(-(5*(l-2)*(l-1)*(l)*(l+1)*(l+2)*(l+3)))/A38)

                a1 = qd_regularSO_3 + qd_indirectSO_0_3
                a2 = qd_indirectSO_2_3
                a3 = qd_indirectSO_4_3

                b1 = qd_regularSO_13 + qd_indirectSO_0_13
                b2 = qd_indirectSO_2_13
                b3 = qd_indirectSO_4_13

                pm_0 = (1/2)*np.sqrt(a1**2+4*b1**2)
                pm_2 = (1/4)*(a1*a2 + 4*b1*b2) / pm_0
                pm_4 = (a1**3*a3 + 4*a1*b1*(a3*b1 - a2*b2) + 2*a1**2*(b2**2 + 2*b1*b3) + 2*b1**2*(a2**2 + 8*b1*b3)) / (2 * (a1**2 + 4* b1**2)**(3/2))
                pm_6= (a1*b2-a2*b1)*(4*b1**2*(a2*b2-2*a3*b1)-a1**2*(2*a3*b1 + a2*b2)+2*a1**3*b3+a1*b1*(a2**2 - 4*b2**2 + 8*b1*b3)) / (2 * (a1**2 + 4* b1**2)**(5/2))


                # The G channels are closer to being jj coupled thatn they are to being LS coupled, lets treat the high l channels as jj coupled
                
                QDT_pLL = mqdt_class_rydberg_ritz(channels=qdt['channels'][0],
                                                deltas=[self.p.value('pLL_mu0',mu0-(1/2)*qd_regularSO_3-(1/2)*qd_indirectSO_0_3-pm_0), self.p.value('pLL_mu0_1',mu1), self.p.value('pLL_mu0_2',mu2-(1/2)*qd_indirectSO_2_3-pm_2), self.p.value('pLL_mu0_3',mu3), self.p.value('pLL_mu0_4',mu4-(1/2)*qd_indirectSO_4_3-pm_4), self.p.value('pLL_mu0_5',mu5), self.p.value('pLL_mu1_6',mu6-pm_6)], atom=self,odd_powers=True)
                
                
                QDT_mLL = mqdt_class_rydberg_ritz(channels=qdt['channels'][1],
                                                deltas=[self.p.value('mLL_mu1',mu0-(1/2)*qd_regularSO_3-(1/2)*qd_indirectSO_0_3+pm_0), self.p.value('mLL_mu1_1',mu1), self.p.value('mLL_mu1_2',mu2-(1/2)*qd_indirectSO_2_3+pm_2), self.p.value('mLL_mu1_3',mu3), self.p.value('mLL_mu1_4',mu4-(1/2)*qd_indirectSO_4_3+pm_4), self.p.value('mLL_mu1_5',mu5), self.p.value('mLL_mu1_6',mu6+pm_6)], atom=self,odd_powers=True)

                
                MQDT_pmLL = mqdt_class_wrapper([QDT_pLL,QDT_mLL])

                self.mqdt_models.append({'L': l, 'F': f, 'model': MQDT_pmLL})
                self.channels.extend(qdt['channels'])
                    
            elif f==l-1:

                fsprefac = (1/2)*cs.alpha**2/((l+1/2)*l)

                qd_regularSO = (1/2)*fsprefac - fsprefac/(2*l-1)
                qd_indirectSO_0 = -fsprefac*(1-bdprefac*35/A36-bqprefac*231/A38) 
                qd_indirectSO_2 = -fsprefac*(-bdprefac*((25-30*l*(l+1)))/A36-bqprefac*(735-315*l*(l+1))/A38)
                qd_indirectSO_4 = -fsprefac*(-bdprefac*(3*(l-1)*l*(l+1)*(l+2))/A36-bqprefac*(21*(14 + 5*l*(l+1)*(l**2+l-5)))/A38)
                qd_indirectSO_6 = -fsprefac*(-bqprefac*(-(5*(l-2)*(l-1)*(l)*(l+1)*(l+2)*(l+3)))/A38)


                d0 = mu0 + qd_regularSO + qd_indirectSO_0
                d1 = mu1 - (3/2)*d0**2
                d2 = mu2 + qd_indirectSO_2 - 3*d0*d1
                d3 = mu3 - (3/2)*d1**2 - 3*d0*d2
                d4 = mu4 + qd_indirectSO_4 - 3*d0*d3 - 3*d1*d2


                QDT_3LLm1 = mqdt_class_rydberg_ritz(channels=qdt['channels'][1],
                                                deltas=[self.p.value('3LLm1_rr_0',mu0+qd_regularSO+qd_indirectSO_0),self.p.value('3LLm1_rr_1',mu1),self.p.value('3LLm1_rr_2',mu2+qd_indirectSO_2),self.p.value('3LLm1_rr_3',mu3),self.p.value('3LLm1_rr_4',mu4+qd_indirectSO_4),self.p.value('3LLm1_rr_5',mu5),self.p.value('3LLm1_rr_6',qd_indirectSO_6)], atom=self,odd_powers=True)
                
                self.mqdt_models.append({'L': l, 'F': f, 'model': QDT_3LLm1})
                self.channels.extend(qdt['channels'])

            else:

                fsprefac = (1/2)*cs.alpha**2/((l+1/2)*(l+1))

                qd_regularSO = -(1/2)*fsprefac - fsprefac/(2*l+3)
                qd_indirectSO_0 = fsprefac*(1-bdprefac*35/A36-bqprefac*231/A38) 
                qd_indirectSO_2 = fsprefac*(-bdprefac*((25-30*l*(l+1)))/A36-bqprefac*(735-315*l*(l+1))/A38)
                qd_indirectSO_4 = fsprefac*(-bdprefac*(3*(l-1)*l*(l+1)*(l+2))/A36-bqprefac*(21*(14 + 5*l*(l+1)*(l**2+l-5)))/A38)
                qd_indirectSO_6 = fsprefac*(-bqprefac*(-(5*(l-2)*(l-1)*(l)*(l+1)*(l+2)*(l+3)))/A38)

                d0 = mu0 + qd_regularSO + qd_indirectSO_0
                d1 = mu1 - (3/2)*d0**2
                d2 = mu2 + qd_indirectSO_2 - 3*d0*d1
                d3 = mu3 - (3/2)*d1**2 - 3*d0*d2
                d4 = mu4 + qd_indirectSO_4 - 3*d0*d3 - 3*d1*d2

                QDT_3LLp1 = mqdt_class_rydberg_ritz(channels=qdt['channels'][0],
                                                deltas=[self.p.value('3LLp1_rr_0',mu0+qd_regularSO+qd_indirectSO_0),self.p.value('3LLp1_rr_1',mu1),self.p.value('3LLp1_rr_2',mu2+qd_indirectSO_2),self.p.value('3LLp1_rr_3',mu3),self.p.value('3LLp1_rr_4',mu4+qd_indirectSO_4),self.p.value('3LLp1_rr_5',mu5),self.p.value('3LLp1_rr_6',qd_indirectSO_6)], atom=self,odd_powers=True)

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
                    if np.abs(f-l)<=1 and l>4:# and isinstance(f, int) and isinstance(l, int):
                        #print("new model is created")
                        self.create_high_l_MQDT([st.nub,l,f,st.m])
                        solver = [d for d in self.mqdt_models if d['L'] == l and d['F'] == f][0]['model']
                    else:
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
                            st_new.whittaker_wfct = whittaker_wfct
                            st.v_exact = nub

                            ret.append(st_new)

        return ret
