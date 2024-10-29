from .alkali import *

class Hydrogen_n(AlkaliAtom):
    """ Hydrogen, but using numerical techniques from alkali atoms """
    
    name = 'Hydrogen_n'
    mass = 1
    
    Z = 1
    
    ground_state_n = 1
    
    def __init__(self,**kwargs):
        
        self.model_pot = model_potential(0, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [1e-3] * 4,
                                    self.Z, include_so=True, use_model=False)
    
        self.core = core_state((0,0,0,0,0),0,tt='sljif',config='p',potential = self.model_pot)
        
        self.channels = []
        self.defects = []
        self.defects.append(defect_model(0,polcorrection=False,relcorrection=False,SOcorrection=False))
        
        super().__init__(**kwargs)

class Rubidium87(AlkaliAtom):
    """
        Properites of rubidium 87 atoms
    """

    name = 'Rb'
    dipole_data_file = 'rb_dipole_matrix_elements.npy'
    dipole_db_file = 'rb_dipole.db'
    
    # ALL PARAMETERES ARE IN ATOMIC UNITS (HARTREE)
    mass = 86.9091805310
    Z = 37

    ground_state_n = 5

    citations = ['Mack2011Measurement','Li2003Millimeter','Han2006Rb','Berl2020Core','Marinescu1994Dispersion']
    
    defects = []
    #defects.append(defect_Rydberg_Ritz([3.1311804, 0.1784], condition = lambda qns: qns['j']==1/2 and qns['l']==0)) , #85Rb from PHYSICAL REVIEW A 83, 052515 (2011)
    defects.append(defect_Rydberg_Ritz([3.1311807, 0.1787], condition=lambda qns: qns['j'] == 1/2 and qns['l'] == 0)), #87Rb from PHYSICAL REVIEW A 83, 052515 (2011)
    defects.append(defect_Rydberg_Ritz([2.6548849, 0.2900], condition = lambda qns: qns['j']==1/2 and qns['l']==1)) # from PHYSICAL REVIEW A 67, 052502 (2003)
    defects.append(defect_Rydberg_Ritz([2.6416737, 0.2950], condition = lambda qns: qns['j']==3/2 and qns['l']==1)) # from PHYSICAL REVIEW A 67, 052502 (2003)
    defects.append(defect_Rydberg_Ritz([1.34809171, -0.60286], condition = lambda qns: qns['j']==3/2 and qns['l']==2)) # from PHYSICAL REVIEW A 67, 052502 (2003)
    defects.append(defect_Rydberg_Ritz([1.34646572, -0.59600], condition = lambda qns: qns['j']==5/2 and qns['l']==2)) # from PHYSICAL REVIEW A 67, 052502 (2003)
    defects.append(defect_Rydberg_Ritz([0.0165192, -0.085], condition=lambda qns: qns['j'] == 5 / 2 and qns['l'] == 3)) # from 85Rb PHYSICAL REVIEW A 74, 054502 (2006)
    defects.append(defect_Rydberg_Ritz([0.0165437, -0.086], condition=lambda qns: qns['j'] == 7 / 2 and qns['l'] == 3)) # from 85Rb PHYSICAL REVIEW A 74, 054502 (2006)
    defects.append(defect_Rydberg_Ritz([0.004007, -0.02742], condition=lambda qns: qns['l'] == 4,SOcorrection=True)) # PHYSICAL REVIEW A 102, 062818 (2020)
    defects.append(defect_Rydberg_Ritz([0.001423, -0.01438], condition=lambda qns: qns['l'] == 5,SOcorrection=True)) # PHYSICAL REVIEW A 102, 062818 (2020)
    defects.append(defect_Rydberg_Ritz([0.0006074, -0.008550], condition=lambda qns: qns['l'] == 6,SOcorrection=True)) # PHYSICAL REVIEW A 102, 062818 (2020)
    defects.append(defect_model(0,polcorrection=True,relcorrection=True,SOcorrection=True)) # assume ''hydrogenic'' behaviour for remaining states
        
    def __init__(self,**kwargs):
        
        model_pot = model_potential(9.0760,[3.69628474, 4.44088978, 3.78717363, 2.39848933],
                                    [1.64915255, 1.92828831, 1.57027864, 1.76810544],
                                    [-9.86069196, -16.79597770, -11.65588970, -12.07106780],
                                    [0.19579987, -0.8163314, 0.52942835, 0.77256589],
                                    [1.66242117, 1.50195124, 4.86851938, 4.79831327],
                                    self.Z,include_so = True)#Phys. Rev. A 49, 982 (1994)
    
        self.core = core_state((0,0,0,0,0),0,tt='sljif',config='Kr+',potential = model_pot,alpha_d_a03 = 9.116, alpha_q_a05 = 38.4)#  PHYSICAL REVIEW A 102, 062818 (2020)
    
        I = 1.5  # 3/2

        self.channels = []
        
        super().__init__(**kwargs)


class Potassium39(AlkaliAtom):
    """
        Properites of potassium 39 atoms
    """

    name = 'K'
    dipole_data_file = 'k_dipole_matrix_elements.npy'
    dipole_db_file = 'k_dipole.db'

    # ALL PARAMETERES ARE IN ATOMIC UNITS (HARTREE)
    mass = 38.9637064864
    Z = 19

    ground_state_n = 4

    citations = ['Peper2019Precision','Risberg1956A','Johansson1972An','Lorenzen1981Precise','Lorenzen1983Quantum','Marinescu1994Dispersion']

    defects = []
    defects.append(defect_Rydberg_Ritz([2.18020826, 0.134534, 0.0952, 0.0021], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 0)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_Rydberg_Ritz([1.71392626, 0.23114, 0.1948, 0.3683], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 1)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_Rydberg_Ritz([1.71087854,  0.23233,  0.1961, 0.3716], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 1)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_Rydberg_Ritz([0.27698453, -1.02691, -0.665, 10.9], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 2)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_Rydberg_Ritz([0.27715665, -1.02493, -0.640, 10.0], condition=lambda qns: qns['j'] == 5 / 2 and qns['l'] == 2)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_Rydberg_Ritz([0.0094576, -0.0446], condition=lambda qns:  qns['l'] == 3)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_Rydberg_Ritz([0.0024080, -0.0209], condition=lambda qns: qns['l'] == 4,SOcorrection=True)) # PHYSICAL REVIEW A 100, 012501 (2019)
    defects.append(defect_model(0,polcorrection=True,relcorrection=True,SOcorrection=True)) # assume ''hydrogenic'' behaviour for remaining states

    def __init__(self, **kwargs):
        model_pot = model_potential(5.3310, [3.56079437, 3.65670429, 4.12713694, 1.42310446],
                                    [1.83909642, 1.67520788, 1.79837462, 1.27861156],
                                    [-1.74701102, -2.07416615, -1.69935174, 4.77441476],
                                    [-1.03237313, -0.89030421, -0.98913582, -0.94829262],
                                    [0.83167545, 0.85235381, 0.83216907, 6.50294371],
                                    self.Z, include_so=True)#Phys. Rev. A 49, 982 (1994)

        self.core = core_state((0, 0, 0, 0, 0), 0, tt='sljif', config='Ar+', potential=model_pot,alpha_d_a03 = 5.4880, alpha_q_a05 = 17.89)# PHYSICAL REVIEW A 100, 012501 (2019)

        I = 1.5  # 3/2

        self.channels = []

        super().__init__(**kwargs)



class Cesium133(AlkaliAtom):
    """
        Properites of cesium 133 atoms
    """

    name = 'Cs'
    dipole_data_file = 'cs_dipole_matrix_elements.npy'
    dipole_db_file = 'cs_dipole.db'

    # ALL PARAMETERES ARE IN ATOMIC UNITS (HARTREE)
    mass = 132.9054519610
    Z = 55

    ground_state_n = 6

    defects = []
    defects.append(defect_Rydberg_Ritz([4.0493532, 0.239, 0.06, 11,-209], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 0)) # PHYSICAL REVIEW A 93, 013424 (2016)
    defects.append(defect_Rydberg_Ritz([3.5915871, 0.36273], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 1)) # PHYSICAL REVIEW A 93, 013424 (2016)
    defects.append(defect_Rydberg_Ritz([3.5590676,  0.37469], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 1)) # PHYSICAL REVIEW A 93, 013424 (2016)
    defects.append(defect_Rydberg_Ritz([2.475365, 0.5554], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 2)) # PHYSICAL REVIEW A 26, 2733 (1982)
    defects.append(defect_Rydberg_Ritz([2.4663144, 0.01381, -0.392, -1.9], condition=lambda qns: qns['j'] == 5 / 2 and qns['l'] == 2)) # PHYSICAL REVIEW A 93, 013424 (2016)
    defects.append(defect_Rydberg_Ritz([0.033392, -0.191], condition=lambda qns:  qns['j'] == 5 / 2 and qns['l'] == 3)) # PHYSICAL REVIEW A 26, 2733 (1982)
    defects.append(defect_Rydberg_Ritz([0.033537, -0.191], condition=lambda qns: qns['j'] == 7 / 2 and qns['l'] == 3)) # PHYSICAL REVIEW A 26, 2733 (1982)
    defects.append(defect_Rydberg_Ritz([0.00703865, -0.049252,0.0129], condition=lambda qns: qns['l'] == 4,SOcorrection=True)) # PHYSICAL REVIEW A 35, 4650 (1987)
    defects.append(defect_model(0,polcorrection=True,relcorrection=True,SOcorrection=True)) # assume ''hydrogenic'' behaviour for remaining states



    def __init__(self, **kwargs):
        model_pot = model_potential(15.6440, [3.495463, 4.69366096, 4.32466196, 3.01048361],
                                    [1.47533800, 1.71398344, 1.61365288, 1.40000001],
                                    [-9.72143084, -24.65624280, -6.70128850, -3.20036138],
                                    [0.02629242, -0.09543125, -0.74095193, 0.00034538],
                                    [1.92046930, 2.13383095, 0.93007296, 1.99969677],
                                    self.Z, include_so=True)#Phys. Rev. A 49, 982 (1994)

        self.core = core_state((0, 0, 0, 0, 0), 0, tt='sljif', config='Ar+', potential=model_pot,alpha_d_a03 = 15.5440, alpha_q_a05 = 70.7) # Phys. Rev. A 22, 2672 (1980)

        I = 3.5  # 7/2

        self.channels = []

        super().__init__(**kwargs)



class Lithium7(AlkaliAtom):
    """
        Properites of lithium 7 atoms
    """

    name = 'Li'
    dipole_data_file = 'li_dipole_matrix_elements.npy'
    dipole_db_file = 'li_dipole.db'

    # ALL PARAMETERES ARE IN ATOMIC UNITS (HARTREE)
    mass = 7.0160034366
    Z = 3

    ground_state_n = 2

    defects = []
    defects.append(defect_Rydberg_Ritz([0.3995101, 0.029], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 0)) # PHYSICAL REVIEW A 34, 2889 (1986)
    defects.append(defect_Rydberg_Ritz([0.0471835, -0.024], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 1)) # PHYSICAL REVIEW A 34, 2889 (1986)
    defects.append(defect_Rydberg_Ritz([0.0471720,  -0.024], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 1)) # PHYSICAL REVIEW A 34, 2889 (1986)
    defects.append(defect_Rydberg_Ritz([0.002129, -0.01491,0.1759,-0.8507], condition=lambda qns: qns['l'] == 2)) # Ark. f. Fysik 15,169 (1958), Phys. Scr. 27 300 (1983)
    defects.append(defect_Rydberg_Ritz([-0.000077, 0.021856, -0.4211, 2.3891], condition=lambda qns: qns['l'] == 3)) # Ark. f. Fysik 15,169 (1958), Phys. Scr. 27 300 (1983)
    defects.append(defect_model(0,polcorrection=True,relcorrection=True,SOcorrection=True)) # assume ''hydrogenic'' behaviour for remaining states



    def __init__(self, **kwargs):
        model_pot = model_potential(0.1923, [2.47718079, 3.45414648, 2.51909839, 2.51909839],
                                    [1.84150932, 2.55151080, 2.43712450, 2.43712450],
                                    [-0.02169712, -0.21646561, 0.32505524, 0.32505524],
                                    [-0.11988362, -0.06990078, 0.10602430, 0.10602430],
                                    [0.61340824, 0.61566441, 2.34126273, 2.34126273],
                                    self.Z, include_so=True)#Phys. Rev. A 49, 982 (1994)

        self.core = core_state((0, 0, 0, 0, 0), 0, tt='sljif', config='He+', potential=model_pot,alpha_d_a03 = 0.1884, alpha_q_a05 = 0.046) # Phys. Rev. A 16, 1141 (1977)

        I = 1.5  # 3/2

        self.channels = []

        super().__init__(**kwargs)

class Sodium23(AlkaliAtom):
    """
        Properites of sodium 23 atoms
    """

    name = 'Na'
    dipole_data_file = 'na_dipole_matrix_elements.npy'
    dipole_db_file = 'na_dipole.db'

    # ALL PARAMETERES ARE IN ATOMIC UNITS (HARTREE)
    mass = 22.9897692820
    Z = 11

    ground_state_n = 3

    defects = []
    defects.append(defect_Rydberg_Ritz([1.34796938, 0.0609892,0.0196743,-0.001045], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 0)) # PHYSICAL REVIEW A 45, 4720 (1992)
    defects.append(defect_Rydberg_Ritz([0.85544502, 0.112067,0.0479,0.0457], condition=lambda qns: qns['j'] == 1 / 2 and qns['l'] == 1)) # Quantum Electron. 25 914 (1995)
    defects.append(defect_Rydberg_Ritz([0.85462615,  0.112344,0.0497,0.0406], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 1)) # Quantum Electron. 25 914 (1995)
    defects.append(defect_Rydberg_Ritz([0.014909286, -0.042506,0.00840], condition=lambda qns: qns['j'] == 3 / 2 and qns['l'] == 2)) # Quantum Electron. 25 914 (1995)
    defects.append(defect_Rydberg_Ritz([0.01492422, -.042585,0.00840], condition=lambda qns: qns['j'] == 5 / 2 and qns['l'] == 2)) # Quantum Electron. 25 914 (1995)
    defects.append(defect_Rydberg_Ritz([0.001632977, -0.0069906, 0.00423], condition=lambda qns: qns['j'] == 5 / 2 and qns['l'] == 3)) # J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    defects.append(defect_Rydberg_Ritz([0.001630875, -0.0069824, 0.00352], condition=lambda qns: qns['j'] == 7 / 2 and qns['l'] == 3)) # J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    defects.append(defect_Rydberg_Ritz([0.00043825, -0.00283], condition=lambda qns: qns['j'] == 7 / 2 and qns['l'] == 4)) # J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    defects.append(defect_Rydberg_Ritz([0.00043740, -0.00297], condition=lambda qns: qns['j'] == 9 / 2 and qns['l'] == 4)) # J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    defects.append(defect_Rydberg_Ritz([0.00016114, -0.00185], condition=lambda qns: qns['j'] == 9 / 2 and qns['l'] == 5)) # J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    defects.append(defect_Rydberg_Ritz([0.00015796, -0.00148], condition=lambda qns: qns['j'] == 11 / 2 and qns['l'] == 5)) # J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    defects.append(defect_model(0,polcorrection=True,relcorrection=True,SOcorrection=True)) # assume ''hydrogenic'' behaviour for remaining states



    def __init__(self, **kwargs):
        model_pot = model_potential(0.9448, [4.82223117, 5.08382502, 3.53324124, 1.11056646],
                                    [2.45449865, 2.18226881, 2.48697936, 1.05458759],
                                    [-1.12255048, -1.19534623, -0.75688448, 1.73203428],
                                    [-1.42631393, -1.03142861, -1.27852357, -0.09265696],
                                    [0.45489422, 0.45798739, 0.71875312, 28.6735059],
                                    self.Z, include_so=True) #Phys. Rev. A 49, 982 (1994)

        self.core = core_state((0, 0, 0, 0, 0), 0, tt='sljif', config='Ne+', potential=model_pot,alpha_d_a03 = 0.9980, alpha_q_a05 = 0.351) # Phys. Rev. A 38, 4985 (1988)

        I = 1.5  # 3/2

        self.channels = []

        super().__init__(**kwargs)

