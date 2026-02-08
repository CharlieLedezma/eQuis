# import yaml
from numpy import linspace, arange

class Parameters:
    def __init__(self, yaml_file=None):
        self.N = 14  # Número de elos do robô
        self.m = 0.406  # Massa do elo do robô
        self.l = 0.0525  # Raio do elo do robô
        self.g = 9.81  # Aceleração gravitacional
        self.diameter = 0.3  # Diâmetro do tubo
        self.diameterInfluence = 0.10
        self.pipeLength = 3
        self.shotsNumber = 50
        self.dt = 0.01

        self.ct = 0.015
        self.cn = 0.03
        self.ut = 0.15
        self.un = 0.3

        self.ctPipe = 0.08
        self.utPipe = 0.2
        self.umax = 3
        self.Erub = 400000
        self.vrub = 0.49
        self.friction = 1
        self.contact = 1
        self.minLinkVel = 0.001
        self.dimensionPlot3D = 0
        self.resultsShow = 0
        self.alphaA = 0.3981
        self.omega = 0.6936
        self.delta = 0.4914
        self.offset = 0

        self.kp  = 25
        self.kd  = 10
        self.tmax = 2
        self.ti = 0
        self.tf = 20
        # self.npt = 200
        self.t = arange(self.ti, self.tf, self.dt)

        # if yaml_file:
        #     self.load_from_yaml(yaml_file)

        self.d = self.diameter - 2*self.l
        self.qmax = 400 * self.dt
        self.I = (self.m*(2*self.l)**2)/3


#param = Parameters()
    # def load_from_yaml(self, yaml_file):
    #     """Carrega os parâmetros a partir de um arquivo YAML."""
    #     with open(yaml_file, 'r') as file:
    #         data = yaml.safe_load(file)
    #         for key, value in data.items():
    #             if hasattr(self, key):
    #                 setattr(self, key, value)
