from random import gauss
from pylab import *
from scipy import *
from numpy import *

'''
Created on Jan 7, 2011

M-current enabled variant on the Izhikevich quadratic integrate-and-fire neuron

@author: stefan
'''


class Neuron:
    def __init__(
            self,
            tau,
            parameters,
            name,
            inputs=None,
            outputs=None,
            diffuseTransmitters=None,
            externalInput=0.0,
            parentPopulation=None,
            debug=False):

        self.name = name
        self.originalParameters = parameters

        self.a = self.originalParameters["a"]
        self.b = self.originalParameters["b"]
        self.c = self.originalParameters["c"]
        self.d = self.originalParameters["d"]
        self.v = self.originalParameters["v_r"] + gauss(0, 20)
        self.v_r = self.originalParameters["v_r"]
        self.v_t = self.originalParameters["v_t"]
        self.k = self.originalParameters["k"]
        self.C = self.originalParameters["C"]
        self.v_peak = self.originalParameters["v_peak"]

        # # Synapse Parameters
        # self.K_GABA = 0
        # self.Q_GABA = 0
        # self.g_GABA = 0
        # self.tau_r_GABA = 8 # ms
        # self.tau_f_GABA = 30 # ms

        # self.K_GLUT = 0
        # self.Q_AMPA = 0
        # self.Q_NMDA = 0
        # self.g_AMPA = 0
        # self.g_NMDA = 0
        # self.tau_r_AMPA = 2 # ms
        # self.tau_f_AMPA = 10 # ms
        # self.tau_r_NMDA = 8 # ms
        # self.tau_f_NMDA = 100 # ms

        self.tau = tau

        if inputs is None:
            self.inputs = []
        else:
            self.inputs = inputs

        if outputs is None:
            self.outputs = []
        else:
            self.outputs = outputs

        self.parentPopulation = parentPopulation

        self.time = 0
        self.debug = debug
        self.type = parameters["type"]

        # Input
        self.externalInput = externalInput
        self.synapticInput = 0
        self.postSynapticReceptors = []
        self.diffuseReceptors = []
        if diffuseTransmitters is None:
            self.diffuseTransmitters = {}
        else:
            self.diffuseTransmitters = diffuseTransmitters
        self.diffuseCurrent = 0
        self.I = 0

        # Init Variables and membrane equations
        self.u = gauss(self.d, self.d / 6)  # self.b*self.v
        self.dvdt = lambda v, u: (0.04 * v**2 + (5 * v) + 140 - u + self.I)
        # # self.dvdt = lambda v, u: (self.k * (v - self.v_r) * (v - self.v_t) - u + self.I) / self.C
        self.dudt = lambda v, u: (self.a * (self.b * (v - self.v_r) - u))

        # self.dQ_AMPA = lambda vlist: ((1-vlist[0])*self.K - (vlist[0]/self.tau_r_AMPA))/self.tau
        # self.dg_AMPA = lambda vlist: ((self.tau_f_AMPA + self.tau_r_AMPA)/self.tau_fAMPA)*((2/self.tau_r_AMPA)*(1-vlist[0])*vlist[1]-(vlist[0]/self.tau_f))/self.tau

        # self.dQ_NMDA = lambda vlist: ((1 - vlist[0]) * self.K - (vlist[0] / self.tau_r_NMDA)) / self.tau
        # self.dg_NMDA = lambda vlist: ((self.tau_f_NMDA + self.tau_r_NMDA) / self.tau_f_NMDA) * ((2 / self.tau_r_NMDA) * (1 - vlist[0]) * vlist[1] - (vlist[0] / self.tau_f_NMDA)) / self.tau

        # self.dQ_GABA = lambda q_GABA: ((1-q_GABA)*self.K - (q_GABA/self.tau_r))/self.tau
        # self.dg_GABA = lambda g_GABA, q: ((self.tau_f + self.tau_r)/self.tau_f)*((2/self.tau_r)*(1-self.g_SD)*self.Q-(self.g_SD/self.tau_f))/self.tau

        # Recording variables
        self.vv = []
        self.uu = []
        self.bb = []
        self.ii = []
        self.spikeRecord = []

    def registerPostSynapticReceptor(self, receptor):
        self.postSynapticReceptors.append(receptor)

    def addDiffuseReceptor(self, receptor):
        self.diffuseReceptors.append(receptor)
        receptor.setTarget(self)

    def updateDiffuseTransmitters(self, diffuseTransmitters):
        self.diffuseTransmitters = diffuseTransmitters
        # Update somatic diffuse receptor levels
        for somaticReceptor in self.diffuseReceptors:
            try:
                # Set the level to be whatever matches it in the passed
                # dictionary
                somaticReceptor.setLevel(
                    self.diffuseTransmitters[somaticReceptor.getTypeString()])
            except KeyError:
                print(
                    "WARNING: There are somatic receptors of type " +
                    somaticReceptor.getTypeString() +
                    " on neuron " +
                    self.name +
                    " in population " +
                    self.parentPopulation.name +
                    ", which does not currently have any transmitter levels set for that type.  Defaulting to a level of 0.0")
                somaticReceptor.setLevel(0.0)

        # Reset Diffuse Current and Neural Parameters
        self.diffuseCurrent = 0
        self.a = self.originalParameters["a"]
        self.b = self.originalParameters["b"]
        self.c = self.originalParameters["c"]
        self.d = self.originalParameters["d"]
        self.v_r = self.originalParameters["v_r"]
        self.v_t = self.originalParameters["v_t"]
        self.k = self.originalParameters["k"]
        self.C = self.originalParameters["C"]
        self.v_peak = self.originalParameters["v_peak"]

        # Run re-run all diffuse receptors
        for somaticReceptor in self.diffuseReceptors:
            somaticReceptor.doActivity()

    def setInputs(self, inputs):
        self.inputs = inputs

    def addInput(self, inputSynapse):
        self.inputs.append(inputSynapse)

    def setInjectedCurrent(self, current):
        if isinstance(current, (int, float)):
            self.externalInput = current
        else:
            raise ValueError(
                "Error: setInjectedCurrent requires an int or float type, representing current in picoamperes")

    def getInputs(self):
        return self.inputs

    # def setOutputs(self, outputs):
    #     self.outputs = outputs

    def addOutput(self, outputAxon):
        self.outputs.append(outputAxon)

    def getOutputs(self):
        return self.outputs

    def addSynapticTransmission(self, current):
        if self.debug:
            print("INCOMING SPIKE!")
        self.synapticInput += current

    def rk4OneStep(self, dvdt, dudt, v, u, h):
        k1 = h * (dvdt(v, u))
        l1 = h * (dudt(v, u))
        k2 = h * (dvdt(v + 0.5 * k1, u + 0.5 * l1))
        l2 = h * (dudt(v + 0.5 * k1, u + 0.5 * l1))
        k3 = h * (dvdt(v + 0.5 * k2, u + 0.5 * l2))
        l3 = h * (dudt(v + 0.5 * k2, u + 0.5 * l2))

        k4 = h * dvdt(v + k3, u + l3)
        l4 = h * dudt(v + k3, u + l3)

        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        l = (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)

        return ([k, l])

    def step(self):
        self.time += self.tau
        # self.synapticInput = self.g_AMPA + self.g_NMDA - self.g_GABA
        self.I = self.externalInput + self.synapticInput + self.diffuseCurrent
        # print(self.name, ":", self.synapticInput, "/", self.I)
        self.ii.append(self.I)

        # # Euler
        # # self.v = self.v + self.tau*(self.k*(self.v-self.v_r)*(self.v-self.v_t)-self.u+self.I)/self.C
        # self.v = self.v + self.tau*((0.04*self.v**2 + (5*self.v) + 140 - self.u + self.I) / self.C)
        # self.u = self.u + self.tau*(self.a*(self.b*(self.v-self.v_r)-self.u))

        # self.Q_AMPA = self.Q_AMPA + self.tau*((1-self.Q_AMPA)*self.K_GLUT - (self.Q_AMPA/self.tau_r_AMPA))
        # self.g_AMPA = self.g_AMPA + self.tau*((self.tau_f_AMPA + self.tau_r_AMPA)/self.tau_f_AMPA)*((2/self.tau_r_AMPA)*(1-self.g_AMPA)*self.Q_AMPA-(self.g_AMPA/self.tau_f_AMPA))

        # self.Q_NMDA = self.Q_NMDA + self.tau*((1-self.Q_NMDA)*self.K_GLUT - (self.Q_NMDA/self.tau_r_NMDA))
        # self.g_NMDA = self.g_NMDA + self.tau*((self.tau_f_NMDA + self.tau_r_NMDA)/self.tau_f_NMDA)*((2/self.tau_r_NMDA)*(1-self.g_NMDA)*self.Q_NMDA-(self.g_NMDA/self.tau_f_NMDA))

        # self.Q_GABA = self.Q_GABA + self.tau*((1-self.Q_GABA)*self.K_GABA - (self.Q_GABA/self.tau_r_GABA))
        # self.g_GABA = self.g_GABA + self.tau*((self.tau_f_GABA + self.tau_r_GABA)/self.tau_f_GABA)*((2/self.tau_r_GABA)*(1-self.g_GABA)*self.Q_GABA-(self.g_GABA/self.tau_f_GABA))

        # # Uncomment this line if you want to simulate M-Currents
        # # self.b = self.b + ((0.25-self.b) + (0.02 - self.b)*(max((self.V + 67),0)))/100

        # Runge-Kutta
        [dv, du] = self.rk4OneStep(
            self.dvdt, self.dudt, self.v, self.u, self.tau)
        self.v = self.v + dv
        self.u = self.u + du

        self.vv.append(self.v)
        self.bb.append(self.b)
        self.uu.append(self.u)

        if self.v > self.v_peak:  # Spike
            self.spikeRecord.append(self.time)
            for receptor in self.postSynapticReceptors:
                receptor.postSynapticSpikeFeedback()
            self.v = self.c
            self.u = self.u + self.d
            if self.debug:
                print("SPIKE!")
            for axon in self.outputs:
                axon.enqueue()
        if self.u > 500:
            self.u = 500

        self.synapticInput = 0

        # self.diffuseCurrent = 0
        # self.K_GLUT = 0
        # self.K_GABA = 0

    def getState(self):
        vars = {}
        vars["V"] = self.v
        vars["u"] = self.u
        vars["b"] = self.b
        return vars
