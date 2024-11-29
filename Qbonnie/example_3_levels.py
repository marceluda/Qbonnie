#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Solución del sistema de 3 niveles
"""


import numpy as np
import matplotlib.pyplot as plt

# from sympy import MatrixSymbol, Matrix, symbols, linear_eq_to_matrix, simplify
# from sympy.parsing.sympy_parser import parse_expr
# import sympy  as sy

from scipy import constants
import Qbonnie
from Qbonnie import qbonnie, tools, Matrix_sym

from sympy.parsing.sympy_parser import parse_expr

###############################################################################
# Defino constantes 


a0   = constants.physical_constants['Bohr radius'][0]              # m      Radio de Bohr
hbar = constants.hbar                                              # J s    h/2 pi
Kmks = constants.Boltzmann                                         # J/K     Constante de Boltzman
cef  = constants.physical_constants['fine-structure constant'][0]  # Constante de estructura fina
c    = constants.physical_constants['speed of light in vacuum'][0] # m/s  Velocidad de la luz




###############################################################################
# Definimos parámetros del sistema

q = Qbonnie.qbonnie(3)

q.define_environment(globals())
q.new_parameters('Omega_A Omega_B Gamma', real=True , positive=True)
q.new_parameters('Delta delta', real=True)

rho = q.create_rho()


H  = q.make_empty_matrix(name='H')
S  = q.make_empty_matrix(name='S')
L  = q.make_empty_matrix(name='L')



H += q.s(0,0)*Delta - q.s(1,1)*Delta -2*q.s(2,2)*delta
H += q.s(0,2)*Omega_A + q.s(1,2)*Omega_B
H += q.s(2,0)*Omega_A + q.s(2,1)*Omega_B
H /= 2

S += q.s(2,2) * Gamma
L += (q.s(0,0) + q.s(1,1))* Gamma/2 * rho[2,2]


###########################################################
Dro = -(H*rho - rho*H)*parse_expr('1j') - ( S*rho + rho*S )/2 + L
###########################################################

q.load_Drho(Dro)
q.make_symbolic_M()

q.set_constant(Gamma=6e6, Omega_A=1e5, Omega_B=1e5)

q.Ms.prepare_fast_eval()



###############################################################################
# Pruebo funciones de evolucion

rho0 = np.zeros((q.N,q.N))
rho0[0,0] = 1


rho1 = q.evol(1,rho0,100,1000)



fig, axx = plt.subplots(1,3, figsize=(13,6),  constrained_layout=True , sharey=True)

######################
ax = axx[0]

times = np.logspace(-6,0)
populations = np.array([ abs(np.diag(q.evol(t,rho0,Delta=100,delta=1000))) for t in times ]).T

for pop in populations:
    ax.plot(times,pop)

times = np.logspace(-6,0)
populations = np.array([ abs(np.diag(q.evol(t,rho0,Delta=300,delta=1e7))) for t in times ]).T

for ii,pop in enumerate(populations):
    ax.plot(times,pop,'--',color=f'C{ii}')


ax.semilogx()
ax.semilogy()
ax.set_xlabel('time [s]')
ax.set_ylabel('population')
ax.set_title('time evolution\nDelta=100 Hz,delta=1 kHz [-]\nDelta=300 Hz,delta=10 MHz [--]')

######################
ax = axx[1]

delta_vec   = np.linspace(-1e8,1e8,101)
populations = np.array([ abs(np.diag(q.evol(1,rho0,Delta=10000,delta=dd))) for dd in delta_vec ]).T

for pop in populations:
    ax.plot(delta_vec,pop)

delta_vec   = np.linspace(-1e8,1e8,101)
populations = np.array([ abs(np.diag(q.evol(1,rho0,Delta=100,delta=dd))) for dd in delta_vec ]).T

for ii,pop in enumerate(populations):
    ax.plot(delta_vec,pop,'--',color=f'C{ii}')

# ax.semilogx()
ax.semilogy()
ax.set_xlabel('delta [Hz]')
# ax.set_ylabel('population')
ax.set_title('variation of delta\nDelta=10 kHz,time=1s [-]\nDelta=0 Hz,time=1s [--]')

######################
ax = axx[2]

# Delta_vec = np.linspace(-1e8,1e8,101)
Delta_vec   = np.array(sorted(np.logspace(3,1).tolist() + [0] + (-np.logspace(3,1)).tolist()))
populations = np.array([ abs(np.diag(q.evol(1,rho0,Delta=dd,delta=100))) for dd in Delta_vec ]).T

for pop in populations:
    ax.plot(Delta_vec,pop)

# Delta_vec = np.linspace(-1e8,1e8,101)
Delta_vec   = np.array(sorted(np.logspace(3,1).tolist() + [0] + (-np.logspace(3,1)).tolist()))
populations = np.array([ abs(np.diag(q.evol(1,rho0,Delta=dd,delta=1e7))) for dd in Delta_vec ]).T

for ii,pop in enumerate(populations):
    ax.plot(Delta_vec,pop,'--',color=f'C{ii}')


# ax.semilogx()
ax.semilogy()
ax.set_xlabel('Delta [Hz]')
# ax.set_ylabel('population')
ax.set_title('variation of Delta\ndelta=100 Hz,time=1s [-]\ndelta=10 MHz,time=1s [--]')


ax.set_ylim(1e-8,1.2)

for ax in axx.flatten():
    ax.grid(True,ls=":", color='lightgray')


fig.savefig('example_3_levels.png')

plt.show()


