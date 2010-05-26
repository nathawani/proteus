from pyadh import *
from pyadh.default_p import *
from porousObstacle import *
"""
Two-phase incompressible Navier-Stokes flow of an air porousObstacle in water
"""

## \page Tests Test Problems
#\ref twp_navier_stokes_ls_so_porousObstacle_2d_p.py "Two-phase incompressible Navier-Stokes flow of an air porousObstacle in water"
# 

##\ingroup test
# \file twp_navier_stokes_ls_so_porousObstacle_2d_p.py
# @{
#
# \brief Two-phase incompressible Navier-Stokes flow of an air porousObstacle in water.
#
# The governing equations are described by the
# pyadh::TransportCoefficients::TwophaseNavierStokes_LS_SO class. The
# domain is the unit square and the boundary conditions are slip and
# no flow on the whole boundary. The pressure is set (arbitrarily) to
# zero at the node in the top left corner.
#
#
#\image html porousObstacle_1.jpg
#\image html porousObstacle_2.jpg
#
# <A href="https://adh.usace.army.mil/pyadh-images/porousObstacle.avi"> AVI Animation of velocity and pressure </A>
#

name = "NavierStokes"

analyticalSolution = None

coefficients = VolumeAveragedTwophaseNavierStokes_ST_LS_SO(epsFact=epsFact,
                                                           sigma=sigma_01,
                                                           rho_0 = rho_0,
                                                           nu_0 = nu_0,
                                                           rho_1 = rho_1,
                                                           nu_1 = nu_1,
                                                           g=g,
                                                           nd=nd,
                                                           LS_model=2,
                                                           KN_model=0,
                                                           meanGrainSize=0.01,
                                                           setParamsFunc=None,#setObstaclePorosity,
                                                           stokesOnly=False,
                                                           meanGrainSizeTypes=meanGrainTypes,
                                                           porosityTypes=porosityTypes)
# coefficients = TwophaseStokes_LS_SO(epsFact=epsFact,
#                                     rho_0 = rho_0,
#                                     nu_0 = nu_0,
#                                     rho_1 = rho_1,
#                                     nu_1 = nu_1,
#                                     g=g,
#                                     nd=nd,
#                                     LS_model=3)
# coefficients = TwophaseStokes_LS_SO(epsFact=epsFact,
#                                     rho_0 = rho_0,
#                                     nu_0 = nu_0,
#                                     rho_1 = rho_1,
#                                     nu_1 = nu_1,
#                                     g=g,
#                                     nd=nd,
#                                     steady=False)
#coefficients.rho_1 = coefficients.rho_0
#coefficients.mu_1 = coefficients.mu_0
