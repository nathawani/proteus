#!/usr/bin/env python
""" Test modules for Driven Cavity Stokes preconditioners. """

import proteus.test_utils.TestTools as TestTools
from proteus.iproteus import *

Profiling.logLevel = 7
Profiling.verbose = False

import os
import sys
import inspect
import numpy as np
import tables
import pickle
import petsc4py
import pytest

proteus.test_utils.TestTools.addSubFolders( inspect.currentframe() )
import nseDrivenCavity_2d_n
import nseDrivenCavity_2d_p
from NavierStokes_ST_LS_SO_VV import NavierStokes_ST_LS_SO_VV

@pytest.mark.LinearSolvers
@pytest.mark.modelTest
@pytest.mark.navierstokesTest
class Test_NSE_Driven_Cavity(proteus.test_utils.TestTools.SimulationTest):

    def setup_method(self):
        reload(nseDrivenCavity_2d_p)
        reload(nseDrivenCavity_2d_n)
        self.pList = [nseDrivenCavity_2d_p]
        self.nList = [nseDrivenCavity_2d_n]
        self.so = default_so
        self.so.name = self.pList[0].name
        self.so.sList = self.pList[0].name
        self.so.sList = [default_s]
        self._scriptdir = os.path.dirname(__file__)
        Profiling.openLog("proteus.log",10)
        Profiling.logAllProcesses = True

    def teardown_method(self):
        Profiling.closeLog()

    def _runTest(self):
        self.ns = NumericalSolution.NS_base(self.so,
                                            self.pList,
                                            self.nList,
                                            self.so.sList,
                                            opts)
        self.ns.calculateSolution('stokes')

        # The produced output has diverged from the old comparison
        # output. It needs to be confirmed that the new ouput is
        # in fact correct and drivenCavityNSE_LSC_expected.h5 should
        # be updated accordingly.
        actual = tables.open_file('drivenCavityNSETrial.h5','r')
        relpath = 'comparison_files/drivenCavityNSE_LSC_expected.csv'
        #np.savetxt(os.path.join(self._scriptdir,relpath),actual.root.velocity_t7.read(),delimiter=',')
        expected = np.loadtxt(os.path.join(self._scriptdir,relpath),delimiter=',')
        assert np.allclose(expected,
                           actual.root.velocity_t7.read(),
                           rtol=1e-8, atol=1e-8) 
        actual.close()

    @pytest.mark.slowTest
    def test_01_FullRun(self):
        """Runs with nsedriven cavity with the following settings.
        - RE enforced through viscosity
        - THQuads
        - LSC
        """
        nseDrivenCavity_2d_p.coefficients = NavierStokes_ST_LS_SO_VV(epsFact=0.0,
                                                                     sigma=0.0,
                                                                     rho_0=1.0,
                                                                     nu_0=1.0,
                                                                     rho_1=1.0,
                                                                     nu_1=1.0,
                                                                     g=[0.0,0.0],
                                                                     nd=2,
                                                                     LS_model=None,
                                                                     KN_model=None,
                                                                     epsFact_density=None,
                                                                     stokes=False);

        nseDrivenCavity_2d_p.coefficients.variableNames = ['p','u','v']

        nseDrivenCavity_2d_p.dirichletConditions = {0:nseDrivenCavity_2d_p.getDBCp,
                                                    1:nseDrivenCavity_2d_p.getDBCu,
                                                    2:nseDrivenCavity_2d_p.getDBCv}
        nseDrivenCavity_2d_n.nLevels = 1
        nseDrivenCavity_2d_n.linearSmoother = nseDrivenCavity_2d_n.Schur_LSC
        self.so.tnList = self.nList[0].tnList
        self._setPETSc(petsc_file = os.path.join(self._scriptdir,'import_modules/petsc.options.schur_lsc'))
        self._runTest()

if __name__ == '__main__':
    pass