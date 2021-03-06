# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import netket as nk
from mpi4py import MPI

# Constructing a 1d lattice
g = nk.graph.Hypercube(length=20, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

# Variational Monte Carlo
vmc = nk.gs.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    nsamples=1000,
    niter_opt=300,
    output_file='test',
    diag_shift=0.1,
    method='Sr')
vmc.run()
