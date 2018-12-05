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

L = 20

# Constructing a 1d lattice
g = nk.graph.Hypercube(length=L, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)

# Hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Layers
layers = [
    nk.layer.ConvolutionalHypercube(
        length=L,
        n_dim=1,
        input_channels=1,
        output_channels=4,
        kernel_length=4),
    nk.layer.Lncosh(input_size=4 * L),
    nk.layer.ConvolutionalHypercube(
        length=4 * L,
        n_dim=1,
        input_channels=1,
        output_channels=2,
        kernel_length=4),
    nk.layer.Lncosh(input_size=4 * 2 * L)
]

# FFNN Machine
ma = nk.machine.FFNN(hi, layers)
ma.init_random_parameters(seed=1234, sigma=0.1)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Variational Monte Carlo
gs = nk.gs.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    nsamples=1000,
    niter_opt=300,
    output_file='test',
    diag_shift=0.01)
gs.run()