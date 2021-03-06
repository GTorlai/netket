import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

operators = {}

# Ising 1D
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, graph=g)
operators["Ising 1D"] = nk.operator.Ising(h=1.321, hilbert=hi)

# Heisenberg 1D
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
operators["Heisenberg 1D"] = nk.operator.Heisenberg(hilbert=hi)

# Bose Hubbard
g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
operators["Bose Hubbard"] = nk.operator.BoseHubbard(U=4.0, hilbert=hi)


# Graph Hamiltonian
# TODO (jamesETsmith)

# Custom Hamiltonian
# TODO (jamesETsmith)
#sx = [[0,1],[1,0]]
#szsz = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
#sy = np.array([[0,1j],[-1j,0]])
#

sx = [[0, 1], [1, 0]]
sy = [[0, 1.0j], [-1.0j, 0]]
sz = [[1, 0], [0, -1]]
g = nk.graph.CustomGraph(edges=[[i, i + 1] for i in range(20)])
hi = nk.hilbert.CustomHilbert(local_states=[1, -1], graph=g)

sx_hat = nk.operator.LocalOperator(hi, [sx] * 3, [[0], [1], [5]])
sy_hat = nk.operator.LocalOperator(hi, [sy] * 4, [[2], [3], [4], [9]])
szsz_hat = nk.operator.LocalOperator(hi, sz, [0]) * nk.operator.LocalOperator(
    hi, sz, [1])
szsz_hat += nk.operator.LocalOperator(hi, sz, [4]) * nk.operator.LocalOperator(
    hi, sz, [5])
szsz_hat += nk.operator.LocalOperator(hi, sz, [6]) * nk.operator.LocalOperator(
    hi, sz, [8])
szsz_hat += nk.operator.LocalOperator(hi, sz, [7]) * nk.operator.LocalOperator(
    hi, sz, [0])

operators["Custom Hamiltonian"] = sx_hat + sy_hat


operators["Custom Hamiltonian Prod"] = sx_hat * 1.5 + 2.0 * sy_hat

rg = nk.utils.RandomEngine(seed=1234)


def test_produce_elements_in_hilbert():
    for name, ha in operators.items():
        hi = ha.hilbert
        print(name, hi)
        assert (len(hi.local_states) == hi.local_size)

        rstate = np.zeros(hi.size)

        local_states = hi.local_states

        for i in range(1000):
            hi.random_vals(rstate, rg)
            conns = ha.get_conn(rstate)

            for connector, newconf in zip(conns[1], conns[2]):
                rstatet = np.array(rstate)
                hi.update_conf(rstatet, connector, newconf)

                for rs in rstatet:
                    assert(rs in local_states)


def test_operator_is_hermitean():
    for name, ha in operators.items():
        hi = ha.hilbert
        print(name, hi)
        assert (len(hi.local_states) == hi.local_size)

        rstate = np.zeros(hi.size)

        local_states = hi.local_states

        for i in range(100):
            hi.random_vals(rstate, rg)
            conns = ha.get_conn(rstate)

            for mel, connector, newconf in zip(conns[0], conns[1], conns[2]):
                rstatet = np.array(rstate)
                hi.update_conf(rstatet, connector, newconf)

                conns1 = ha.get_conn(rstatet)
                foundinv = False
                for meli, connectori, newconfi in zip(conns1[0], conns1[1], conns1[2]):
                    rstatei = np.array(rstatet)
                    hi.update_conf(rstatei, connectori, newconfi)
                    if(np.array_equal(rstatei, rstate)):
                        foundinv = True
                        assert(meli == np.conj(mel))
                assert(foundinv)
