// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_HEISENBERGSYMM_HPP
#define NETKET_HEISENBERGSYMM_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "abstract_operator.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph

class HeisenbergSymm : public AbstractOperator {
  const AbstractHilbert &hilbert_;
  const AbstractGraph &graph_;

  const int nspins_;
  double offdiag_;
  double k_momentum_;
  std::complex<double> I_;
  std::vector<std::vector<int> > symm_table_;

  // list of bonds for the interaction part
  std::vector<std::vector<int>> bonds_;

 public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  explicit HeisenbergSymm(const AbstractHilbert &hilbert, double k_momentum = 0.0)
      : hilbert_(hilbert), graph_(hilbert.GetGraph()), nspins_(hilbert.Size()),
        k_momentum_(k_momentum),
        I_(0,1){
    Init();
  }

  void Init() {
    if (graph_.IsBipartite()) {
      offdiag_ = -2;
    } else {
      offdiag_ = 2;
    }

    GenerateBonds();
    symm_table_ = graph_.SymmetryTable();
    InfoMessage() << "Heisenberg model created " << std::endl;
  }

  void GenerateBonds() {
    auto adj = graph_.AdjacencyList();

    bonds_.resize(nspins_);

    for (int i = 0; i < nspins_; i++) {
      for (auto s : adj[i]) {
        if (s > i) {
          bonds_[i].push_back(s);
          std::cout<<"i = "<<i<<"bonds = "<<s<<std::endl;
        }
      }
      std::cout<<std::endl;
    }
  }

  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    
    double normalization = nspins_;
    connectors.clear();
    connectors.resize(1);
    newconfs.clear();
    newconfs.resize(1);
    mel.resize(1);

    // computing interaction part Sz*Sz
    mel[0] = 0.;
    connectors[0].resize(0);
    newconfs[0].resize(0);
    Eigen::VectorXd vp(nspins_);
    for (int p = 0; p < nspins_; p++) {
      for(int j=0;j<nspins_;j++){
        vp(j) = v(symm_table_[p][j]);
      }
      std::cout<<vp.transpose()<<std::endl;
      for (int i = 0; i < nspins_; i++) {
        for (auto bond : bonds_[i]) {
          // interaction part
          mel[0] += std::exp(-I_*std::complex<double>(k_momentum_*p)) * vp(bond) * vp(i) / double(normalization);
          
          // spin flips
          if (vp(i) != vp(bond)) {
            connectors.push_back(std::vector<int>({i, bond}));
            newconfs.push_back(std::vector<double>({vp(bond), vp(i)}));
            mel.push_back(std::exp(-I_*std::complex<double>(k_momentum_*p)) * offdiag_/ double(normalization));
          }
        }
      }
    }
    std::cout<<std::endl;
  }


//  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
//                std::vector<std::vector<int>> &connectors,
//                std::vector<std::vector<double>> &newconfs) const override {
//    double normalization = nspins_;
//    connectors.clear();
//    connectors.resize(1);
//    newconfs.clear();
//    newconfs.resize(1);
//    mel.resize(1);
//
//    // computing interaction part Sz*Sz
//    mel[0] = 0.;
//    connectors[0].resize(0);
//    newconfs[0].resize(0);
//    
//    for (int i = 0; i < nspins_; i++) {
//      for (int p = 0; p < nspins_; p++) {
//        std::cout<<"hi = "<<i<<"  p = "<<p<<"  symm_tab = "<<symm_table_[p][i]<<std::endl;
//        for (auto bond : bonds_[symm_table_[p][i]]) {
//          // interaction part
//          std::cout<<"  bond = "<<bond;
//          mel[0] += std::exp(-I_*std::complex<double>(k_momentum_*p))*v(symm_table_[p][i]) * v(bond) / normalization;
//          
//          // spin flips
//          if (v(symm_table_[p][i]) != v(bond)) {
//            connectors.push_back(std::vector<int>({symm_table_[p][i], bond}));
//            newconfs.push_back(std::vector<double>({v(bond), v(symm_table_[p][i])}));
//            mel.push_back(std::exp(-I_*std::complex<double>(k_momentum_*p))*offdiag_/normalization);
//          }
//        }
//        std::cout<<std::endl;
//      }
//    }
//  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }
};

}  // namespace netket

#endif
