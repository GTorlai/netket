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

#ifndef NETKET_SPINTRANSLATIONPROJECTOR_HPP
#define NETKET_SPINTRANSLATIONPROJECTOR_HPP

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>
#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/exceptions.hpp"
#include "Utils/json_helper.hpp"
#include "Utils/messages.hpp"
#include "abstract_operator.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph
class SpinTranslationProjector : public AbstractOperator {
  const AbstractHilbert &hilbert_;
  const AbstractGraph &graph_;

  int nspins_;
  // cutoff in occupation number
  double k_momentum_;
  std::complex<double> I_;
  std::vector<std::vector<int> > symm_table_;

  double alpha_;

  // list of bonds for the interaction part
  std::vector<std::vector<int>> bonds_;

public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  explicit SpinTranslationProjector(const AbstractHilbert &hilbert,int k_index = 0,
      double alpha = -1.0) 
      : hilbert_(hilbert),
        graph_(hilbert.GetGraph()),
        nspins_(hilbert.Size()),
        alpha_(alpha),
        I_(0,1){
    k_momentum_ = 2*M_PI*k_index/double(nspins_);
    Init();
  }

  void Init() {
    GenerateBonds();
    symm_table_ = graph_.SymmetryTable();
  }

  void GenerateBonds() {
    auto adj = graph_.AdjacencyList();

    bonds_.resize(nspins_);

    for (int i = 0; i < nspins_; i++) {
      for (auto s : adj[i]) {
        if (s > i) {
          bonds_[i].push_back(s);
        }
      }
    }
  }

  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> 
&mel,
                 std::vector<std::vector<int>> &connectors,
                 std::vector<std::vector<double>> &newconfs) const 
override {
  
    double normalization = nspins_;
    connectors.clear();
    connectors.resize(1);
    newconfs.clear();
    newconfs.resize(1);
    mel.resize(1);
    std::vector<int> conn_tmp;
    std::vector<double> newconf_tmp;
    std::complex<double> phase_k;

    mel[0] = 0.;
    connectors[0].resize(0);
    newconfs[0].resize(0);
    Eigen::VectorXd vp(nspins_);

    for (int p = 0; p < nspins_; p++) { 
      
      phase_k = std::exp(-I_*std::complex<double>(k_momentum_*p))/double(normalization);

      for(int j=0;j<nspins_;j++){
        vp(j) = v(symm_table_[p][j]);
      }
      if(vp==v){
        mel[0] += alpha_ * phase_k;
      }
      else{
        conn_tmp.clear();
        newconf_tmp.clear();
        for(int j=0;j<nspins_;j++){
          if (vp(j) != v(j)){
            conn_tmp.push_back(j);
            newconf_tmp.push_back(vp(j));
          }
        }
        connectors.push_back(conn_tmp);
        newconfs.push_back(newconf_tmp);
        mel.push_back(alpha_ * phase_k);
      }
    }
  }//function


  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }
};

}  // namespace netket

#endif
