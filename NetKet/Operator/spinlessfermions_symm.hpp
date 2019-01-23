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

#ifndef NETKET_SPINLESSFERMIONSSYMM_HPP
#define NETKET_SPINLESSFERMIONSSYMM_HPP

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
class SpinlessFermionsSymm : public AbstractOperator {
  const AbstractHilbert &hilbert_;
  const AbstractGraph &graph_;

  int nsites_;
  int L_;
  // cutoff in occupation number
  int nmax_;
  std::vector<double> k_momentum_;
  std::complex<double> I_;
  std::vector<std::vector<int> > symm_table_;

  double V_;

  double mu_;

  // list of bonds for the interaction part
  std::vector<std::vector<int>> bonds_;

public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  explicit SpinlessFermionsSymm(const AbstractHilbert &hilbert,std::vector<int> k_index, 
      double V = 0.,double mu = 0.)
      : hilbert_(hilbert),
        graph_(hilbert.GetGraph()),
        nsites_(hilbert.Size()),
        V_(V),
        mu_(mu) {
    L_ = int(std::sqrt(nsites_));
    nmax_ = hilbert_.LocalSize() - 1;
    for (int i=0;i<k_index.size();i++){
      k_momentum_.push_back(2*M_PI*k_index[i]/double(nsites_));
    }
    Init();
  }

  void Init() {
    GenerateBonds();
    symm_table_ = graph_.SymmetryTable();
    //for (int i=0;i<symm_table_.size();i++){
    //  std::cout<<"i = " << i << " :  ";
    //  for(int j=0;j<symm_table_[i].size();j++){
    //    std::cout<<symm_table_[i][j]<<" ";
    //  }
    //  std::cout<<std::endl;
    //}
    //std::cout<<"\n\n";

    InfoMessage() << "Spinless Fermion model created \n";
    InfoMessage() << "V= " << V_ << std::endl;
    InfoMessage() << "mu= " << mu_ << std::endl;
    InfoMessage() << "Nmax= " << nmax_ << std::endl;
  }

  void GenerateBonds() {
    auto adj = graph_.AdjacencyList();

    bonds_.resize(nsites_);

    for (int i = 0; i < nsites_; i++) {
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
  
    double normalization = nsites_;
    connectors.clear();
    connectors.resize(1);
    newconfs.clear();
    newconfs.resize(1);
    mel.resize(1);
    std::vector<int> conn_tmp;
    std::vector<double> newconf_tmp;
    int symm_index;
    std::complex<double> phase_k;
    double fermisign;

    mel[0] = 0.;
    connectors[0].resize(0);
    newconfs[0].resize(0);
    Eigen::VectorXd vp(nsites_);
    Eigen::VectorXd voff(nsites_);
    Eigen::VectorXd vcheck(nsites_);

    for(int y=0;y<L_;y++){
      for(int x=0;x<L_;x++){
        symm_index = y*L_+x;
        
        double dot_product = k_momentum_[0]*x + k_momentum_[1]*y;
        phase_k = std::exp(-I_*std::complex<double>(dot_product))/double(normalization);

        for(int j=0;j<nsites_;j++){
          vp(j) = v(symm_table_[symm_index][j]);
        }
        for (int i = 0; i < nsites_; i++) {
          for (auto bond : bonds_[i]) {
            if(vp==v){
              mel[0] += phase_k * V_ * 0.25*(1.0+vp(i)) * (1.0+vp(bond));
            }//if
            else{
              conn_tmp.clear();
              newconf_tmp.clear();
              for(int j=0;j<nsites_;j++){
                if (vp(j) != v(j)){
                  conn_tmp.push_back(j);
                  newconf_tmp.push_back(vp(j));
                }//if
              }//for
              connectors.push_back(conn_tmp);
              newconfs.push_back(newconf_tmp);
              mel.push_back(phase_k*V_ * 0.25*(1.0+vp(i)) * (1.0+vp(bond)));

              //Off diagonal
              voff = vp;
              conn_tmp.clear();
              newconf_tmp.clear();
              // Fermi sign
              int s1 = std::min(i, bond);
              int s2 = std::max(i, bond);
              fermisign = 1.;
              for (int k = s1+1; k < s2; ++k){
                if (vp(k)==1) fermisign *= -1.;
              }//for
              if (vp(i) != vp(bond)) {
                double tmp = voff(i);
                voff(i) = voff(bond);
                voff(bond) = tmp;
                for(int j=0;j<nsites_;j++){
                  if (voff(j) != v(j)){
                    conn_tmp.push_back(j);
                    newconf_tmp.push_back(voff(j));
                  }//if
                }//for
                connectors.push_back(conn_tmp);
                newconfs.push_back(newconf_tmp);
                mel.push_back(-phase_k*fermisign);
              }//if
            }//else
          }//for (bond)
        }//for (sites)
      }//x
    }//y
  }//function

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }
};

}  // namespace netket

#endif
