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

#ifndef NETKET_SPINLESSFERMIONS_HPP
#define NETKET_SPINLESSFERMIONS_HPP

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
class SpinlessFermions : public AbstractOperator {
   const AbstractHilbert &hilbert_;
   const AbstractGraph &graph_;

   int nsites_;

   // cutoff in occupation number
   int nmax_;

   double V_;

   double mu_;

   // list of bonds for the interaction part
   std::vector<std::vector<int>> bonds_;

  public:
   using VectorType = AbstractOperator::VectorType;
   using VectorRefType = AbstractOperator::VectorRefType;
   using VectorConstRefType = AbstractOperator::VectorConstRefType;

   explicit SpinlessFermions(const AbstractHilbert &hilbert, double V = 0.,
                 double mu = 0.)
       : hilbert_(hilbert),
         graph_(hilbert.GetGraph()),
         nsites_(hilbert.Size()),
         V_(V),
         mu_(mu) {
     nmax_ = hilbert_.LocalSize() - 1;
     Init();
   }

   void Init() {
     GenerateBonds();
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
     connectors.clear();
     connectors.resize(1);
     newconfs.clear();
     newconfs.resize(1);
     mel.resize(1);

     mel[0] = 0.;
     connectors[0].resize(0);
     newconfs[0].resize(0);

     for (int i = 0; i < nsites_; i++) {
       // chemical potential
       mel[0] -= mu_ * 0.5*(1.0+v(i));

       for (auto bond : bonds_[i]) {
         // nn interaction
         mel[0] += V_ * 0.25*(1.0+v(i)) * (1.0+v(bond));

     // Fermi sign
     int s1 = std::min(i, bond);
     int s2 = std::max(i, bond);
     double fermisign = 1.;
     for (int k = s1+1; k < s2; ++k)
       if (v(k)==1) fermisign *= -1.;

         // hopping
         if (v(i)!=v(bond)) {
           connectors.push_back(std::vector<int>({i, bond}));
           newconfs.push_back(std::vector<double>({v(bond),v(i)}));
           mel.push_back(-fermisign);
         }
       }
     }
   }

   const AbstractHilbert &GetHilbert() const noexcept override {
     return hilbert_;
   }
};

}  // namespace netket

#endif
