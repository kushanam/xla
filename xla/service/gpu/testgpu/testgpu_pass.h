/* Copyright 2019 The TESTGPU Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TESTGPU_PASS_H
#define TESTGPU_PASS_H

#include <memory>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/testgpu/testgpu_c_api.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

struct TESTGPUShardingParams
{
    int num_cores;
    // Device IDs in the mesh.
    // Device mesh shape.
  std::vector<int64_t> device_mesh_shape;
  std::vector<int64_t> device_mesh_ids;
  // We use an alpha-beta model as the communication model:
  //   latency = alpha + beta * size
  // the following two vectors have the same size as device_mesh_shape and each
  // element models the communication performance along each mesh dimension.
  std::vector<double> device_mesh_alpha;
  std::vector<double> device_mesh_beta;

};

class TESTGPUsoPass : public HloModulePass {
 public:
  explicit TESTGPUsoPass(TESTGPUPassType pass_type, void* pass_args = nullptr) : pass_type_(pass_type), pass_args_(pass_args) {}
  ~TESTGPUsoPass() override = default;
  // TODO: change to have customized name.
  absl::string_view name() const override { return "testgpu_so_pass"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  TESTGPUPassType pass_type_;
  void* pass_args_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TESTGPU_PASS_H
