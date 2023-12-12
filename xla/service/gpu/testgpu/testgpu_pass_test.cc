/* Copyright 2021 The TESTGPU Authors. All Rights Reserved.

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

#include "xla/service/gpu/testgpu/testgpu_pass.h"

#include <cstddef>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/gpu/testgpu/testgpu_c_api.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/statusor.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace m = xla::testing::opcode_matchers;
using ::testing::_;

class TESTGPUsoPassTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(absl::string_view hlo_module,
                                               bool expect_change) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed =
        TESTGPUsoPass(TESTGPUPassType::TESTGPU_TEST_PASS).Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
  template <HloOpcode oc>
  size_t CollectiveCount(std::unique_ptr<HloModule> &module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<oc>);
  }
};

TEST_F(TESTGPUsoPassTest, Simple) {
  std::string hlo_string = R"(
    HloModule add
    ENTRY add {
      x = f32[3,2] parameter(0)
      y = f32[3,2] parameter(1)
      ROOT add = f32[3,2] add(x, y)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(CollectiveCount<HloOpcode::kAdd>(module), 0);
  EXPECT_EQ(CollectiveCount<HloOpcode::kMultiply>(module), 1);
}

TEST_F(TESTGPUsoPassTest, PipelineTest) {
  // Test an HLO module pass which changes a module.
  const std::string module_str = R"(
HloModule ModulePassChanged
ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<TESTGPUsoPass>(TESTGPUPassType::TESTGPU_TEST_PASS);

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);
}

// TEST_F(TESTGPUsoPassTest, ReduceScatterAllGatherCombiner) {
//   // Test an HLO module pass which changes a module.
//   const std::string module_str = R"(
//  HloModule ReduceScatter

//  add {
//    x = bf16[] parameter(0)
//    y = bf16[] parameter(1)
//    ROOT add = bf16[] add(x, y)
//  }

//  ENTRY main {
//  param.1 = bf16[8,128,1024] parameter(0)
//  param.2 = bf16[8,128,1024] parameter(1)
//  reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
//  all-gather.1 = bf16[8,128,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
//  reduce-scatter.2 = bf16[8,64,1024] reduce-scatter(param.2), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
//  all-gather.2 = bf16[8,128,1024] all-gather(reduce-scatter.2), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
//  add.1 = bf16[8,128,1024] add(all-gather.1, all-gather.2)
//  }
//  )";

//   TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
//                           ParseAndReturnVerifiedModule(module_str));

//   HloPassPipeline pipeline(TestName());
//   pipeline.AddPass<TESTGPUsoPass>(
//       TESTGPUPassType::REDUCE_SCATTER_ALL_GATHER_COMBINER_PASS);

//   TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));

//   EXPECT_TRUE(changed);
// }

//enable to run the auto sharding test

// TEST_F(TESTGPUsoPassTest, AutoShardingTest) {
//   // Test an HLO module pass which changes a module.
//   const std::string module_str = R"(
// HloModule module
// ENTRY twomatmul {
//   parameter.1 = f32[64,64]{1,0} parameter(0)
//   parameter.2 = f32[64,128]{1,0} parameter(1)
//   dot.4 = f32[64,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
//   parameter.3 = f32[128,64]{1,0} parameter(2)
//   ROOT dot.5 = f32[64,64]{1,0} dot(dot.4, parameter.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// })";
//   TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
//                           ParseAndReturnVerifiedModule(module_str));
//   TESTGPUShardingParams *params =
//       new TESTGPUShardingParams{/*  num_cores = */ 4,
//                               /* device_mesh_shape = */ {2, 2},
//                               /* device_mesh_ids = */ {0, 1, 2, 3},
//                               /* device_mesh_alpha = */ {1.0, 1.0},
//                               /* device_mesh_beta = */ {0.01, 1.0}};

//   HloPassPipeline pipeline(TestName());
//   pipeline.AddPass<TESTGPUsoPass>(TESTGPUPassType::TESTGPU_AUTO_PARTITION,
//                                     (void *)params);

//   TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
//   std::cout << module.get()->ToString();
//   EXPECT_TRUE(changed);
// }

}  // namespace
}  // namespace gpu
}  // namespace xla