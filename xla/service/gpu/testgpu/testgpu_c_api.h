/* Copyright 2023 The TESTGPU SO Authors. All Rights Reserved.

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

#ifndef TESTGPU_C_API_H_
#define TESTGPU_C_API_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------------- API access ---------------------------------

#define _TESTGPU_STRUCT_FIELD(fn_type) fn_type* fn_type

struct TESTGPU_so_Initialize_Args {
  void* arg0;
};

// TODO: (Kushan) should be TESTGPUError
typedef void TESTGPU_so_Initialize(TESTGPU_so_Initialize_Args* args);

struct TESTGPU_Pass_Args {
  void* module;
  const void* execution_threads;
  void* status;
  void* args;
};

typedef void TESTGPU_PASS_TEST(TESTGPU_Pass_Args* args);

typedef struct {
  _TESTGPU_STRUCT_FIELD(TESTGPU_so_Initialize);
  _TESTGPU_STRUCT_FIELD(TESTGPU_PASS_TEST);
} TESTGPU_Api;

typedef enum {
  TESTGPU_TEST_PASS = 0,
} TESTGPUPassType;

#ifdef __cplusplus
}
#endif

}  // namespace gpu
}  // namespace xla

#endif  // XTESTGPU_C_API_H_