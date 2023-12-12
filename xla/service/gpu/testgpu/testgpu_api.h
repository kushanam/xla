/* Copyright 2023 The TESTGPU SO  Authors. All Rights Reserved.

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

#ifndef TESTGPU_API_H_
#define TESTGPU_API_H_

#include "absl/strings/string_view.h"
#include "xla/service/gpu/testgpu/testgpu_c_api.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

xla::StatusOr<void*> GetDsoHandle(const std::string& library_path);
xla::StatusOr<void*> GetDsoSymbol(void* dso_handle, const char* symbol_name);
// Loads the TESTGPU so.
xla::StatusOr<const TESTGPU_Api*> LoadTESTGPUso(
    const std::string& library_path);
// Requires that SetTestGPUApi has been successfully called
xla::StatusOr<const TESTGPU_Api*> IsTESTGPUsoInitialized();
// Initializes a TESTGPU so with `TESTGPU_so_Initialize`.
xla::Status InitializeTestgpuso();

}  // namespace gpu
}  // namespace xla

#endif  // TESTGPU_API_H_