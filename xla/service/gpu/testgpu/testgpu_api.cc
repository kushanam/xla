/* Copyright 2023 The TESTGPU Authors. All Rights Reserved.

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

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "xla/service/gpu/testgpu/testgpu_c_api.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

static TESTGPU_Api* testgpu_api;
static const char* sym_name = "GetTESTGPUApi";

StatusOr<void*> GetDsoHandle(const std::string& library_path) {
  void* dso_handle;
  Status status = tsl::Env::Default()->LoadDynamicLibrary(library_path.c_str(),
                                                          &dso_handle);
  if (status.ok()) {
    VLOG(1) << "Successfully opened dynamic library " << library_path;
    return dso_handle;
  }

  auto message = absl::StrCat("Could not load dynamic library '", library_path,
                              "'; dlerror: ", status.message());
  VLOG(1) << message;
  return Status(absl::StatusCode::kFailedPrecondition, message);
}

StatusOr<void*> GetDsoSymbol(void* dso_handle, const char* symbol_name) {
  void* dso_symbol;
  Status status = tsl::Env::Default()->GetSymbolFromLibrary(
      dso_handle, symbol_name, &dso_symbol);
  if (status.ok()) {
    VLOG(1) << "Successfully loaded symbol " << symbol_name;
    return dso_symbol;
  }
  auto message = absl::StrCat("Could not load symbol '", symbol_name,
                              "'; dlsym error: ", status.message());
  VLOG(1) << message;
  return Status(absl::StatusCode::kFailedPrecondition, message);
}

typedef TESTGPU_Api* (*TESTGPUApiInitFn)();
xla::StatusOr<const TESTGPU_Api*> LoadTESTGPUso(
    const std::string& library_path) {
#ifdef PLATFORM_WINDOWS
  return tsl::errors::Unimplemented(
      "LoadTESTGPUso is not implemented on windows yet.");
#else
  auto status_or_handle = GetDsoHandle(library_path);
  if (!status_or_handle.ok()) {
    return tsl::errors::Internal("Failed to open the library", library_path);
  }
  auto status_or_symbol = GetDsoSymbol(status_or_handle.value(), sym_name);

  if (!status_or_symbol.ok()) {
    return tsl::errors::Internal("Failed to load symbol", sym_name);
  }
  TESTGPUApiInitFn init_fn;
  *reinterpret_cast<void**>(&init_fn) = status_or_symbol.value();
  testgpu_api = init_fn();
  return testgpu_api;
#endif
}
xla::Status InitializeTestgpuso() {
  // TODO: Add initialization
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
