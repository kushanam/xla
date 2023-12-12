/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/testgpu/testgpu_library.h"

namespace xla {
namespace gpu {

// TODO: move this to a helper class
static std::string GetEnvVar() {
  const char* env_value = std::getenv("NVSO_LIB_PATH");
  if (!env_value) return "testgpulib_so.so";
  return std::string(env_value);
}

/* static */ TESTGpuLibrary& TESTGpuLibrary::Singleton() {
  static TESTGpuLibrary* c = new TESTGpuLibrary;
  return *c;
}

TESTGpuLibrary::TESTGpuLibrary() { _api = LoadTESTGPUso(GetEnvVar()).value(); }

/* static */ const TESTGPU_Api* TESTGpuLibrary::GetOrCreateLibraryAPI() {
  TESTGpuLibrary& n = Singleton();
  return n._api;  // TODO: use a getter instead
}

/* static*/ const PassPtr TESTGpuLibrary::GetPassAPI(TESTGPUPassType pass_type) {
  auto api = GetOrCreateLibraryAPI();
  switch (pass_type) {
    case TESTGPUPassType::TESTGPU_TEST_PASS:
      return api->TESTGPU_PASS_TEST;
    default:
      return nullptr;  // TODO: handle this as error
  }
}

void TESTGpuLibrary::DestroyLibraryInstances() {
  // We will need the insance to clear options
  // TESTGpuLibrary& nvgl = Singleton();
}

}  // namespace gpu
}  // namespace xla