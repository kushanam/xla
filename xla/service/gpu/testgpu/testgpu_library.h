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

#ifndef TESTGPU_LIBRARY_H_
#define TESTGPU_LIBRARY_H_

#include "xla/service/gpu/testgpu/testgpu_api.h"
#include "xla/service/gpu/testgpu/testgpu_c_api.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

typedef void (*PassPtr)(TESTGPU_Pass_Args* args);

class TESTGpuLibrary {
 public:
  // Singleton constructor-or-accessor

  // Get the Library API
  const static TESTGPU_Api* GetOrCreateLibraryAPI();

  // Get a pointer for a specific api call
  const static PassPtr GetPassAPI(TESTGPUPassType pass_type);

  // Clears the local instance
  static void DestroyLibraryInstances();

 private:
  // Returns the singleton instance of TESTGpuLibrary.
  static TESTGpuLibrary& Singleton();

  TESTGpuLibrary();
  ~TESTGpuLibrary();

  const TESTGPU_Api* _api;

  TESTGpuLibrary(const TESTGpuLibrary&) = delete;
  TESTGpuLibrary& operator=(const TESTGpuLibrary&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // TESTGPU_LIBRARY_H_