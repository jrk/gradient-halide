#ifndef HL_PYTORCH_WRAPPER_H
#define HL_PYTORCH_WRAPPER_H

#include "HalideBuffer.h"
#include "HalideRuntimeCuda.h"
#include <vector>
#include <iostream>

#include "TH/TH.h"
#include "THC/THC.h"

#define WEAK __attribute__((weak))

extern THCState *state;

using Halide::Runtime::Buffer;

namespace Halide {
namespace Pytorch {

inline Buffer<float> wrap(THFloatTensor* tensor) {
  int ndims = THFloatTensor_nDimension(tensor);
  std::vector<int> dims(ndims, 0);
  for(int dim = 0; dim < ndims; ++dim) {
    dims[dim] = THFloatTensor_size(tensor, ndims-1-dim);
  }
  float* pData  = THFloatTensor_data(tensor);
  Buffer<float> buffer(pData, dims);
  return buffer;
}

inline Buffer<float> wrap(THCudaTensor* tensor) {
  const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();

  int ndims = THCudaTensor_nDimension(state, tensor);
  std::vector<int> dims(ndims, 0);
  for(int dim = 0; dim < ndims; ++dim) {
    dims[dim] = THCudaTensor_size(state, tensor, ndims-1-dim);
  }

  Buffer<float> buffer(dims);

  float* pData  = THCudaTensor_data(state, tensor);
  int err = buffer.device_wrap_native(cuda_interface, (uint64_t)pData);
  if (err != 0) {
    throw "halide_device_wrap failed";
  }
  buffer.set_device_dirty();

  return buffer;
}

typedef struct UserContext {
  UserContext(int id, CUcontext *ctx, cudaStream_t* stream) :
    device_id(id), cuda_context(ctx), stream(stream) {};

  int device_id;
  CUcontext *cuda_context;
  cudaStream_t *stream;
} UserContext;

} // namespace Pytorch
} // namespace Halide


// Replace Halide weakly-linked cuda handles
extern "C" {

WEAK int halide_cuda_acquire_context(void *user_context, CUcontext *ctx, bool create = true) {
  if(user_context != NULL) {
    Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
    // std::cerr << "PyWrap get ctx " << *user_ctx->cuda_context << "\n";
    *ctx = *user_ctx->cuda_context;
  } else {
    // std::cerr << "no user context\n";
    *ctx = NULL;
  }
  return 0;
}

WEAK int halide_cuda_get_stream(void *user_context, CUcontext ctx, CUstream *stream) {
  if(user_context != NULL) {
    Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
    // std::cerr << "PyWrap's get stream " <<  *user_ctx->stream << "\n";
    *stream = *user_ctx->stream;
  } else {
    // printf("no user context, using default stream \n");
    *stream = 0;
  }
  return 0;
}

WEAK int halide_get_gpu_device(void *user_context) {
  if(user_context != NULL) {
    Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
    // std::cerr << "PyWrap's get gpu device " <<  user_ctx->device_id << "\n";
    return user_ctx->device_id;
  } else {
    // std::cerr << "no user context, using default device \n";
    return 0;
  }
}

}  // extern "C"

#endif  // HL_PYTORCH_WRAPPER_H
