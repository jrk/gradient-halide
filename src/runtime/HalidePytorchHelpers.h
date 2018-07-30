#ifndef HL_PYTORCH_WRAPPER_H
#define HL_PYTORCH_WRAPPER_H

#include "HalideBuffer.h"
#include "HalideRuntimeCuda.h"
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <exception>

#include "TH/TH.h"
#include "THC/THC.h"

#define WEAK __attribute__((weak))

extern THCState *state;

using Halide::Runtime::Buffer;

namespace Halide {
namespace Pytorch {

struct DeviceNotSynchronizedException : public std::exception {
  std::string buffer_name;
  DeviceNotSynchronizedException(std::string buffer_name)
    : buffer_name(buffer_name) { }
  const char* what() const throw() {
    std::stringstream buf;
    buf << "Halide output buffer "
        << buffer_name
        << " is on CPU, please compute it on GPU.";
    return buf.str().c_str();
  }
};

struct InvalidDeviceException : public std::exception {
  const char* what() const throw() {
    return "Halide operators attempts to access a buffer on the wrong device";
  }
};

struct CudaContextException : public std::exception {
  const char* what() const throw() {
    return "Could not acquire CUDA context.";
  }
};

struct CudaRunException : public std::exception {
  const char* what() const throw() {
    return "Could not run Halide CUDA op.";
  }
};

template <typename T>
inline int get_ndims(const THTensor* tensor);

template <typename T>
inline int get_size(const THTensor* tensor, int idx);

template <typename T>
inline T* get_torch_data(const THTensor* tensor);

template <typename T>
inline int get_ndims(THCState* state, const THCTensor* tensor);

template <typename T>
inline int get_size(THCState* state, const THCTensor* tensor, int idx);

template <typename T>
inline T* get_torch_data(THCState* state, const THCTensor* tensor);

// Tensor API specializations ---------------------
template <>
inline int get_ndims<float>(const THTensor* tensor) { return THFloatTensor_nDimension(tensor); };

template <>
inline int get_ndims<double>(const THTensor* tensor) { return THDoubleTensor_nDimension(tensor); };

template <>
inline int get_ndims<int32_t>(const THTensor* tensor) { return THIntTensor_nDimension(tensor); };

template <>
inline int get_ndims<int64_t>(const THTensor* tensor) { return THLongTensor_nDimension(tensor); };

template <>
inline int get_ndims<float>(THCState* state, const THCTensor* tensor) { return THCudaTensor_nDimension(state, tensor); };

template <>
inline int get_ndims<double>(THCState* state, const THCTensor* tensor) { return THCudaDoubleTensor_nDimension(state, tensor); };

template <>
inline int get_ndims<int32_t>(THCState* state, const THCTensor* tensor) { return THCudaIntTensor_nDimension(state, tensor); };

template <>
inline int get_ndims<int64_t>(THCState* state, const THCTensor* tensor) { return THCudaDoubleTensor_nDimension(state, tensor); };

template <>
inline int get_size<float>(const THTensor* tensor, int idx) { return THFloatTensor_size(tensor, idx); };

template <>
inline int get_size<double>(const THTensor* tensor, int idx) { return THDoubleTensor_size(tensor, idx); };

template <>
inline int get_size<int32_t>(const THTensor* tensor, int idx) { return THIntTensor_size(tensor, idx); };

template <>
inline int get_size<int64_t>(const THTensor* tensor, int idx) { return THLongTensor_size(tensor, idx); };

template <>
inline int get_size<float>(THCState* state, const THCTensor* tensor, int idx) { return THCudaTensor_size(state, tensor, idx); };

template <>
inline int get_size<double>(THCState* state, const THCTensor* tensor, int idx) { return THCudaDoubleTensor_size(state, tensor, idx); };

template <>
inline int get_size<int32_t>(THCState* state, const THCTensor* tensor, int idx) { return THCudaIntTensor_size(state, tensor, idx); };

template <>
inline int get_size<int64_t>(THCState* state, const THCTensor* tensor, int idx) { return THCudaLongTensor_size(state, tensor, idx); };

template <>
inline float* get_torch_data<float>(const THTensor* tensor) { return THFloatTensor_data(tensor); };

template <>
inline double* get_torch_data<double>(const THTensor* tensor) { return THDoubleTensor_data(tensor); };

template <>
inline int32_t* get_torch_data<int32_t>(const THTensor* tensor) { return THIntTensor_data(tensor); };

template <>
inline int64_t* get_torch_data<int64_t>(const THTensor* tensor) { return THLongTensor_data(tensor); };

template <>
inline float* get_torch_data<float>(THCState* state, const THCTensor* tensor) { return THCudaTensor_data(state, tensor); };

template <>
inline double* get_torch_data<double>(THCState* state, const THCTensor* tensor) { return THCudaDoubleTensor_data(state, tensor); };

template <>
inline int32_t* get_torch_data<int32_t>(THCState* state, const THCTensor* tensor) { return THCudaIntTensor_data(state, tensor); };

template <>
inline int64_t* get_torch_data<int64_t>(THCState* state, const THCTensor* tensor) { return THCudaLongTensor_data(state, tensor); };
// End Tensor API specializations -----------------


template <typename T>
inline Buffer<T> wrap(THTensor* tensor) {
  int ndims = get_ndims<T>(tensor);
  std::vector<int> dims(ndims, 0);
  for(int dim = 0; dim < ndims; ++dim) {
    dims[dim] = get_size<T>(tensor, ndims-1-dim);
  }
  T* pData  = get_torch_data<T>(tensor);
  Buffer<T> buffer(pData, dims);
  return buffer;
}

template <typename T>
inline Buffer<T> wrap(THCudaTensor* tensor) {
  const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();

  int ndims = get_ndims<T>(state, tensor);
  std::vector<int> dims(ndims, 0);
  for(int dim = 0; dim < ndims; ++dim) {
    dims[dim] = get_size<T>(state, tensor, ndims-1-dim);
  }

  Buffer<T> buffer(dims);

  T* pData  = get_torch_data<T>(state, tensor);
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
