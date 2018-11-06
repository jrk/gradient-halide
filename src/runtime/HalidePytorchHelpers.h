#ifndef HL_PYTORCH_WRAPPER_H
#define HL_PYTORCH_WRAPPER_H

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <exception>

#include <torch/extension.h>

#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>

// TODO: if cuda


// #include <ATen/Config.h>
// #if AT_CUDA_ENABLED()
#include <cuda.h>
#include <cuda_runtime.h>
// #endif

#define WEAK __attribute__((weak))

// extern THCState *state;
#define HLPT_CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define HLPT_CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define HLPT_CHECK_DEVICE(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

using Halide::Runtime::Buffer;

namespace Halide {
namespace Pytorch {

// struct DeviceNotSynchronizedException : public std::exception {
//   std::string buffer_name;
//   DeviceNotSynchronizedException(std::string buffer_name)
//     : buffer_name(buffer_name) { }
//   const char* what() const throw() {
//     std::stringstream buf;
//     buf << "Halide output buffer "
//         << buffer_name
//         << " is on CPU, please compute it on GPU.";
//     return buf.str().c_str();
//   }
// };
//
// struct InvalidDeviceException : public std::exception {
//   const char* what() const throw() {
//     return "Halide operators attempts to access a buffer on the wrong device";
//   }
// };
//
// struct CudaContextException : public std::exception {
//   const char* what() const throw() {
//     return "Could not acquire CUDA context.";
//   }
// };
//
// struct CudaRunException : public std::exception {
//   const char* what() const throw() {
//     return "Could not run Halide CUDA op.";
//   }
// };

inline std::vector<int> getDims(const at::Tensor tensor) {
  int ndims = tensor.ndimension();
  std::vector<int> dims(ndims, 0);
  // PyTorch dim order is reverse of Halide
  for(int dim = 0; dim < ndims; ++dim) {
    dims[dim] = tensor.size(ndims-1-dim);
  }
  return dims;
}


template<class scalar_t>
inline void check_type(at::Tensor &tensor) {
  AT_ERROR("Scalar type ", tensor.scalar_type(), " not handled by Halide's Pytorch wrapper");
}


#define HL_PT_DEFINE_TYPECHECK(ctype,ttype,_3) \
  template<> \
  inline void check_type<ctype>(at::Tensor &tensor) { \
    AT_ASSERTM(tensor.scalar_type() == at::ScalarType::ttype, "scalar type do not match"); \
  }
  AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(HL_PT_DEFINE_TYPECHECK)
#undef HL_PT_DEFINE_TYPECHECK


  // TODO(mgharbi): const data in the buffer
template<class scalar_t>
inline Buffer<scalar_t> wrap(at::Tensor &tensor) {
  check_type<scalar_t>(tensor);
  std::vector<int> dims = getDims(tensor);
  scalar_t* pData  = tensor.data<scalar_t>();
  Buffer<scalar_t> buffer;

  // TODO(mgharbi): how to force Halide to put input/output on GPU?
  if(tensor.is_cuda()) {
    std::cout << "cuda device\n";
    buffer = Buffer<scalar_t>(dims);
    // // TODO: device interface no
    const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();
    int err = buffer.device_wrap_native(cuda_interface, (uint64_t)pData);
    if (err != 0) {
      throw "halide_device_wrap failed";
    }
    buffer.set_device_dirty();
  } else {
    buffer = Buffer<scalar_t>(pData, dims);
  }

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

CUcontext WEAK context = NULL;

WEAK int halide_cuda_acquire_context(void *user_context, CUcontext *ctx, bool create = true) {
  // halide_assert(user_context, ctx != NULL);
  if(user_context != NULL) {
    Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
    std::cerr << "PyWrap acquire user ctx " << user_ctx << "\n";
    *ctx = *user_ctx->cuda_context;
  } else {
    std::cerr << "no user context, in halide acquire, cuda ctx is " << ctx << "\n";
    *ctx = NULL;
  }
  return 0;
}

WEAK int halide_cuda_get_stream(void *user_context, CUcontext ctx, CUstream *stream) {
  if(user_context != NULL) {
    Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
    std::cerr << "PyWrap's get stream " <<  *user_ctx->stream << "\n";
    *stream = *user_ctx->stream;
  } else {
    std::cerr << "no user context, using default stream\n";
    *stream = 0;
  }
  return 0;
}

WEAK int halide_get_gpu_device(void *user_context) {
  if(user_context != NULL) {
    Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
    std::cerr << "PyWrap's get gpu device " <<  user_ctx->device_id << "\n";
    return user_ctx->device_id;
  } else {
    std::cerr << "no user context, using default device \n";
    return 0;
  }
}

}  // extern "C"

#endif  // HL_PYTORCH_WRAPPER_H
