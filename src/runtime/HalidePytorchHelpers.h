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

#define WEAK __attribute__((weak))

// extern THCState *state;

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
    std::cout << "  size " << dims[dim] << "\n";
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
    std::cout << "tensor has proper type " << tensor.scalar_type() << "\n"; \
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

  if(tensor.is_cuda()) {
    buffer = Buffer<scalar_t>(dims);
    // TODO: device interface no
    // const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();
    // int err = buffer.device_wrap_native(cuda_interface, (uint64_t)pData);
    // if (err != 0) {
    //   throw "halide_device_wrap failed";
    // }
    // buffer.set_device_dirty();
    std::cout << "cuda not implemented in HL wrap\n";
  } else {
    buffer = Buffer<scalar_t>(pData, dims);
  }

  return buffer;
}

// template <typename T>
// inline Buffer<T> wrap(THCudaTensor* tensor) {
//   const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();
//
//   int ndims = get_ndims<T>(state, tensor);
//   std::vector<int> dims(ndims, 0);
//   for(int dim = 0; dim < ndims; ++dim) {
//     dims[dim] = get_size<T>(state, tensor, ndims-1-dim);
//   }
//
//   Buffer<T> buffer(dims);
//
//   T* pData  = get_torch_data<T>(state, tensor);
//   int err = buffer.device_wrap_native(cuda_interface, (uint64_t)pData);
//   if (err != 0) {
//     throw "halide_device_wrap failed";
//   }
//   buffer.set_device_dirty();
//
//   return buffer;
// }

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

#include <cuda.h>
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

// WEAK int halide_cuda_get_stream(void *user_context, CUcontext ctx, CUstream *stream) {
//   if(user_context != NULL) {
//     Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
//     // std::cerr << "PyWrap's get stream " <<  *user_ctx->stream << "\n";
//     *stream = *user_ctx->stream;
//   } else {
//     // printf("no user context, using default stream \n");
//     *stream = 0;
//   }
//   return 0;
// }
//
// WEAK int halide_get_gpu_device(void *user_context) {
//   if(user_context != NULL) {
//     Halide::Pytorch::UserContext *user_ctx = (Halide::Pytorch::UserContext*) user_context;
//     // std::cerr << "PyWrap's get gpu device " <<  user_ctx->device_id << "\n";
//     return user_ctx->device_id;
//   } else {
//     // std::cerr << "no user context, using default device \n";
//     return 0;
//   }
// }

}  // extern "C"

#endif  // HL_PYTORCH_WRAPPER_H
