#ifndef HL_PYTORCH_WRAPPER_H
#define HL_PYTORCH_WRAPPER_H

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <exception>

#include <torch/extension.h>

#include <HalideBuffer.h>

#define WEAK __attribute__((weak))

using Halide::Runtime::Buffer;


namespace Halide {
namespace Pytorch {

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
#ifdef HL_PT_CUDA
    // TODO(mgharbi): device interface no
    // const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();
    // int err = buffer.device_wrap_native(cuda_interface, (uint64_t)pData);
    // if (err != 0) {
    //   throw "halide_device_wrap failed";
    // }
    // buffer.set_device_dirty();
#else
    std::cout << "Cuda was not available at compile time for the HL wrap.\n";
#endif
  } else {
    buffer = Buffer<scalar_t>(pData, dims);
  }

  return buffer;
}

} // namespace Pytorch
} // namespace Halide

#endif  // HL_PYTORCH_WRAPPER_H
