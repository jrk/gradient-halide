#include <iostream>
#include <limits>

#include "CodeGen_PyTorch.h"
#include "CodeGen_Internal.h"
#include "Substitute.h"
#include "IROperator.h"
#include "Param.h"
#include "Var.h"
#include "Lerp.h"
#include "Simplify.h"
#include "Deinterleave.h"

namespace Halide {
namespace Internal {

using std::ostream;
using std::endl;
using std::string;
using std::vector;
using std::ostringstream;
using std::map;

namespace {

const string headers =
    "#include <TH/TH.h>\n"
    "#include <stdio.h>\n"
    "#include <HalideBuffer.h>\n"
    "\n"
    "using Halide::Runtime::Buffer;\n"
    ;
}

CodeGen_PyTorch::CodeGen_PyTorch(ostream &s, Target t) :
    IRPrinter(s), target(t) 
{
  stream << headers;
  stream << "\n#ifdef __cplusplus\n";
  stream << "extern \"C\" {\n";
  stream << "#endif\n\n";

  // Check ndimensions
  // Grab continuous buffer references
  // Resize output? done in the python interface now

  stream << "\n#ifdef __cplusplus\n";
  stream << "}  // extern \"C\"\n";
  stream << "#endif\n\n";
}

CodeGen_PyTorch::~CodeGen_PyTorch() {
}

void CodeGen_PyTorch::compile(const Module &input) {
    for (const auto &f : input.functions()) {
        std::cerr << f.name << "\n";
        if (f.body.defined()) {
          std::cerr << "body defined\n";
        }
    }
}

}
}
