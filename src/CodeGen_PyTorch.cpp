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

CodeGen_PyTorch::CodeGen_PyTorch(ostream &s, Target t, OutputKind output_kind) :
    IRPrinter(s), target(t), output_kind(output_kind)
{
  if(is_header()) {
    // header guard
    stream << headers;
    stream << "\n#ifdef __cplusplus\n";
    stream << "extern \"C\" {\n";
    stream << "#endif\n\n";
  } else {
    // include Halide header
  }

}

CodeGen_PyTorch::~CodeGen_PyTorch() {
  if(is_header()) {
    stream << "\n#ifdef __cplusplus\n";
    stream << "}  // extern \"C\"\n";
    stream << "#endif\n\n";
  }
}

void CodeGen_PyTorch::compile(const Module &input) {
    for (const auto &f : input.functions()) {
        compile(f);
    }
}

void CodeGen_PyTorch::compile(const LoweredFunc &f) {
  // Don't put non-external function declarations in headers.
  if (is_header() && f.linkage == LoweredFunc::Internal) {
    return;
  }

  std::vector<std::string> namespaces;
  std::string simple_name = extract_namespaces(f.name, namespaces);

  if (!namespaces.empty()) {
    for (const auto &ns : namespaces) {
      stream << "namespace " << ns << " {\n";
    }
    stream << "\n";
    std::cerr << "namespace\n";
  }
  const std::vector<LoweredArgument> &args = f.args;

  stream << "int " << simple_name << "(";
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].is_buffer()) {
      stream << "struct halide_buffer_t *"
        << print_name(args[i].name)
        << "_buffer";
    } else {
      stream <<  "type "
        // print_type(args[i].type, AppendSpace)
        << print_name(args[i].name);
    }

    if (i < args.size()-1) stream << ", ";
  }

  if (is_header()) {
    stream << ");\n";
  } else {
    stream << ") {\n";
    indent += 1;
    do_indent();
    // Check ndimensions
    // Grab continuous buffer references
    // Resize output? done in the python interface now
    stream << "return 0;\n";
    indent -= 1;
    stream << "}\n";
  }

  if (!namespaces.empty()) {
    stream << "\n";
    for (size_t i = namespaces.size(); i > 0; i--) {
      stream << "}  // namespace " << namespaces[i-1] << "\n";
    }
    stream << "\n";
  }
}

string CodeGen_PyTorch::print_name(const string &name) {
    ostringstream oss;

    // Prefix an underscore to avoid reserved words (e.g. a variable named "while")
    if (isalpha(name[0])) {
        oss << '_';
    }

    for (size_t i = 0; i < name.size(); i++) {
        if (name[i] == '.') {
            oss << '_';
        } else if (name[i] == '$') {
            oss << "__";
        } else if (name[i] != '_' && !isalnum(name[i])) {
            oss << "___";
        }
        else oss << name[i];
    }
    return oss.str();
}

}
}
