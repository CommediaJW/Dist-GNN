#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "sampling/sampler.h"

using namespace dgs;
namespace py = pybind11;

PYBIND11_MODULE(dgs, m) {
  // classes
  auto m_classes = m.def_submodule("classes");
  // tensor p2p cache manager
  py::class_<sampling::P2PCacheSampler>(m_classes, "P2PCacheSampler")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, int64_t>())
      .def("_CAPI_sample_node_classifiction",
           &sampling::P2PCacheSampler::NodeClassifictionSample);
}
