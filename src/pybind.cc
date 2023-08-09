#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "nccl/nccl_context.h"
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
           &sampling::P2PCacheSampler::NodeClassifictionSample)
      .def("_CAPI_get_cpu_structure_tensors",
           &sampling::P2PCacheSampler::GetCPUStructureTensors)
      .def("_CAPI_get_cpu_hashmap_tensors",
           &sampling::P2PCacheSampler::GetCPUHashTensors)
      .def("_CAPI_get_local_cache_structure_tensors",
           &sampling::P2PCacheSampler::GetLocalCachedStructureTensors)
      .def("_CAPI_get_local_cache_hashmap_tensors",
           &sampling::P2PCacheSampler::GetLocalCachedHashTensors);

  // ops
  auto m_ops = m.def_submodule("ops");
  // nccl communicating
  m_ops.def("_CAPI_get_unique_id", &nccl::GetUniqueId)
      .def("_CAPI_set_nccl", &nccl::SetNCCL);
}
