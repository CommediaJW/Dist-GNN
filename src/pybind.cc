#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cache/cuda/ops.h"
#include "common/pin_memory.h"
#include "feature/cuda/ops.h"
#include "nccl/nccl_context.h"
#include "sampling/cuda/ops.h"
#include "sampling/sampler.h"

#include "context/context.h"

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
  // cache preprocess ops
  m_ops.def("_CAPI_compute_frontier_heat", &cache::cuda::ComputeFrontierHeat)
      .def("_CAPI_compute_frontier_heat_with_bias",
           &cache::cuda::ComputeFrontierHeatWithBias);
  // tensor pin memory
  m_ops.def("_CAPI_tensor_pin_memory", &TensorPinMemory)
      .def("_CAPI_tensor_unpin_memory", &TensorUnpinMemory);
  // cuda tensor index
  m_ops.def("_CAPI_cuda_index", &feature::cuda::IndexCUDA);
  // cuda sampling
  m_ops
      .def("_CAPI_cuda_sample_neighbors",
           &sampling::cuda::RowWiseSamplingUniformCUDA)
      .def("_CAPI_cuda_sample_neighbors_bias",
           &sampling::cuda::RowWiseSamplingBiasCUDA)
      .def("_CAPI_cuda_sampled_tensor_relabel",
           &sampling::cuda::TensorRelabelCUDA);

  m_ops.def("_Test_Randn", &ctx::randn_uint64);
  m_ops.def("_Test_NCCLTensorAllGather", &nccl::NCCLTensorAllGather);
  m_ops.def("_Test_GetLocalRank", &nccl::GetLocalRank);
  m_ops.def("_Test_GetWorldSize", &nccl::GetWorldSize);
}
