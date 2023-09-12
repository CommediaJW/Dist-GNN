#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cache/cuda/ops.h"
#include "common/pin_memory.h"
#include "common/shared_mem.h"
#include "feature/cuda/ops.h"
#include "feature/feature_sever.h"
#include "nccl/nccl_context.h"
#include "sampling/cuda/ops.h"
#include "sampling/sampler.h"

#include "context/context.h"

using namespace dgs;
namespace py = pybind11;

PYBIND11_MODULE(dgs, m) {
  // classes
  auto m_classes = m.def_submodule("classes");

  py::class_<sampling::P2PCacheSampler>(m_classes, "P2PCacheSampler")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    int64_t>())
      .def("_CAPI_sample_node_classifiction",
           &sampling::P2PCacheSampler::NodeClassifictionSample)
      .def("_CAPI_get_cpu_structure_tensors",
           &sampling::P2PCacheSampler::GetCPUStructureTensors)
      .def("_CAPI_get_local_cache_structure_tensors",
           &sampling::P2PCacheSampler::GetLocalCachedStructureTensors)
      .def("_CAPI_get_local_cache_hashmap_tensors",
           &sampling::P2PCacheSampler::GetLocalCachedHashTensors);

  py::class_<feature::P2PCacheFeatureServer>(m_classes, "P2PCacheFeatureServer")
      .def(py::init<torch::Tensor, torch::Tensor, int64_t>())
      .def("_CAPI_get_cpu_feature",
           &feature::P2PCacheFeatureServer::GetCPUFeature)
      .def("_CAPI_get_gpu_feature",
           &feature::P2PCacheFeatureServer::GetGPUFeature)
      .def("_CAPI_get_feature", &feature::P2PCacheFeatureServer::GetFeatures);

  py::class_<cache::TensorP2PServer>(m_classes, "TensorP2PServer")
      .def(py::init<torch::Tensor>())
      .def("_CAPI_get_device_tensor", &cache::TensorP2PServer::GetDeviceTensor)
      .def("_CAPI_get_local_device_tensor",
           &cache::TensorP2PServer::GetLocalDeviceTensor);

  py::class_<SharedTensor>(m_classes, "SharedTensor")
      .def(py::init<std::vector<int64_t>, py::object>())
      .def("_CAPI_get_tensor", &SharedTensor::Tensor)
      .def("_CAPI_load_from_tensor", &SharedTensor::LoadFromTensor)
      .def("_CAPI_load_from_disk", &SharedTensor::LoadFromDisk);

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

  // cuda sampling
  m_ops
      .def("_CAPI_cuda_sample_neighbors",
           &sampling::cuda::RowWiseSamplingUniformCUDA)
      .def("_CAPI_cuda_sample_neighbors_bias",
           &sampling::cuda::RowWiseSamplingBiasCUDA)
      .def("_CAPI_cuda_sampled_tensor_relabel",
           &sampling::cuda::TensorRelabelCUDA);

  // cuda feature loading
  m_ops.def("_CAPI_cuda_index_select", &feature::cuda::GetFeaturesCUDA);

  m_ops.def("_CAPI_nccl_is_initialized", &nccl::IsInitialized);

  m_ops.def("_Test_Randn", &ctx::randn_uint64);
  m_ops.def("_Test_NCCLTensorAllGather", &nccl::NCCLTensorAllGather);
  m_ops.def("_Test_GetLocalRank", &nccl::GetLocalRank);
  m_ops.def("_Test_GetWorldSize", &nccl::GetWorldSize);
  m_ops.def("_Test_ExtractEdgeData", &sampling::cuda::ExtractEdgeData);
  m_ops.def("_Test_ExtractIndptr", &sampling::cuda::ExtractIndptr);

  m_ops.def("_CAPI_save_tensor_to_disk", &SaveTensor2Disk);
}
