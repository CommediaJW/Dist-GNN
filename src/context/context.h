#ifndef DGS_CONTEXT_H_
#define DGS_CONTEXT_H_

#include <torch/script.h>
#include <random>

class RandomEngine {
 public:
  RandomEngine() : gen_(std::random_device()()) {
    dis_ = std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFF);
  }
  RandomEngine(const RandomEngine&) = delete;
  RandomEngine& operator=(const RandomEngine&) = delete;

  uint64_t randn() { return dis_(gen_); }

 private:
  std::mt19937_64 gen_;
  std::uniform_int_distribution<uint64_t> dis_;
};

namespace dgs {
namespace ctx {

static RandomEngine random_engine;
uint64_t randn_uint64();

}  // namespace ctx
}  // namespace dgs

#endif
