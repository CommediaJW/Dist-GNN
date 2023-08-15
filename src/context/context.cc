#include "context.h"

namespace dgs {
namespace ctx {
RandomEngine random_engine;
uint64_t randn_uint64() { return random_engine.randn(); }
}  // namespace ctx
}  // namespace dgs