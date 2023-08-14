#include "context.h"

namespace dgs {
namespace ctx {
uint64_t randn_uint64() { return random_engine.randn(); }
}  // namespace ctx
}  // namespace dgs