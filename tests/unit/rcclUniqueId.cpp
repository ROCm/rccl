#include <rccl/rccl.h>
#include <limits.h>
#include "gtest/gtest.h"

TEST(UniqueIdTest, Zero) {
    rcclUniqueId id1;
    EXPECT_EQ(rcclSuccess, rcclGetUniqueId(&id1));
    EXPECT_EQ(rcclInvalidArgument, rcclGetUniqueId(0));
}
