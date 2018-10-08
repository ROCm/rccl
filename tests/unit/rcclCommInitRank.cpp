#include <rccl/rccl.h>
#include "gtest/gtest.h"

TEST(CommInitRankTest, T01) {
    EXPECT_EQ(rcclInvalidArgument, rcclCommInitRank(nullptr, 0, nullptr, 0));
}
TEST(CommInitRankTest, T02) {
    rcclComm_t comm;
    EXPECT_EQ(rcclInvalidRank, rcclCommInitRank(&comm, 1, nullptr, 1));
}
TEST(CommInitRankTest, T03) {
    rcclComm_t comm;
    EXPECT_EQ(rcclInvalidArgument, rcclCommInitRank(&comm, 1, nullptr, 0));
}
TEST(CommInitRankTest, T04) {
    rcclComm_t comm;
    EXPECT_EQ(rcclUnsupportedDeviceCount, rcclCommInitRank(&comm, 0, nullptr, -1));
}
TEST(CommInitRankTest, T05) {
    rcclComm_t comm;
    EXPECT_EQ(rcclInvalidArgument, rcclCommInitRank(&comm, 1, nullptr, 0));
}
TEST(CommInitRankTest, T06) {
    rcclComm_t comm;
    rcclUniqueId id;
    EXPECT_EQ(rcclSuccess, rcclGetUniqueId(&id));
    EXPECT_EQ(rcclSuccess, rcclCommInitRank(&comm, 1, id, 0));
    EXPECT_EQ(rcclUnsupportedDeviceCount, rcclCommInitRank(&comm, 2, id, 1));
}
TEST(CommInitRankTest, T07) {
    rcclUniqueId id;
    rcclComm_t comm;
    EXPECT_EQ(rcclSuccess, rcclGetUniqueId(&id));
    EXPECT_EQ(rcclSuccess, rcclCommInitRank(&comm, 1, id, 0));
}
