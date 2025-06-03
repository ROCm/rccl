// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gtest/gtest.h>
#include "comms/testinfra/MpiTestUtils.h"
#include "comms/utils/checks.h"
#include "rccl.h" // @manual

namespace meta::rcclx {

//
// Generic RCCLX test fixture, internally uses MPI to initialize
// globalRank, localRank, numRanks and commId
//
// example usage:
//
// TEST_F(RcclxBaseTestFixture, YourTestCase) {
//   XLOG(INFO) << fmt::format("globalRank: {}, localRank: {}, numRanks: {}",
//   globalRank, localRank, numRanks);
// }
//
// see HelloWorldTest.cu as a concrete example
//
class RcclxBaseTestFixture : public meta::comms::MpiBaseTestFixture {
 public:
 protected:
  void SetUp() override;

  void TearDown() override;

  ncclUniqueId commId;
};

} // namespace meta::rcclx
