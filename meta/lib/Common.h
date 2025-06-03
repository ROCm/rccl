// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/logging/xlog.h>

inline void META_INTERNAL_INIT() {
#if defined(ENABLE_META_PROXY_TRACE) || defined(ENABLE_META_COLLTRACE)
  folly::LoggerDB::get().setLevel("", folly::LogLevel::INFO);
#endif
}
