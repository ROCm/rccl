/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCL_PARSER_H_
#define MSCCL_PARSER_H_

#include <cstdio>
#include <cstdlib>
#include <string>

#include "rccl/rccl.h"
#include "msccl/msccl_scheduler.h"

// A few constraints to make the implementation easy
#define MAX_STR_LEN 255
#define MAX_ATTR_COUNT 16
#define MAX_NODES 4096

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

struct mscclXmlNode {
  char name[MAX_STR_LEN+1];
  struct {
    char key[MAX_STR_LEN+1];
    char value[MAX_STR_LEN+1];
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
};

struct mscclAlgoMeta {
  // Path to algorithm file
  std::string filePath;
  // number of chunks of input/output in each MSCCL algorithm loop
  int nChunksPerLoop;
  // number of ranks required by this algorithm
  int nRanks;
  // MSCCL function type
  mscclFunc_t func;
  // need to times nRanks for all-gather, reduce-scatter and all-to-all
  int sizeMultiplier;
  // Min message size allowed for this algorithm.
  int64_t minBytes;
  // Max message size allowed for this algorithm, 0 for no limit.
  int64_t maxBytes;
  // Whether this algorithm is suitable for in-place.
  bool inPlace;
  // Whether this algorithm is suitable for out-of-place.
  bool outOfPlace;
};

static ncclResult_t mscclXmlGetAttrIndex(struct mscclXmlNode* node, const char* attrName, int* index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a=0; a<nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t mscclXmlGetAttr(struct mscclXmlNode* node, const char* attrName, const char** value) {
  ncclResult_t ret = ncclSuccess;
  int index;
  ret = mscclXmlGetAttrIndex(node, attrName, &index);
  if (ret != ncclSuccess) {
    return ret;
  }
  *value = index == -1 ? NULL : node->attrs[index].value;
  return ncclSuccess;
}

static ncclResult_t mscclXmlGetAttrStr(struct mscclXmlNode* node, const char* attrName, const char** value) {
  ncclResult_t ret = ncclSuccess;
  ret = mscclXmlGetAttr(node, attrName, value);
  if (ret != ncclSuccess) {
    return ret;
  }
  if (*value == NULL) {
    fprintf(stderr, "Attribute %s of node %s not found", attrName, node->name);
    return ncclInternalError;
  }
  return ncclSuccess;
}
static ncclResult_t mscclXmlGetAttrInt(struct mscclXmlNode* node, const char* attrName, int* value) {
  ncclResult_t ret = ncclSuccess;
  const char* str;
  ret = mscclXmlGetAttrStr(node, attrName, &str);
  if (ret != ncclSuccess) {
    return ret;
  }
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t mscclXmlGetAttrInt64(struct mscclXmlNode* node, const char* attrName, int64_t* value) {
  ncclResult_t ret = ncclSuccess;
  const char* str;
  ret = mscclXmlGetAttrStr(node, attrName, &str);
  if (ret != ncclSuccess) {
    return ret;
  }
  *value = strtoll(str, NULL, 0);
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoMetaFromXmlFile(const char* xmlGraphFile, struct mscclAlgoMeta* algoMeta);

#endif
