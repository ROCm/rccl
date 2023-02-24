/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include "parser.h"

ncclResult_t mscclXmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    fprintf(stderr, "XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetValue(FILE* file, char* value, char* last) {
  ncclResult_t ret = ncclSuccess;
  char c;
  ret = mscclXmlGetChar(file, &c);
  if (ret != ncclSuccess) {
    return ret;
  }
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      ret = mscclXmlGetChar(file, &c);
      if (ret != ncclSuccess) {
        return ret;
      }
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    fprintf(stderr, "XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    ret = mscclXmlGetChar(file, &c);
    if (ret != ncclSuccess) {
      return ret;
    }
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  ret = mscclXmlGetChar(file, last);
  if (ret != ncclSuccess) {
    return ret;
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetToken(FILE* file, char* name, char* value, char* last) {
  ncclResult_t ret = ncclSuccess;
  char c;
  char* ptr = name;
  int o = 0;
  do {
    ret = mscclXmlGetChar(file, &c);
    if (ret != ncclSuccess) {
      return ret;
    }
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        fprintf(stderr, "XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return mscclXmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      fprintf(stderr, "Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t mscclXmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      fprintf(stderr, "XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetNode(FILE* file, struct mscclXmlNode* node) {
  ncclResult_t ret = ncclSuccess;
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    fprintf(stderr, "XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  ret = mscclXmlGetToken(file, node->name, NULL, &c);
  if (ret != ncclSuccess) {
    return ret;
  }

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    ret = mscclXmlSkipComment(file, node->name+3, c);
    if (ret != ncclSuccess) {
      return ret;
    }
    return mscclXmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    ret = mscclXmlGetToken(file, node->name, NULL, &c);
    if (ret != ncclSuccess) {
      return ret;
    }
    if (c != '>') {
      fprintf(stderr, "XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    ret = mscclXmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c);
    if (ret != ncclSuccess) {
      return ret;
    }
    if (a == MAX_ATTR_COUNT) {
      fprintf(stdout, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    ret = mscclXmlGetToken(file, str, NULL, &c);
    if (ret != ncclSuccess) {
      return ret;
    }
  }
  if (c != '>') {
    fprintf(stderr, "XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlLoadSingleNode(FILE* file, struct mscclXmlNode* node) {
  memset(node, 0, sizeof(struct mscclXmlNode));
  return mscclXmlGetNode(file, node);
}

ncclResult_t mscclAlgoMetaXmlLoad(const char* xmlFilePath, struct mscclXmlNode* node) {
  ncclResult_t ret = ncclSuccess;
  FILE* file = fopen(xmlFilePath, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open MSCCL XML algorithm file %s : %s", xmlFilePath, strerror(errno));
    return ncclSystemError;
  }
  ret = mscclXmlLoadSingleNode(file, node);
  if (ret != ncclSuccess) {
    return ret;
  }
  fclose(file);
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoMetaFromXmlFile(const char* str, struct mscclAlgoMeta* algoMeta) {
  ncclResult_t ret = ncclSuccess;
  struct mscclXmlNode* node;
  node = (struct mscclXmlNode *)malloc(sizeof(struct mscclXmlNode));
  ret = mscclAlgoMetaXmlLoad(str, node);
  if (ret != ncclSuccess) {
    return ret;
  }

  algoMeta->filePath = str;

  int nChunksPerLoop;
  ret = mscclXmlGetAttrInt(node, "nchunksperloop", &nChunksPerLoop);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->nChunksPerLoop  = nChunksPerLoop;

  int nGpus;
  ret = mscclXmlGetAttrInt(node, "ngpus", &nGpus);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->nRanks = nGpus;

  const char* coll;
  ret = mscclXmlGetAttrStr(node, "coll", &coll);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->sizeMultiplier = 1;
  if (strcmp(coll, "reduce") == 0) {
    algoMeta->func = mscclFuncReduce;
  } else if (strcmp(coll, "broadcast") == 0) {
    algoMeta->func = mscclFuncBroadcast;
  } else if (strcmp(coll, "allreduce") == 0) {
    algoMeta->func = mscclFuncAllReduce;
  } else if (strcmp(coll, "reducescatter") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncReduceScatter;
  } else if (strcmp(coll, "allgather") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncAllGather;
  } else if (strcmp(coll, "send") == 0) {
    algoMeta->func = mscclFuncSend;
  } else if (strcmp(coll, "recv") == 0) {
    algoMeta->func = mscclFuncRecv;
  } else if (strcmp(coll, "gather") == 0) {
    algoMeta->func = mscclFuncGather;
  } else if (strcmp(coll, "scatter") == 0) {
    algoMeta->func = mscclFuncScatter;
  } else if (strcmp(coll, "alltoall") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncAllToAll;
  } else if (strcmp(coll, "alltoallv") == 0) {
    algoMeta->func = mscclFuncAllToAllv;
  } else {
    return ncclInvalidUsage;
  }

  int64_t minBytes;
  ret = mscclXmlGetAttrInt64(node, "minBytes", &minBytes);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->minBytes = minBytes;

  int64_t maxBytes;
  ret = mscclXmlGetAttrInt64(node, "maxBytes", &maxBytes);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->maxBytes = maxBytes;

  int inplace;
  ret = mscclXmlGetAttrInt(node, "inplace", &inplace);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->inPlace = (bool)inplace;

  int outofplace;
  ret = mscclXmlGetAttrInt(node, "outofplace", &outofplace);
  if (ret != ncclSuccess) {
    return ret;
  }
  algoMeta->outOfPlace = (bool)outofplace;

  free(node);
  return ncclSuccess;
}
