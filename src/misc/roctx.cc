/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "roctx.h"

std::map<uint64_t, roctxPayloadEntryType> nvtxToRoctx {
  {NVTX_PAYLOAD_ENTRY_TYPE_INT, ROCTX_PAYLOAD_ENTRY_TYPE_INT},
  {NVTX_PAYLOAD_ENTRY_TYPE_SIZE, ROCTX_PAYLOAD_ENTRY_TYPE_SIZE},
  {NVTX_PAYLOAD_ENTRY_TYPE_REDOP, ROCTX_PAYLOAD_ENTRY_TYPE_REDOP}};

const char* roctxEntryTypeStr[ROCTX_PAYLOAD_NUM_ENTRY_TYPES] = {"ROCTX_PAYLOAD_ENTRY_TYPE_INT", "ROCTX_PAYLOAD_ENTRY_TYPE_SIZE", "ROCTX_PAYLOAD_ENTRY_TYPE_REDOP"};
const char* ncclRedOpStr[ncclNumDevRedOps]                   = { "Sum", "Prod", "MinMax", "PreMulSum", "SumPostDiv" };

void roctxAlloc(roctxPayloadInfo_t payloadInfo, const size_t numEntries) {
#ifndef ROCTX_NO_IMPL
  // Allocate enough memory for numEntries in payloadEntries
  payloadInfo->payloadEntries = (roctxPayloadSchemaEntryInfo*)malloc(numEntries * sizeof(roctxPayloadSchemaEntryInfo));

  // Allocate memory for the message that will be constructed
  payloadInfo->message = (char*)malloc(MAX_MESSAGE_LENGTH * sizeof(char));
#endif
}

void roctxFree(roctxPayloadInfo_t payloadInfo) {
#ifndef ROCTX_NO_IMPL
  // Free all the dynamically allocated resources by roctx
  if (payloadInfo->payloadEntries) free(payloadInfo->payloadEntries);
  if (payloadInfo->message) free((void*)payloadInfo->message);
#endif
}

void extractPayloadInfo(const nvtxPayloadSchemaEntry_t* schema, const nvtxPayloadData_t* data, const size_t numEntries,
                        const char* schemaName, roctxPayloadInfo_t payloadInfo) {

  if (payloadInfo->payloadEntries == nullptr) return;

  payloadInfo->id = schemaName;
  payloadInfo->numEntries = numEntries;

  // Iterate over each entry in the schema
  for (size_t i = 0; i < payloadInfo->numEntries; ++i) {
    // Populate payload schema entry info for roctx
    payloadInfo->payloadEntries[i].name = schema[i].name;
    payloadInfo->payloadEntries[i].type = nvtxToRoctx[schema[i].type];

    // Offset to index into the data stored in nvtxPayloadData_t->payload
    uint64_t offset = schema[i].offset;
    const void* entryData = reinterpret_cast<const char*>(data->payload) + offset;

    // Populate payload union based on the roctx type
    switch (payloadInfo->payloadEntries[i].type) {
      case ROCTX_PAYLOAD_ENTRY_TYPE_INT: payloadInfo->payloadEntries[i].payload.typeInt = *reinterpret_cast<const int*>(entryData);                break;
      case ROCTX_PAYLOAD_ENTRY_TYPE_SIZE: payloadInfo->payloadEntries[i].payload.typeSize = *reinterpret_cast<const size_t*>(entryData);           break;
      case ROCTX_PAYLOAD_ENTRY_TYPE_REDOP: payloadInfo->payloadEntries[i].payload.typeRedOp = *reinterpret_cast<const ncclDevRedOp_t*>(entryData); break;
      default:                                                                                                                                     break;
    }
  }

  // Stringify payloadInfo
  stringify(payloadInfo);
}

void stringify(roctxPayloadInfo_t payloadInfo) {
  if (!payloadInfo->payloadEntries || !payloadInfo->message) return;

  int offset = snprintf(payloadInfo->message, MAX_MESSAGE_LENGTH, "{%s: ", payloadInfo->id);

  for (size_t i = 0; i < payloadInfo->numEntries; ++i)
  {
    roctxPayloadSchemaEntryInfo entry = payloadInfo->payloadEntries[i];

    offset += snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, "%s: ", entry.name);

    switch (entry.type)
    {
      case ROCTX_PAYLOAD_ENTRY_TYPE_INT:
        offset += snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, "%d", entry.payload.typeInt); 
        break;
      case ROCTX_PAYLOAD_ENTRY_TYPE_SIZE:
        offset += snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, "%zu", entry.payload.typeSize); 
        break;
      case ROCTX_PAYLOAD_ENTRY_TYPE_REDOP:
        offset += snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, "%s", 
                          entry.payload.typeRedOp < ncclNumDevRedOps ? ncclRedOpStr[entry.payload.typeRedOp] : "unknown"); 
        break;
      default:
        offset += snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, "unknown roctx payload type"); 
        break;
    }

    if (i != payloadInfo->numEntries - 1) 
      offset += snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, ", ");
  }

  snprintf(payloadInfo->message + offset, MAX_MESSAGE_LENGTH - offset, "}");
}