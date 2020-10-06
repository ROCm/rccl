#include "HandleCache.h"

#include "Hash.h"

// djb2 hash function for hashing char array in hipIpcMemHandle_t
unsigned long hipIpcMemHandleHash(const hipIpcMemHandle_t& handle)
{
    return djb2Hash(handle.reserved);
}