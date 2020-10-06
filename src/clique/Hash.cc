#include "Hash.h"

unsigned long djb2Hash(const char* data)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *(data)++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;    
}