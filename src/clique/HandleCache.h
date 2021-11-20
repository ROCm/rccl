/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef NCCL_HANDLE_CACHE_H_
#define NCCL_HANDLE_CACHE_H_

#include <list>
#include <unordered_map>
#include <functional>

#include "core.h"

//#include "llvm/ADT/DenseMap.h"

template <
    class Key,
    class Value,
    class Hash,
    class KeyEqual,
    class Allocator
>
class NcclIpcHandleCache
{
public:
    typedef std::pair<Value, typename std::list<Key>::iterator> NcclIpcHandleCacheValueType;
    typedef std::unordered_map<Key, NcclIpcHandleCacheValueType, Hash, KeyEqual, Allocator> LRUCache;
    using iterator = typename LRUCache::iterator;
    NcclIpcHandleCache(size_t size,
                       size_t bucket_count = 100,
                       const Hash& hash = Hash(),
                       const KeyEqual& eql = KeyEqual(),
                       const Allocator& alloc = Allocator() ) : m_cache(bucket_count, hash, eql, alloc)
    {
        m_capacity = size;
    }

    ~NcclIpcHandleCache()
    {
        m_lruHistory.clear();
        m_cache.clear();
    }

    iterator begin()
    {
        return m_cache.begin();
    }

    iterator end()
    {
        return m_cache.end();
    }

    iterator find(const Key& key)
    {
        iterator it = m_cache.find(key);
        if (it != m_cache.end())
        {
            updateHistory(it);
        }

        return it;
    }

    std::pair<iterator, bool> insert(const Key& key, const Value& value)
    {
        if (m_cache.size() == m_capacity)
        {
            // remove entry
            pop();
        }

        typename LRUCache::iterator it = m_cache.find(key);
        bool inserted;
        if (it == m_cache.end())
        {
            typename std::list<Key>::iterator it = m_lruHistory.insert(m_lruHistory.end(), key);
            m_cache.insert(std::make_pair(key, std::make_pair(value, it)));
            inserted = true;
        }
        else
        {
            inserted = false;
        }

        return std::pair<iterator, bool>(it, inserted);
    }

private:
    void pop()
    {
        typename LRUCache::iterator it = m_cache.find(m_lruHistory.front());
        m_cache.erase(it);
        m_lruHistory.pop_front();
    }

    void updateHistory(const iterator& it)
    {
        if (!m_lruHistory.empty())
        {
            m_lruHistory.splice(m_lruHistory.end(), m_lruHistory, (it->second).second);
        }
    }
    size_t m_capacity;
    std::list<Key> m_lruHistory;
    LRUCache m_cache;
};

// djb2 hash function for hashing char array in hipIpcMemHandle_t
unsigned long hipIpcMemHandleHash(const hipIpcMemHandle_t& handle);

// equality function required for unordered_map
auto hipIpcMemHandleEqual = [](const hipIpcMemHandle_t& l, const hipIpcMemHandle_t& r)
{
    return memcmp(l.reserved, r.reserved, sizeof(l.reserved)) == 0;
};

//typedef llvm::DenseMap<uint64_t, hipIpcMemHandle_t> SendCache;
//typedef llvm::DenseMap<hipIpcMemHandle_t, void*, decltype(&HandleHash), decltype(HandleEqual)> RecvCache;

typedef NcclIpcHandleCache<uint64_t, hipIpcMemHandle_t, std::hash<uint64_t>, std::equal_to<uint64_t>, std::allocator< std::pair<const uint64_t, std::pair<hipIpcMemHandle_t, std::list<uint64_t>::iterator>>>> NcclIpcHandleSendCache;
typedef NcclIpcHandleCache<hipIpcMemHandle_t, void*, decltype(&hipIpcMemHandleHash), decltype(hipIpcMemHandleEqual), std::allocator< std::pair<const hipIpcMemHandle_t, std::pair<void*, std::list<hipIpcMemHandle_t>::iterator>>>> NcclIpcHandleRecvCache;

#endif
