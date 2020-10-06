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
    typedef std::unordered_map<Key, std::pair<Value, typename std::list<Key>::iterator>, Hash, KeyEqual, Allocator> LRUCache;    
public:
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
            updateHistory(key);
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

    void updateHistory(const Key& key)
    {
        if (m_lruHistory.size() > 0)
        {
            m_lruHistory.splice(m_lruHistory.end(), m_lruHistory, m_cache[key].second);
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

typedef NcclIpcHandleCache<uint64_t, hipIpcMemHandle_t, std::hash<uint64_t>, std::equal_to<uint64_t>, std::allocator< std::pair<const uint64_t, hipIpcMemHandle_t>>> NcclIpcHandleSendCache;
typedef NcclIpcHandleCache<hipIpcMemHandle_t, void*, decltype(&hipIpcMemHandleHash), decltype(hipIpcMemHandleEqual), std::allocator< std::pair<const uint64_t, hipIpcMemHandle_t>>> NcclIpcHandleRecvCache;

#endif
