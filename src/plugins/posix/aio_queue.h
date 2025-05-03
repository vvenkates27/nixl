#ifndef AIO_QUEUE_H
#define AIO_QUEUE_H

#include <vector>
#include <aio.h>
#include "async_queue.h"
#include "common/status.h"

// Forward declare Error class
class nixlPosixBackendReqH;

class aioQueue : public nixlPosixQueue {
private:
    std::vector<struct aiocb> aiocbs;  // Array of AIO control blocks
    int num_entries;                   // Total number of entries expected
    int num_completed;                 // Number of completed operations
    int num_submitted;  // Track number of submitted I/Os
    bool is_read;                      // Whether this is a read operation

    // Delete copy and move operations
    aioQueue(const aioQueue&) = delete;
    aioQueue& operator=(const aioQueue&) = delete;
    aioQueue(aioQueue&&) = delete;
    aioQueue& operator=(aioQueue&&) = delete;

public:
    aioQueue(int num_entries, bool is_read);
    ~aioQueue();
    nixl_status_t submit() override;
    nixl_status_t checkCompleted() override;
    nixl_status_t prepareIO(int fd, void* buf, size_t len, off_t offset) override;
};

#endif // AIO_QUEUE_H
