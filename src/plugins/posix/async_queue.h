#ifndef ASYNC_QUEUE_H
#define ASYNC_QUEUE_H

#include "common/status.h"
#include "nixl_types.h"
#include <sys/types.h>

// Abstract base class for async I/O operations
class nixlPosixQueue {
public:
    virtual ~nixlPosixQueue() = default;
    virtual nixl_status_t submit() = 0;
    virtual nixl_status_t checkCompleted() = 0;
    virtual nixl_status_t prepareIO(int fd, void* buf, size_t len, off_t offset) = 0;
};

#endif // ASYNC_QUEUE_H
