/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef POSIX_MT_QUEUE_H
#define POSIX_MT_QUEUE_H

#include <atomic>
#include <future>
#include <sys/types.h>
#include <vector>
#include "posix_queue.h"
#include "taskflow/core/executor.hpp"
#include "taskflow/taskflow.hpp"

// posixMtQueue implements nixlPosixQueue using pwrite/pread syscalls dispatched
// via a taskflow thread pool (one task per descriptor). This mirrors the
// approach used by the gds_mt plugin and is useful when kernel async I/O
// (io_uring / libaio) is unavailable or when a synchronous-but-parallel path
// is preferred.
class posixMtQueue : public nixlPosixQueue {
private:
    struct TransferReq {
        int    fd;
        void  *buf;
        size_t len;
        off_t  offset;
    };

    nixl_xfer_op_t             operation_;
    std::vector<TransferReq>   requests_;
    tf::Taskflow               taskflow_;
    std::future<void>          running_transfer_;
    std::atomic<nixl_status_t> overall_status_;
    tf::Executor              *executor_; // Non-owning; lifetime managed by nixlPosixEngine

    posixMtQueue(const posixMtQueue &) = delete;
    posixMtQueue &operator=(const posixMtQueue &) = delete;
    posixMtQueue(posixMtQueue &&) = delete;
    posixMtQueue &operator=(posixMtQueue &&) = delete;

public:
    posixMtQueue(int num_entries, nixl_xfer_op_t operation, tf::Executor *executor);
    ~posixMtQueue() override;

    // Store one transfer descriptor. All prepIO() calls must complete before submit().
    nixl_status_t prepIO(int fd, void *buf, size_t len, off_t offset) override;

    // Build the taskflow (one task per stored descriptor) and dispatch to the
    // executor. Returns NIXL_IN_PROG immediately; use checkCompleted() to poll.
    nixl_status_t submit(const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote) override;

    // Poll whether all taskflow tasks have completed.
    nixl_status_t checkCompleted() override;
};

#endif // POSIX_MT_QUEUE_H
