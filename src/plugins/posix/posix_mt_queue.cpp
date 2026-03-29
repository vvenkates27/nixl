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

#include <cerrno>
#include <cstring>
#include <unistd.h>
#include "common/nixl_log.h"
#include "common/nixl_time.h"
#include "posix_mt_queue.h"

posixMtQueue::posixMtQueue(int num_entries, nixl_xfer_op_t operation, tf::Executor *executor)
    : operation_(operation), executor_(executor) {
    requests_.reserve(num_entries);
    overall_status_.store(NIXL_SUCCESS);
}

posixMtQueue::~posixMtQueue() {
    // Wait for any in-flight taskflow to finish before releasing queue resources.
    if (running_transfer_.valid()) {
        running_transfer_.wait();
    }
}

nixl_status_t posixMtQueue::prepIO(int fd, void *buf, size_t len, off_t offset) {
    // Append to the pre-reserved vector so pointers stay stable across all
    // prepIO() calls (no reallocation since we reserved num_entries upfront).
    requests_.push_back({fd, buf, len, offset});
    return NIXL_SUCCESS;
}

nixl_status_t posixMtQueue::submit(const nixl_meta_dlist_t & /*local*/,
                                   const nixl_meta_dlist_t & /*remote*/) {
    overall_status_.store(NIXL_SUCCESS);

    // Build one taskflow task per transfer request. Taking the address of each
    // element is safe because requests_ was reserved in the constructor and no
    // further push_back() calls can occur after submit().
    for (TransferReq &req : requests_) {
        TransferReq *captured_req = &req;
        taskflow_.emplace(
            [captured_req, op = operation_, status = &overall_status_]() {
                ssize_t ret;
                if (op == NIXL_READ) {
                    ret = pread(captured_req->fd, captured_req->buf,
                                captured_req->len, captured_req->offset);
                } else {
                    ret = pwrite(captured_req->fd, captured_req->buf,
                                 captured_req->len, captured_req->offset);
                }

                if (ret < 0) {
                    NIXL_ERROR << "POSIX_MT: " << (op == NIXL_READ ? "pread" : "pwrite")
                               << " failed: " << strerror(errno);
                    status->store(NIXL_ERR_BACKEND);
                } else if (static_cast<size_t>(ret) != captured_req->len) {
                    NIXL_ERROR << "POSIX_MT: short " << (op == NIXL_READ ? "read" : "write")
                               << ": " << ret << " of " << captured_req->len << " bytes";
                    status->store(NIXL_ERR_BACKEND);
                }
            });
    }

    running_transfer_ = executor_->run(taskflow_);
    return NIXL_IN_PROG;
}

nixl_status_t posixMtQueue::checkCompleted() {
    if (running_transfer_.wait_for(nixlTime::seconds(0)) != std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    running_transfer_.get();
    return overall_status_.load();
}
