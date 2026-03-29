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

#ifndef QUEUE_FACTORY_IMPL_H
#define QUEUE_FACTORY_IMPL_H

#include "posix_queue.h"
#include "taskflow/core/executor.hpp"

namespace QueueFactory {
std::unique_ptr<nixlPosixQueue>
createPosixAioQueue(int num_entries, nixl_xfer_op_t operation);

std::unique_ptr<nixlPosixQueue>
createUringQueue(int num_entries, nixl_xfer_op_t operation);

std::unique_ptr<nixlPosixQueue>
createLinuxAioQueue(int num_entries, nixl_xfer_op_t operation);

// Create a queue that dispatches pwrite/pread calls via a taskflow thread pool.
// executor must outlive the returned queue (typically owned by nixlPosixEngine).
std::unique_ptr<nixlPosixQueue>
createPwriteQueue(int num_entries, nixl_xfer_op_t operation, tf::Executor *executor);

bool
isPosixAioAvailable();
bool
isLinuxAioAvailable();
bool
isUringAvailable();
bool
isPwriteAvailable();
}; // namespace QueueFactory

#endif // QUEUE_FACTORY_IMPL_H
