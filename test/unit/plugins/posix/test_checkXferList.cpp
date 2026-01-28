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

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"
#include <absl/strings/str_format.h>

namespace {
    const size_t page_size = sysconf(_SC_PAGESIZE);
    constexpr size_t test_size = 4096; // 4KB test
    constexpr int num_entries = 4;
    constexpr char test_pattern[] = "NIXL checkXferList test pattern";
    constexpr char test_dir[] = "tmp/checkXferList_test";

    // Custom deleter for posix_memalign allocated memory
    struct PosixMemalignDeleter {
        void operator()(void* ptr) const {
            if (ptr) free(ptr);
        }
    };

    class TempFile {
    public:
        int fd;
        std::string path;

        TempFile(const std::string& filename, int flags, mode_t mode = 0600)
            : path(filename) {
            fd = open(filename.c_str(), flags, mode);
            if (fd == -1) {
                throw std::runtime_error("Failed to open file: " + filename);
            }
        }

        ~TempFile() {
            if (fd != -1) {
                close(fd);
            }
            if (!path.empty()) {
                unlink(path.c_str());
            }
        }

        operator int() const { return fd; }
    };

    void fill_pattern(void* buffer, size_t size) {
        char* buf = static_cast<char*>(buffer);
        size_t pattern_len = strlen(test_pattern);
        for (size_t i = 0; i < size; i += pattern_len) {
            size_t copy_len = std::min(pattern_len, size - i);
            memcpy(buf + i, test_pattern, copy_len);
        }
    }

    bool verify_pattern(void* buffer, size_t size) {
        char* buf = static_cast<char*>(buffer);
        size_t pattern_len = strlen(test_pattern);
        for (size_t i = 0; i < size; i += pattern_len) {
            size_t cmp_len = std::min(pattern_len, size - i);
            if (memcmp(buf + i, test_pattern, cmp_len) != 0) {
                return false;
            }
        }
        return true;
    }
}

// Test checkXferList returns NOT_SUPPORTED
int test_checkXferList_not_supported(nixlBackendH* backend) {
    std::cout << "\nTest: checkXferList returns NOT_SUPPORTED" << std::endl;

    try {
        // Allocate memory buffers
        std::vector<std::unique_ptr<void, PosixMemalignDeleter>> buffers;
        std::vector<TempFile> files;

        for (int i = 0; i < num_entries; ++i) {
            void* ptr;
            if (posix_memalign(&ptr, page_size, test_size) != 0) {
                std::cerr << "Memory allocation failed" << std::endl;
                return 1;
            }
            buffers.emplace_back(ptr, PosixMemalignDeleter());
            fill_pattern(ptr, test_size);

            std::string filename = std::string(test_dir) + "/test_file_" + std::to_string(i);
            files.emplace_back(filename, O_RDWR | O_CREAT, 0600);
        }

        // Create descriptor lists
        nixl_reg_dlist_t dram_reg(DRAM_SEG);
        nixl_reg_dlist_t file_reg(FILE_SEG);
        nixl_xfer_dlist_t dram_xfer(DRAM_SEG);
        nixl_xfer_dlist_t file_xfer(FILE_SEG);

        std::unique_ptr<nixlBlobDesc[]> dram_descs(new nixlBlobDesc[num_entries]);
        std::unique_ptr<nixlBlobDesc[]> file_descs(new nixlBlobDesc[num_entries]);

        // Register memory
        for (int i = 0; i < num_entries; ++i) {
            dram_descs[i].len = test_size;
            dram_descs[i].addr = reinterpret_cast<uintptr_t>(buffers[i].get());
            dram_reg.addDesc(dram_descs[i]);

            file_descs[i].len = test_size;
            file_descs[i].devId = files[i].fd;
            file_descs[i].addr = 0;
            file_descs[i].metaInfo.buffer = reinterpret_cast<void*>(const_cast<char*>(files[i].path.c_str()));
            file_descs[i].metaInfo.len = files[i].path.length();
            file_reg.addDesc(file_descs[i]);
        }

        nixl_status_t status = backend->registerDList(dram_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        status = backend->registerDList(file_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register files: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        // Prepare transfer
        dram_xfer = nixl_xfer_dlist_t(dram_reg);
        file_xfer = nixl_xfer_dlist_t(file_reg);

        nixlXferReqH* xfer_handle = nullptr;
        status = backend->prepXfer(nixl_xfer_op_t::WRITE, dram_xfer, file_xfer, xfer_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "prepXfer failed: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        // Post transfer
        status = backend->postXfer(xfer_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "postXfer failed: " << nixlEnumStrings::statusStr(status) << std::endl;
            backend->releaseReqH(xfer_handle);
            return 1;
        }

        // Test checkXferList - should return NOT_SUPPORTED
        std::vector<nixl_status_t> entry_status;
        status = backend->checkXferList(xfer_handle, entry_status);

        if (status != NIXL_ERR_NOT_SUPPORTED) {
            std::cerr << "Expected NIXL_ERR_NOT_SUPPORTED, got: " << nixlEnumStrings::statusStr(status) << std::endl;
            backend->releaseReqH(xfer_handle);
            return 1;
        }

        std::cout << "  PASS: checkXferList correctly returns NOT_SUPPORTED" << std::endl;

        // Clean up with standard checkXfer
        while ((status = backend->checkXfer(xfer_handle)) == NIXL_INPROGRESS) {
            usleep(1000);
        }

        if (status != NIXL_SUCCESS) {
            std::cerr << "checkXfer failed: " << nixlEnumStrings::statusStr(status) << std::endl;
            backend->releaseReqH(xfer_handle);
            return 1;
        }

        backend->releaseReqH(xfer_handle);
        backend->deregisterDList(dram_reg);
        backend->deregisterDList(file_reg);

        std::cout << "  PASS: Standard checkXfer works correctly" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

// Test graceful fallback behavior
int test_graceful_fallback(nixlBackendH* backend) {
    std::cout << "\nTest: Graceful fallback to checkXfer" << std::endl;

    try {
        // Single entry test
        void* ptr;
        if (posix_memalign(&ptr, page_size, test_size) != 0) {
            std::cerr << "Memory allocation failed" << std::endl;
            return 1;
        }
        std::unique_ptr<void, PosixMemalignDeleter> buffer(ptr, PosixMemalignDeleter());
        fill_pattern(ptr, test_size);

        std::string filename = std::string(test_dir) + "/fallback_test";
        TempFile file(filename, O_RDWR | O_CREAT, 0600);

        // Create descriptor lists
        nixl_reg_dlist_t dram_reg(DRAM_SEG);
        nixl_reg_dlist_t file_reg(FILE_SEG);

        nixlBlobDesc dram_desc;
        dram_desc.len = test_size;
        dram_desc.addr = reinterpret_cast<uintptr_t>(ptr);
        dram_reg.addDesc(dram_desc);

        nixlBlobDesc file_desc;
        file_desc.len = test_size;
        file_desc.devId = file.fd;
        file_desc.addr = 0;
        file_desc.metaInfo.buffer = reinterpret_cast<void*>(const_cast<char*>(filename.c_str()));
        file_desc.metaInfo.len = filename.length();
        file_reg.addDesc(file_desc);

        nixl_status_t status = backend->registerDList(dram_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM" << std::endl;
            return 1;
        }

        status = backend->registerDList(file_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register file" << std::endl;
            return 1;
        }

        nixl_xfer_dlist_t dram_xfer(dram_reg);
        nixl_xfer_dlist_t file_xfer(file_reg);

        nixlXferReqH* xfer_handle = nullptr;
        status = backend->prepXfer(nixl_xfer_op_t::WRITE, dram_xfer, file_xfer, xfer_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "prepXfer failed" << std::endl;
            return 1;
        }

        status = backend->postXfer(xfer_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "postXfer failed" << std::endl;
            backend->releaseReqH(xfer_handle);
            return 1;
        }

        // Try checkXferList first
        std::vector<nixl_status_t> entry_status;
        status = backend->checkXferList(xfer_handle, entry_status);

        if (status == NIXL_ERR_NOT_SUPPORTED) {
            std::cout << "  INFO: checkXferList not supported, using fallback" << std::endl;

            // Fall back to standard checkXfer
            while ((status = backend->checkXfer(xfer_handle)) == NIXL_INPROGRESS) {
                usleep(1000);
            }

            if (status != NIXL_SUCCESS) {
                std::cerr << "Fallback checkXfer failed" << std::endl;
                backend->releaseReqH(xfer_handle);
                return 1;
            }

            std::cout << "  PASS: Fallback to checkXfer successful" << std::endl;
        } else {
            std::cout << "  INFO: checkXferList supported" << std::endl;
        }

        backend->releaseReqH(xfer_handle);
        backend->deregisterDList(dram_reg);
        backend->deregisterDList(file_reg);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================================" << std::endl;
    std::cout << "              POSIX checkXferList Unit Test" << std::endl;
    std::cout << "==============================================================" << std::endl;

    // Create test directory
    mkdir(test_dir, 0755);

    // Initialize NIXL agent
    nixlAgent agent("CheckXferListTester", nixlAgentConfig(true));

    // Test with different queue types
    std::vector<std::string> queue_types = {"AIO", "URING", "POSIXAIO"};

    for (const auto& queue_type : queue_types) {
        std::cout << "\n--------------------------------------------------------------" << std::endl;
        std::cout << "Testing with " << queue_type << " backend" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        nixl_b_params_t params;
        if (queue_type == "AIO") {
            params["use_aio"] = "true";
        } else if (queue_type == "URING") {
            params["use_uring"] = "true";
        } else if (queue_type == "POSIXAIO") {
            params["use_posix_aio"] = "true";
        }

        nixlBackendH* backend = nullptr;
        nixl_status_t status = agent.createBackend("POSIX", params, backend);

        if (status != NIXL_SUCCESS) {
            std::cout << "SKIP: " << queue_type << " backend not available" << std::endl;
            continue;
        }

        std::cout << "Backend created successfully" << std::endl;

        int result = 0;
        result |= test_checkXferList_not_supported(backend);
        result |= test_graceful_fallback(backend);

        if (result == 0) {
            std::cout << "\n" << queue_type << " backend tests: PASSED" << std::endl;
        } else {
            std::cout << "\n" << queue_type << " backend tests: FAILED" << std::endl;
            return 1;
        }
    }

    std::cout << "\n==============================================================" << std::endl;
    std::cout << "                  All tests PASSED" << std::endl;
    std::cout << "==============================================================" << std::endl;

    // Clean up test directory
    rmdir(test_dir);

    return 0;
}
