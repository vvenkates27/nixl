/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <filesystem>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"
#include "temp_file.h"

namespace {
    const size_t page_size = sysconf(_SC_PAGESIZE);
    constexpr size_t test_size = 4096; // 4KB test
    constexpr int num_entries = 4;
    constexpr char test_pattern[] = "NIXL HF3FS checkXferList test pattern";
    constexpr char test_dir[] = "/mnt/3fs/checkXferList_test";

    // Custom deleter for aligned memory
    struct AlignedDeleter {
        void operator()(void* ptr) const {
            if (ptr) free(ptr);
        }
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

    void clear_buffer(void* buffer, size_t size) {
        memset(buffer, 0, size);
    }
}

// Test per-IO status tracking with async I/O
int test_per_io_status_tracking(nixlAgent& agent, nixlBackendH* backend) {
    std::cout << "\nTest: Per-IO status tracking with async I/O" << std::endl;

    try {
        // Setup test directory
        std::filesystem::create_directories(test_dir);

        // Allocate memory buffers
        std::vector<std::unique_ptr<void, AlignedDeleter>> buffers;
        std::vector<tempFile> files;

        for (int i = 0; i < num_entries; ++i) {
            void* ptr = aligned_alloc(page_size, test_size);
            if (!ptr) {
                std::cerr << "Memory allocation failed" << std::endl;
                return 1;
            }
            buffers.emplace_back(ptr);
            fill_pattern(ptr, test_size);

            std::string filename = std::string(test_dir) + "/test_file_" + std::to_string(i);
            files.emplace_back(filename, O_RDWR | O_CREAT, 0600);
        }

        // Create descriptor lists
        nixl_reg_dlist_t dram_reg(DRAM_SEG);
        nixl_reg_dlist_t file_reg(FILE_SEG);

        std::unique_ptr<nixlBlobDesc[]> dram_descs(new nixlBlobDesc[num_entries]);
        std::unique_ptr<nixlBlobDesc[]> file_descs(new nixlBlobDesc[num_entries]);

        // Register memory
        for (int i = 0; i < num_entries; ++i) {
            dram_descs[i].len = test_size;
            dram_descs[i].addr = reinterpret_cast<uintptr_t>(buffers[i].get());
            dram_reg.addDesc(dram_descs[i]);

            file_descs[i].len = test_size;
            file_descs[i].devId = files[i];
            file_descs[i].addr = 0;
            file_reg.addDesc(file_descs[i]);
        }

        nixl_status_t status = agent.registerMem(dram_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM: " << status << std::endl;
            return 1;
        }

        status = agent.registerMem(file_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register files: " << status << std::endl;
            return 1;
        }

        // Prepare transfer
        nixl_xfer_dlist_t dram_xfer = dram_reg.trim();
        nixl_xfer_dlist_t file_xfer = file_reg.trim();

        nixlXferReqH* xfer_handle = nullptr;
        status = agent.createXferReq(NIXL_WRITE, dram_xfer, file_xfer, "HF3FSTester", xfer_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "createXferReq failed: " << status << std::endl;
            return 1;
        }

        // Post transfer
        status = agent.postXferReq(xfer_handle);
        if (status < 0) {
            std::cerr << "postXferReq failed: " << status << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        // Test checkXferList
        std::vector<nixl_status_t> entry_status;
        int max_polls = 10000;
        int poll_count = 0;

        while (poll_count < max_polls) {
            status = backend->checkXferList(xfer_handle, entry_status);

            if (status != NIXL_IN_PROG) {
                break;
            }

            usleep(1000); // 1ms
            poll_count++;
        }

        if (status < 0) {
            std::cerr << "checkXferList failed: " << status << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        if (poll_count >= max_polls) {
            std::cerr << "Transfer timed out" << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        // Verify entry status
        if (entry_status.size() != num_entries) {
            std::cerr << "Expected " << num_entries << " entries, got " << entry_status.size() << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        bool all_success = true;
        for (size_t i = 0; i < entry_status.size(); ++i) {
            if (entry_status[i] != NIXL_SUCCESS) {
                std::cerr << "Entry " << i << " status: " << entry_status[i] << std::endl;
                all_success = false;
            }
        }

        if (!all_success) {
            std::cerr << "Not all entries succeeded" << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        std::cout << "  PASS: All IOs completed successfully with correct per-IO status" << std::endl;

        agent.releaseXferReq(xfer_handle);
        agent.deregisterMem(dram_reg);
        agent.deregisterMem(file_reg);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

// Test batch transfer with all IOs succeeding
int test_batch_transfer_success(nixlAgent& agent, nixlBackendH* backend) {
    std::cout << "\nTest: Batch transfer with all IOs succeeding" << std::endl;

    try {
        // Setup test directory
        std::filesystem::create_directories(test_dir);

        // Allocate memory buffers
        std::vector<std::unique_ptr<void, AlignedDeleter>> write_buffers;
        std::vector<std::unique_ptr<void, AlignedDeleter>> read_buffers;
        std::vector<tempFile> files;

        for (int i = 0; i < num_entries; ++i) {
            void* write_ptr = aligned_alloc(page_size, test_size);
            void* read_ptr = aligned_alloc(page_size, test_size);
            if (!write_ptr || !read_ptr) {
                std::cerr << "Memory allocation failed" << std::endl;
                return 1;
            }
            write_buffers.emplace_back(write_ptr);
            read_buffers.emplace_back(read_ptr);
            fill_pattern(write_ptr, test_size);
            clear_buffer(read_ptr, test_size);

            std::string filename = std::string(test_dir) + "/batch_file_" + std::to_string(i);
            files.emplace_back(filename, O_RDWR | O_CREAT, 0600);
        }

        // Create descriptor lists for write
        nixl_reg_dlist_t write_dram_reg(DRAM_SEG);
        nixl_reg_dlist_t file_reg(FILE_SEG);

        std::unique_ptr<nixlBlobDesc[]> write_dram_descs(new nixlBlobDesc[num_entries]);
        std::unique_ptr<nixlBlobDesc[]> file_descs(new nixlBlobDesc[num_entries]);

        for (int i = 0; i < num_entries; ++i) {
            write_dram_descs[i].len = test_size;
            write_dram_descs[i].addr = reinterpret_cast<uintptr_t>(write_buffers[i].get());
            write_dram_reg.addDesc(write_dram_descs[i]);

            file_descs[i].len = test_size;
            file_descs[i].devId = files[i];
            file_descs[i].addr = 0;
            file_reg.addDesc(file_descs[i]);
        }

        nixl_status_t status = agent.registerMem(write_dram_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register write DRAM" << std::endl;
            return 1;
        }

        status = agent.registerMem(file_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register files" << std::endl;
            return 1;
        }

        // Write transfer
        nixl_xfer_dlist_t write_dram_xfer = write_dram_reg.trim();
        nixl_xfer_dlist_t file_xfer = file_reg.trim();

        nixlXferReqH* write_handle = nullptr;
        status = agent.createXferReq(NIXL_WRITE, write_dram_xfer, file_xfer, "HF3FSTester", write_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "createXferReq for write failed" << std::endl;
            return 1;
        }

        status = agent.postXferReq(write_handle);
        if (status < 0) {
            std::cerr << "postXferReq for write failed" << std::endl;
            agent.releaseXferReq(write_handle);
            return 1;
        }

        // Wait for write to complete
        std::vector<nixl_status_t> write_status;
        int max_polls = 10000;
        int poll_count = 0;

        while (poll_count < max_polls) {
            status = backend->checkXferList(write_handle, write_status);
            if (status != NIXL_IN_PROG) {
                break;
            }
            usleep(1000);
            poll_count++;
        }

        if (status != NIXL_SUCCESS) {
            std::cerr << "Write transfer failed: " << status << std::endl;
            agent.releaseXferReq(write_handle);
            return 1;
        }

        agent.releaseXferReq(write_handle);

        // Sync files
        for (int i = 0; i < num_entries; ++i) {
            fsync(files[i]);
        }

        // Create descriptor list for read
        nixl_reg_dlist_t read_dram_reg(DRAM_SEG);
        std::unique_ptr<nixlBlobDesc[]> read_dram_descs(new nixlBlobDesc[num_entries]);

        for (int i = 0; i < num_entries; ++i) {
            read_dram_descs[i].len = test_size;
            read_dram_descs[i].addr = reinterpret_cast<uintptr_t>(read_buffers[i].get());
            read_dram_reg.addDesc(read_dram_descs[i]);
        }

        status = agent.registerMem(read_dram_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register read DRAM" << std::endl;
            return 1;
        }

        // Read transfer
        nixl_xfer_dlist_t read_dram_xfer = read_dram_reg.trim();

        nixlXferReqH* read_handle = nullptr;
        status = agent.createXferReq(NIXL_READ, read_dram_xfer, file_xfer, "HF3FSTester", read_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "createXferReq for read failed" << std::endl;
            return 1;
        }

        status = agent.postXferReq(read_handle);
        if (status < 0) {
            std::cerr << "postXferReq for read failed" << std::endl;
            agent.releaseXferReq(read_handle);
            return 1;
        }

        // Wait for read to complete
        std::vector<nixl_status_t> read_status;
        poll_count = 0;

        while (poll_count < max_polls) {
            status = backend->checkXferList(read_handle, read_status);
            if (status != NIXL_IN_PROG) {
                break;
            }
            usleep(1000);
            poll_count++;
        }

        if (status != NIXL_SUCCESS) {
            std::cerr << "Read transfer failed: " << status << std::endl;
            agent.releaseXferReq(read_handle);
            return 1;
        }

        // Verify all entries succeeded
        if (read_status.size() != num_entries) {
            std::cerr << "Expected " << num_entries << " read entries" << std::endl;
            agent.releaseXferReq(read_handle);
            return 1;
        }

        for (size_t i = 0; i < read_status.size(); ++i) {
            if (read_status[i] != NIXL_SUCCESS) {
                std::cerr << "Read entry " << i << " failed with status: " << read_status[i] << std::endl;
                agent.releaseXferReq(read_handle);
                return 1;
            }
        }

        // Verify data
        for (int i = 0; i < num_entries; ++i) {
            if (!verify_pattern(read_buffers[i].get(), test_size)) {
                std::cerr << "Data verification failed for buffer " << i << std::endl;
                agent.releaseXferReq(read_handle);
                return 1;
            }
        }

        std::cout << "  PASS: Batch transfer completed with all IOs succeeding" << std::endl;

        agent.releaseXferReq(read_handle);
        agent.deregisterMem(write_dram_reg);
        agent.deregisterMem(read_dram_reg);
        agent.deregisterMem(file_reg);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

// Test status collection from io_list
int test_status_collection(nixlAgent& agent, nixlBackendH* backend) {
    std::cout << "\nTest: Status collection from io_list" << std::endl;

    try {
        // Setup test directory
        std::filesystem::create_directories(test_dir);

        const int test_count = 8; // More entries to test collection
        std::vector<std::unique_ptr<void, AlignedDeleter>> buffers;
        std::vector<tempFile> files;

        for (int i = 0; i < test_count; ++i) {
            void* ptr = aligned_alloc(page_size, test_size);
            if (!ptr) {
                std::cerr << "Memory allocation failed" << std::endl;
                return 1;
            }
            buffers.emplace_back(ptr);
            fill_pattern(ptr, test_size);

            std::string filename = std::string(test_dir) + "/collect_file_" + std::to_string(i);
            files.emplace_back(filename, O_RDWR | O_CREAT, 0600);
        }

        nixl_reg_dlist_t dram_reg(DRAM_SEG);
        nixl_reg_dlist_t file_reg(FILE_SEG);

        std::unique_ptr<nixlBlobDesc[]> dram_descs(new nixlBlobDesc[test_count]);
        std::unique_ptr<nixlBlobDesc[]> file_descs(new nixlBlobDesc[test_count]);

        for (int i = 0; i < test_count; ++i) {
            dram_descs[i].len = test_size;
            dram_descs[i].addr = reinterpret_cast<uintptr_t>(buffers[i].get());
            dram_reg.addDesc(dram_descs[i]);

            file_descs[i].len = test_size;
            file_descs[i].devId = files[i];
            file_descs[i].addr = 0;
            file_reg.addDesc(file_descs[i]);
        }

        nixl_status_t status = agent.registerMem(dram_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM" << std::endl;
            return 1;
        }

        status = agent.registerMem(file_reg);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Failed to register files" << std::endl;
            return 1;
        }

        nixl_xfer_dlist_t dram_xfer = dram_reg.trim();
        nixl_xfer_dlist_t file_xfer = file_reg.trim();

        nixlXferReqH* xfer_handle = nullptr;
        status = agent.createXferReq(NIXL_WRITE, dram_xfer, file_xfer, "HF3FSTester", xfer_handle);
        if (status != NIXL_SUCCESS) {
            std::cerr << "createXferReq failed" << std::endl;
            return 1;
        }

        status = agent.postXferReq(xfer_handle);
        if (status < 0) {
            std::cerr << "postXferReq failed" << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        // Poll and verify status collection
        std::vector<nixl_status_t> entry_status;
        int max_polls = 10000;
        int poll_count = 0;

        while (poll_count < max_polls) {
            status = backend->checkXferList(xfer_handle, entry_status);

            // Verify status vector is properly populated
            if (entry_status.size() != test_count) {
                std::cerr << "Status vector size mismatch: " << entry_status.size() << " vs " << test_count << std::endl;
                agent.releaseXferReq(xfer_handle);
                return 1;
            }

            if (status != NIXL_IN_PROG) {
                break;
            }

            usleep(1000);
            poll_count++;
        }

        if (status != NIXL_SUCCESS) {
            std::cerr << "Transfer failed: " << status << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        // Final verification
        if (entry_status.size() != test_count) {
            std::cerr << "Final status vector size incorrect" << std::endl;
            agent.releaseXferReq(xfer_handle);
            return 1;
        }

        for (size_t i = 0; i < entry_status.size(); ++i) {
            if (entry_status[i] != NIXL_SUCCESS) {
                std::cerr << "Entry " << i << " has incorrect status: " << entry_status[i] << std::endl;
                agent.releaseXferReq(xfer_handle);
                return 1;
            }
        }

        std::cout << "  PASS: Status collection from io_list works correctly" << std::endl;

        agent.releaseXferReq(xfer_handle);
        agent.deregisterMem(dram_reg);
        agent.deregisterMem(file_reg);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

int main() {
    std::cout << "HF3FS checkXferList Tests" << std::endl;
    std::cout << "=========================" << std::endl;

    // Initialize NIXL
    nixlAgentConfig cfg(true);
    nixlAgent agent("HF3FSTester", cfg);

    nixl_b_params_t params;
    nixlBackendH* backend;
    nixl_status_t status = agent.createBackend("HF3FS", params, backend);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Failed to create HF3FS backend: " << status << std::endl;
        return 1;
    }

    if (!backend) {
        std::cerr << "Backend is null" << std::endl;
        return 1;
    }

    // Setup test directory
    try {
        std::filesystem::create_directories(test_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create test directory: " << e.what() << std::endl;
        return 1;
    }

    int result = 0;

    // Run tests
    result |= test_per_io_status_tracking(agent, backend);
    result |= test_batch_transfer_success(agent, backend);
    result |= test_status_collection(agent, backend);

    // Cleanup
    try {
        std::filesystem::remove_all(test_dir);
    } catch (...) {
        // Ignore cleanup errors
    }

    if (result == 0) {
        std::cout << "\nAll tests PASSED!" << std::endl;
    } else {
        std::cout << "\nSome tests FAILED!" << std::endl;
    }

    return result;
}
