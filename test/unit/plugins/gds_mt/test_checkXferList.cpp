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
#include <string>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <getopt.h>
#include "nixl_descriptors.h"
#include "nixl_params.h"
#include "nixl.h"

// Test configuration
#define DEFAULT_NUM_ENTRIES 10
#define DEFAULT_TRANSFER_SIZE (1 * 1024 * 1024)  // 1MB
#define TEST_PHRASE "NIXL GDS MT checkXferList Test Pattern"
#define TEST_PHRASE_LEN (sizeof(TEST_PHRASE) - 1)

// Get system page size
static size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options] <directory_path>\n"
              << "Options:\n"
              << "  -n, --num-entries N     Number of entries in batch (default: " << DEFAULT_NUM_ENTRIES << ")\n"
              << "  -s, --size SIZE         Size of each transfer (default: " << DEFAULT_TRANSFER_SIZE << " bytes)\n"
              << "  -t, --num-threads N     Number of threads for GDS MT (default: auto)\n"
              << "  -h, --help              Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " -n 5 -s 1048576 -t 4 /tmp\n";
}

size_t parse_size(const char* size_str) {
    char* end;
    size_t size = strtoull(size_str, &end, 10);
    if (end == size_str) {
        return 0;
    }
    if (*end) {
        switch (toupper(*end)) {
            case 'K': size *= 1024; break;
            case 'M': size *= 1024 * 1024; break;
            case 'G': size *= 1024 * 1024 * 1024; break;
            default: return 0;
        }
    }
    return size;
}

void fill_test_pattern(void* buffer, size_t size) {
    char* buf = (char*)buffer;
    size_t phrase_len = TEST_PHRASE_LEN;
    size_t offset = 0;
    while (offset < size) {
        size_t remaining = size - offset;
        size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
        memcpy(buf + offset, TEST_PHRASE, copy_len);
        offset += copy_len;
    }
}

cudaError_t fill_gpu_test_pattern(void* gpu_buffer, size_t size) {
    char* host_buffer = (char*)malloc(size);
    if (!host_buffer) {
        return cudaErrorMemoryAllocation;
    }
    fill_test_pattern(host_buffer, size);
    cudaError_t err = cudaMemcpy(gpu_buffer, host_buffer, size, cudaMemcpyHostToDevice);
    free(host_buffer);
    return err;
}

std::string generate_test_filename(const std::string& base_name, int index) {
    return base_name + "_" + std::to_string(index);
}

bool run_parallel_batch_test(const std::string& dir_path, int num_entries, size_t transfer_size, size_t num_threads) {
    std::cout << "\n=== Test 1: Parallel batch with all entries succeeding (TaskFlow) ===" << std::endl;

    nixl_status_t ret = NIXL_SUCCESS;
    void **vram_addr = new void*[num_entries];
    int *fd = new int[num_entries];

    nixlAgentConfig cfg(true);
    nixl_b_params_t params;
    nixlBlobDesc *vram_buf = new nixlBlobDesc[num_entries];
    nixlBlobDesc *ftrans = new nixlBlobDesc[num_entries];
    nixlBackendH *gds_mt;
    nixl_reg_dlist_t vram_for_gds_mt(VRAM_SEG);
    nixl_reg_dlist_t file_for_gds_mt(FILE_SEG);

    // Initialize agent
    nixlAgent agent("TestGdsMtCheckXferList", cfg);

    // Configure GDS MT backend
    if (num_threads > 0) {
        params["thread_count"] = std::to_string(num_threads);
    }

    ret = agent.createBackend("GDS_MT", params, gds_mt);
    if (ret != NIXL_SUCCESS || gds_mt == NULL) {
        std::cerr << "Failed to create GDS_MT backend" << std::endl;
        goto cleanup;
    }

    // Allocate and initialize buffers
    for (int i = 0; i < num_entries; i++) {
        // Allocate VRAM
        if (cudaMalloc(&vram_addr[i], transfer_size) != cudaSuccess) {
            std::cerr << "CUDA malloc failed" << std::endl;
            goto cleanup;
        }
        if (fill_gpu_test_pattern(vram_addr[i], transfer_size) != cudaSuccess) {
            std::cerr << "CUDA buffer initialization failed" << std::endl;
            goto cleanup;
        }

        // Create test file
        std::string filename = dir_path + "/" + generate_test_filename("test_gds_mt_parallel", i);
        fd[i] = open(filename.c_str(), O_RDWR|O_CREAT, 0744);
        if (fd[i] < 0) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            goto cleanup;
        }

        // Setup descriptors
        vram_buf[i].addr = (uintptr_t)(vram_addr[i]);
        vram_buf[i].len = transfer_size;
        vram_buf[i].devId = 0;
        vram_for_gds_mt.addDesc(vram_buf[i]);

        ftrans[i].addr = 0;
        ftrans[i].len = transfer_size;
        ftrans[i].devId = fd[i];
        file_for_gds_mt.addDesc(ftrans[i]);
    }

    // Register memory
    ret = agent.registerMem(file_for_gds_mt);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register file memory" << std::endl;
        goto cleanup;
    }

    ret = agent.registerMem(vram_for_gds_mt);
    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to register VRAM memory" << std::endl;
        goto cleanup;
    }

    // Create and execute transfer
    {
        nixl_xfer_dlist_t vram_list = vram_for_gds_mt.trim();
        nixl_xfer_dlist_t file_list = file_for_gds_mt.trim();
        nixlXferReqH* write_req = nullptr;

        ret = agent.createXferReq(NIXL_WRITE, vram_list, file_list, "TestGdsMtCheckXferList", write_req);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to create write transfer request" << std::endl;
            goto cleanup;
        }

        // Post transfer
        int status = agent.postXferReq(write_req);
        if (status < 0) {
            std::cerr << "Failed to post write transfer request" << std::endl;
            agent.releaseXferReq(write_req);
            goto cleanup;
        }

        // Check per-entry status with atomic updates
        std::vector<nixl_status_t> entry_status;
        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(write_req, entry_status);
            if (status < 0) {
                std::cerr << "Error during write transfer" << std::endl;
                agent.releaseXferReq(write_req);
                goto cleanup;
            }
        }

        // Verify all entries succeeded
        if (entry_status.size() != (size_t)num_entries) {
            std::cerr << "Expected " << num_entries << " status entries, got "
                      << entry_status.size() << std::endl;
            agent.releaseXferReq(write_req);
            goto cleanup;
        }

        bool all_success = true;
        for (int i = 0; i < num_entries; i++) {
            if (entry_status[i] != NIXL_SUCCESS) {
                std::cerr << "Entry " << i << " failed with status " << entry_status[i] << std::endl;
                all_success = false;
            }
        }

        agent.releaseXferReq(write_req);

        if (!all_success) {
            goto cleanup;
        }

        std::cout << "Test 1 PASSED: All " << num_entries << " entries completed successfully in parallel" << std::endl;
    }

cleanup:
    // Cleanup
    agent.deregisterMem(file_for_gds_mt);
    agent.deregisterMem(vram_for_gds_mt);

    for (int i = 0; i < num_entries; i++) {
        if (fd[i] > 0) {
            char proc_path[64];
            char filename[PATH_MAX];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd[i]);
            ssize_t len = readlink(proc_path, filename, sizeof(filename) - 1);
            if (len != -1) {
                filename[len] = '\0';
                close(fd[i]);
                unlink(filename);
            }
        }
        if (vram_addr[i]) cudaFree(vram_addr[i]);
    }

    delete[] vram_addr;
    delete[] fd;
    delete[] vram_buf;
    delete[] ftrans;

    return ret == NIXL_SUCCESS;
}

bool run_atomic_status_test(const std::string& dir_path, int num_entries, size_t transfer_size, size_t num_threads) {
    std::cout << "\n=== Test 2: Thread-safe atomic status tracking (TaskFlow parallel execution) ===" << std::endl;

    nixl_status_t ret = NIXL_SUCCESS;
    void **vram_addr = new void*[num_entries];
    int *fd = new int[num_entries];

    nixlAgentConfig cfg(true);
    nixl_b_params_t params;
    nixlBlobDesc *vram_buf = new nixlBlobDesc[num_entries];
    nixlBlobDesc *ftrans = new nixlBlobDesc[num_entries];
    nixlBackendH *gds_mt;
    nixl_reg_dlist_t vram_for_gds_mt(VRAM_SEG);
    nixl_reg_dlist_t file_for_gds_mt(FILE_SEG);

    nixlAgent agent("TestAtomicStatus", cfg);

    if (num_threads > 0) {
        params["thread_count"] = std::to_string(num_threads);
    }

    ret = agent.createBackend("GDS_MT", params, gds_mt);
    if (ret != NIXL_SUCCESS || gds_mt == NULL) {
        std::cerr << "Failed to create GDS_MT backend" << std::endl;
        goto cleanup;
    }

    // Setup buffers and files
    for (int i = 0; i < num_entries; i++) {
        if (cudaMalloc(&vram_addr[i], transfer_size) != cudaSuccess) {
            std::cerr << "CUDA malloc failed" << std::endl;
            goto cleanup;
        }
        if (fill_gpu_test_pattern(vram_addr[i], transfer_size) != cudaSuccess) {
            std::cerr << "CUDA buffer initialization failed" << std::endl;
            goto cleanup;
        }

        std::string filename = dir_path + "/" + generate_test_filename("test_gds_mt_atomic", i);
        fd[i] = open(filename.c_str(), O_RDWR|O_CREAT, 0744);
        if (fd[i] < 0) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            goto cleanup;
        }

        vram_buf[i].addr = (uintptr_t)(vram_addr[i]);
        vram_buf[i].len = transfer_size;
        vram_buf[i].devId = 0;
        vram_for_gds_mt.addDesc(vram_buf[i]);

        ftrans[i].addr = 0;
        ftrans[i].len = transfer_size;
        ftrans[i].devId = fd[i];
        file_for_gds_mt.addDesc(ftrans[i]);
    }

    ret = agent.registerMem(file_for_gds_mt);
    if (ret != NIXL_SUCCESS) goto cleanup;

    ret = agent.registerMem(vram_for_gds_mt);
    if (ret != NIXL_SUCCESS) goto cleanup;

    {
        nixl_xfer_dlist_t vram_list = vram_for_gds_mt.trim();
        nixl_xfer_dlist_t file_list = file_for_gds_mt.trim();
        nixlXferReqH* write_req = nullptr;

        ret = agent.createXferReq(NIXL_WRITE, vram_list, file_list, "TestAtomicStatus", write_req);
        if (ret != NIXL_SUCCESS) goto cleanup;

        int status = agent.postXferReq(write_req);
        if (status < 0) {
            agent.releaseXferReq(write_req);
            goto cleanup;
        }

        // Poll for status and verify atomic updates
        std::vector<nixl_status_t> entry_status;
        bool seen_in_progress = false;
        int poll_count = 0;

        while (status == NIXL_IN_PROG && poll_count < 100) {
            status = agent.getXferStatus(write_req, entry_status);

            // Verify status consistency - each entry should be either NIXL_IN_PROG or NIXL_SUCCESS
            for (size_t i = 0; i < entry_status.size(); i++) {
                if (entry_status[i] == NIXL_IN_PROG) {
                    seen_in_progress = true;
                } else if (entry_status[i] != NIXL_SUCCESS && entry_status[i] >= 0) {
                    std::cerr << "Unexpected status " << entry_status[i] << " for entry " << i << std::endl;
                    agent.releaseXferReq(write_req);
                    goto cleanup;
                }
            }

            poll_count++;
            if (status < 0) {
                agent.releaseXferReq(write_req);
                goto cleanup;
            }
        }

        // Final check - all entries should be NIXL_SUCCESS
        if (entry_status.size() == (size_t)num_entries) {
            for (int i = 0; i < num_entries; i++) {
                if (entry_status[i] != NIXL_SUCCESS) {
                    std::cerr << "Entry " << i << " did not complete successfully: status=" << entry_status[i] << std::endl;
                    agent.releaseXferReq(write_req);
                    goto cleanup;
                }
            }
        }

        agent.releaseXferReq(write_req);

        std::cout << "Test 2 PASSED: Atomic status tracking worked correctly across "
                  << num_threads << " threads (polled " << poll_count << " times)" << std::endl;
    }

cleanup:
    agent.deregisterMem(file_for_gds_mt);
    agent.deregisterMem(vram_for_gds_mt);

    for (int i = 0; i < num_entries; i++) {
        if (fd[i] > 0) {
            char proc_path[64];
            char filename[PATH_MAX];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd[i]);
            ssize_t len = readlink(proc_path, filename, sizeof(filename) - 1);
            if (len != -1) {
                filename[len] = '\0';
                close(fd[i]);
                unlink(filename);
            }
        }
        if (vram_addr[i]) cudaFree(vram_addr[i]);
    }

    delete[] vram_addr;
    delete[] fd;
    delete[] vram_buf;
    delete[] ftrans;

    return ret == NIXL_SUCCESS;
}

int main(int argc, char *argv[])
{
    int num_entries = DEFAULT_NUM_ENTRIES;
    size_t transfer_size = DEFAULT_TRANSFER_SIZE;
    size_t num_threads = 0;  // 0 means auto-detect
    std::string dir_path;
    int opt;

    static struct option long_options[] = {
        {"num-entries",  required_argument, 0, 'n'},
        {"size",         required_argument, 0, 's'},
        {"num-threads",  required_argument, 0, 't'},
        {"help",         no_argument,       0, 'h'},
        {0,              0,                 0,  0}
    };

    while ((opt = getopt_long(argc, argv, "n:s:t:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                num_entries = atoi(optarg);
                if (num_entries <= 0) {
                    std::cerr << "Error: Number of entries must be positive\n";
                    return 1;
                }
                break;
            case 's':
                transfer_size = parse_size(optarg);
                if (transfer_size == 0) {
                    std::cerr << "Error: Invalid transfer size format\n";
                    return 1;
                }
                break;
            case 't':
                num_threads = atoi(optarg);
                if (num_threads <= 0) {
                    std::cerr << "Error: Number of threads must be positive\n";
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (optind >= argc) {
        std::cerr << "Error: Directory path is required\n";
        print_usage(argv[0]);
        return 1;
    }
    dir_path = argv[optind];

    std::cout << "\n============================================================" << std::endl;
    std::cout << "        NIXL GDS MT checkXferList Unit Tests               " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Number of entries: " << num_entries << std::endl;
    std::cout << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    std::cout << "- Thread count: " << (num_threads > 0 ? std::to_string(num_threads) : "auto-detect") << std::endl;
    std::cout << "- Directory: " << dir_path << std::endl;
    std::cout << "============================================================\n" << std::endl;

    bool test1_passed = run_parallel_batch_test(dir_path, num_entries, transfer_size, num_threads);
    bool test2_passed = run_atomic_status_test(dir_path, num_entries, transfer_size, num_threads);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "                    TEST SUMMARY                            " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Test 1 (Parallel batch success): " << (test1_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Test 2 (Atomic status tracking): " << (test2_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================" << std::endl;

    return (test1_passed && test2_passed) ? 0 : 1;
}
