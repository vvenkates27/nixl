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
#include <cstring>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <absl/strings/str_format.h>
#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"
#include "common/nixl_time.h"

// Test configuration
#define DEFAULT_NUM_ENTRIES 4
#define DEFAULT_TRANSFER_SIZE (1 * 1024 * 1024)  // 1MB
#define TEST_PHRASE "|NIXL bdev 32[b] GUSLI pattern |"
#define TEST_PHRASE_LEN (sizeof(TEST_PHRASE) - 1)

static std::ostream &err_log = std::cerr;
static std::ostream &out_log = std::cout;

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -n, --num-entries N     Number of entries in batch (default: " << DEFAULT_NUM_ENTRIES << ")\n"
              << "  -s, --size SIZE         Size of each transfer (default: " << DEFAULT_TRANSFER_SIZE << " bytes)\n"
              << "  -c, --config FILE       GUSLI config file path\n"
              << "  -d, --device UUID       Block device UUID (decimal, e.g., 27)\n"
              << "  -h, --help              Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " -n 4 -s 1048576 -c /path/to/gusli.conf -d 27\n";
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

bool verify_test_pattern(const void* buffer, size_t size) {
    const char* buf = (const char*)buffer;
    size_t phrase_len = TEST_PHRASE_LEN;
    size_t offset = 0;
    while (offset < size) {
        size_t remaining = size - offset;
        size_t copy_len = (remaining < phrase_len) ? remaining : phrase_len;
        if (memcmp(buf + offset, TEST_PHRASE, copy_len) != 0) {
            err_log << "Pattern mismatch at offset " << offset << std::endl;
            return false;
        }
        offset += copy_len;
    }
    return true;
}

bool run_single_device_test(const std::string& config_file, uint64_t devId, size_t transfer_size) {
    out_log << "\n=== Test 1: Single-device I/O per-entry status ===" << std::endl;

    nixl_status_t ret = NIXL_SUCCESS;
    void* dram_buf = nullptr;
    nixlBackendH* gusli = nullptr;
    nixlXferReqH* write_req = nullptr;
    nixlXferReqH* read_req = nullptr;
    bool test_passed = false;

    nixlAgentConfig cfg(true);
    nixl_b_params_t params;

    // Initialize agent
    nixlAgent agent("TestGUSLISingle", cfg);

    // Configure GUSLI backend
    if (!config_file.empty()) {
        params["config_file"] = config_file;
    }
    params["client_name"] = "nixl_test_checkXferList";

    ret = agent.createBackend("GUSLI", params, gusli);
    if (ret != NIXL_SUCCESS || gusli == nullptr) {
        err_log << "Failed to create GUSLI backend: " << ret << std::endl;
        goto cleanup;
    }

    // Allocate aligned DRAM buffer
    long page_size = sysconf(_SC_PAGESIZE);
    if (posix_memalign(&dram_buf, page_size, transfer_size) != 0) {
        err_log << "Failed to allocate DRAM buffer" << std::endl;
        goto cleanup;
    }

    // Fill buffer with test pattern
    fill_test_pattern(dram_buf, transfer_size);

    // Setup descriptors
    {
        nixl_reg_dlist_t dram_list(DRAM_SEG);
        nixl_reg_dlist_t bdev_list(BLK_SEG);

        nixlBlobDesc dram_desc;
        dram_desc.addr = (uintptr_t)dram_buf;
        dram_desc.len = transfer_size;
        dram_desc.devId = 0;
        dram_list.addDesc(dram_desc);

        nixlBlobDesc bdev_desc;
        bdev_desc.addr = (1UL << 20);  // 1MB offset
        bdev_desc.len = transfer_size;
        bdev_desc.devId = devId;
        bdev_list.addDesc(bdev_desc);

        // Register memory
        ret = agent.registerMem(dram_list);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to register DRAM memory: " << ret << std::endl;
            goto cleanup;
        }

        ret = agent.registerMem(bdev_list);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to register block device memory: " << ret << std::endl;
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Create and execute write transfer
        nixl_xfer_dlist_t dram_xfer = dram_list.trim();
        nixl_xfer_dlist_t bdev_xfer = bdev_list.trim();

        ret = agent.createXferReq(NIXL_WRITE, dram_xfer, bdev_xfer, "TestGUSLISingle", write_req);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to create write transfer request: " << ret << std::endl;
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Post transfer
        int status = agent.postXferReq(write_req);
        if (status < 0) {
            err_log << "Failed to post write transfer request: " << status << std::endl;
            agent.releaseXferReq(write_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Check per-entry status
        std::vector<nixl_status_t> entry_status;
        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(write_req, entry_status);
            if (status < 0) {
                err_log << "Error during write transfer: " << status << std::endl;
                agent.releaseXferReq(write_req);
                agent.deregisterMem(bdev_list);
                agent.deregisterMem(dram_list);
                goto cleanup;
            }
        }

        // Verify single entry status
        if (entry_status.size() != 1) {
            err_log << "Expected 1 status entry, got " << entry_status.size() << std::endl;
            agent.releaseXferReq(write_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        if (entry_status[0] != NIXL_SUCCESS) {
            err_log << "Write entry failed with status: " << entry_status[0] << std::endl;
            agent.releaseXferReq(write_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        agent.releaseXferReq(write_req);
        write_req = nullptr;

        // Verify read-back
        memset(dram_buf, 0, transfer_size);

        ret = agent.createXferReq(NIXL_READ, dram_xfer, bdev_xfer, "TestGUSLISingle", read_req);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to create read transfer request: " << ret << std::endl;
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        status = agent.postXferReq(read_req);
        if (status < 0) {
            err_log << "Failed to post read transfer request: " << status << std::endl;
            agent.releaseXferReq(read_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        entry_status.clear();
        while (status == NIXL_IN_PROG) {
            status = agent.getXferStatus(read_req, entry_status);
            if (status < 0) {
                err_log << "Error during read transfer: " << status << std::endl;
                agent.releaseXferReq(read_req);
                agent.deregisterMem(bdev_list);
                agent.deregisterMem(dram_list);
                goto cleanup;
            }
        }

        if (entry_status.size() != 1 || entry_status[0] != NIXL_SUCCESS) {
            err_log << "Read entry status check failed" << std::endl;
            agent.releaseXferReq(read_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        agent.releaseXferReq(read_req);
        read_req = nullptr;

        // Verify data
        if (!verify_test_pattern(dram_buf, transfer_size)) {
            err_log << "Data verification failed" << std::endl;
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        agent.deregisterMem(bdev_list);
        agent.deregisterMem(dram_list);
    }

    out_log << "Test 1 PASSED: Single-device I/O per-entry status works correctly" << std::endl;
    test_passed = true;

cleanup:
    if (write_req) agent.releaseXferReq(write_req);
    if (read_req) agent.releaseXferReq(read_req);
    if (dram_buf) free(dram_buf);

    return test_passed;
}

bool run_compound_test(const std::string& config_file, uint64_t devId, int num_entries, size_t transfer_size) {
    out_log << "\n=== Test 2: Compound I/O with multiple entries ===" << std::endl;

    nixl_status_t ret = NIXL_SUCCESS;
    void** dram_bufs = nullptr;
    nixlBackendH* gusli = nullptr;
    nixlXferReqH* write_req = nullptr;
    bool test_passed = false;

    nixlAgentConfig cfg(true);
    nixl_b_params_t params;

    // Initialize agent
    nixlAgent agent("TestGUSLICompound", cfg);

    // Configure GUSLI backend
    if (!config_file.empty()) {
        params["config_file"] = config_file;
    }
    params["client_name"] = "nixl_test_checkXferList_compound";

    ret = agent.createBackend("GUSLI", params, gusli);
    if (ret != NIXL_SUCCESS || gusli == nullptr) {
        err_log << "Failed to create GUSLI backend: " << ret << std::endl;
        goto cleanup;
    }

    // Allocate DRAM buffers
    long page_size = sysconf(_SC_PAGESIZE);
    dram_bufs = new void*[num_entries];
    for (int i = 0; i < num_entries; i++) {
        dram_bufs[i] = nullptr;
        if (posix_memalign(&dram_bufs[i], page_size, transfer_size) != 0) {
            err_log << "Failed to allocate DRAM buffer " << i << std::endl;
            goto cleanup;
        }
        fill_test_pattern(dram_bufs[i], transfer_size);
    }

    // Setup descriptors
    {
        nixl_reg_dlist_t dram_list(DRAM_SEG);
        nixl_reg_dlist_t bdev_list(BLK_SEG);

        for (int i = 0; i < num_entries; i++) {
            nixlBlobDesc dram_desc;
            dram_desc.addr = (uintptr_t)dram_bufs[i];
            dram_desc.len = transfer_size;
            dram_desc.devId = 0;
            dram_list.addDesc(dram_desc);

            nixlBlobDesc bdev_desc;
            bdev_desc.addr = (1UL << 20) + (i * transfer_size);  // Offset each entry
            bdev_desc.len = transfer_size;
            bdev_desc.devId = devId;
            bdev_list.addDesc(bdev_desc);
        }

        // Register memory
        ret = agent.registerMem(dram_list);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to register DRAM memory: " << ret << std::endl;
            goto cleanup;
        }

        ret = agent.registerMem(bdev_list);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to register block device memory: " << ret << std::endl;
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Create and execute write transfer
        nixl_xfer_dlist_t dram_xfer = dram_list.trim();
        nixl_xfer_dlist_t bdev_xfer = bdev_list.trim();

        ret = agent.createXferReq(NIXL_WRITE, dram_xfer, bdev_xfer, "TestGUSLICompound", write_req);
        if (ret != NIXL_SUCCESS) {
            err_log << "Failed to create write transfer request: " << ret << std::endl;
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Post transfer
        int status = agent.postXferReq(write_req);
        if (status < 0) {
            err_log << "Failed to post write transfer request: " << status << std::endl;
            agent.releaseXferReq(write_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Check per-entry status
        std::vector<nixl_status_t> entry_status;
        int poll_count = 0;
        while (status == NIXL_IN_PROG && poll_count < 1000) {
            status = agent.getXferStatus(write_req, entry_status);
            if (status < 0) {
                err_log << "Error during write transfer: " << status << std::endl;
                agent.releaseXferReq(write_req);
                agent.deregisterMem(bdev_list);
                agent.deregisterMem(dram_list);
                goto cleanup;
            }
            poll_count++;
        }

        // Verify entry count matches
        if (entry_status.size() != (size_t)num_entries) {
            err_log << "Expected " << num_entries << " status entries, got "
                      << entry_status.size() << std::endl;
            agent.releaseXferReq(write_req);
            agent.deregisterMem(bdev_list);
            agent.deregisterMem(dram_list);
            goto cleanup;
        }

        // Verify all entries succeeded
        bool all_success = true;
        for (int i = 0; i < num_entries; i++) {
            if (entry_status[i] != NIXL_SUCCESS) {
                err_log << "Entry " << i << " failed with status: " << entry_status[i] << std::endl;
                all_success = false;
            }
        }

        agent.releaseXferReq(write_req);
        agent.deregisterMem(bdev_list);
        agent.deregisterMem(dram_list);

        if (!all_success) {
            goto cleanup;
        }
    }

    out_log << "Test 2 PASSED: Compound I/O with " << num_entries
              << " entries completed successfully" << std::endl;
    test_passed = true;

cleanup:
    if (write_req) agent.releaseXferReq(write_req);
    if (dram_bufs) {
        for (int i = 0; i < num_entries; i++) {
            if (dram_bufs[i]) free(dram_bufs[i]);
        }
        delete[] dram_bufs;
    }

    return test_passed;
}

int main(int argc, char *argv[])
{
    int num_entries = DEFAULT_NUM_ENTRIES;
    size_t transfer_size = DEFAULT_TRANSFER_SIZE;
    std::string config_file;
    uint64_t devId = 27;  // Default NVME device UUID
    int opt;

    static struct option long_options[] = {
        {"num-entries", required_argument, 0, 'n'},
        {"size",        required_argument, 0, 's'},
        {"config",      required_argument, 0, 'c'},
        {"device",      required_argument, 0, 'd'},
        {"help",        no_argument,       0, 'h'},
        {0,             0,                 0,  0}
    };

    while ((opt = getopt_long(argc, argv, "n:s:c:d:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'n':
                num_entries = atoi(optarg);
                if (num_entries <= 0) {
                    err_log << "Error: Number of entries must be positive\n";
                    return 1;
                }
                break;
            case 's':
                transfer_size = parse_size(optarg);
                if (transfer_size == 0) {
                    err_log << "Error: Invalid transfer size format\n";
                    return 1;
                }
                break;
            case 'c':
                config_file = optarg;
                break;
            case 'd':
                devId = strtoull(optarg, nullptr, 10);
                if (devId == 0) {
                    err_log << "Error: Invalid device UUID\n";
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

    out_log << "\n============================================================" << std::endl;
    out_log << "         NIXL GUSLI checkXferList Unit Tests               " << std::endl;
    out_log << "============================================================" << std::endl;
    out_log << "Configuration:" << std::endl;
    out_log << "- Number of entries: " << num_entries << std::endl;
    out_log << "- Transfer size: " << transfer_size << " bytes" << std::endl;
    out_log << "- Device UUID: " << devId << std::endl;
    if (!config_file.empty()) {
        out_log << "- Config file: " << config_file << std::endl;
    }
    out_log << "============================================================\n" << std::endl;

    bool test1_passed = run_single_device_test(config_file, devId, transfer_size);
    bool test2_passed = run_compound_test(config_file, devId, num_entries, transfer_size);

    out_log << "\n============================================================" << std::endl;
    out_log << "                    TEST SUMMARY                            " << std::endl;
    out_log << "============================================================" << std::endl;
    out_log << "Test 1 (Single-device I/O):    " << (test1_passed ? "PASSED" : "FAILED") << std::endl;
    out_log << "Test 2 (Compound I/O):         " << (test2_passed ? "PASSED" : "FAILED") << std::endl;
    out_log << "============================================================" << std::endl;

    return (test1_passed && test2_passed) ? 0 : 1;
}
