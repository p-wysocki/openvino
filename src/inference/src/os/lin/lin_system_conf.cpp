// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sched.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "ie_common.h"
#include "openvino/core/except.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "streams_executor.hpp"

namespace ov {

CPU::CPU() {
    std::vector<std::vector<std::string>> system_info_table;

    _num_threads = parallel_get_max_threads();
    auto get_cache_info_linux = [&]() {
        int cpu_index = 0;
        int cache_index = 0;
        int cache_files = 3;

        std::vector<std::string> one_info(cache_files);

        while (1) {
            for (int n = 0; n < cache_files; n++) {
                cache_index = (n == 0) ? n : n + 1;

                std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) + "/cache/index" +
                                         std::to_string(cache_index) + "/shared_cpu_list");
                if (!cache_file.is_open()) {
                    cache_index = -1;
                    break;
                }
                std::string cache_info;
                std::getline(cache_file, cache_info);
                one_info[n] = cache_info;
            }

            if (cache_index == -1) {
                if (cpu_index == 0) {
                    return -1;
                } else {
                    return 0;
                }
            } else {
                system_info_table.push_back(one_info);
                cpu_index++;
            }
        }

        return 0;
    };

    auto get_freq_info_linux = [&]() {
        int cpu_index = 0;
        int cache_index = 0;

        std::vector<std::string> file_name = {"/topology/core_cpus_list",
                                              "/topology/physical_package_id",
                                              "/cpufreq/cpuinfo_max_freq"};
        int num_of_files = file_name.size();
        std::vector<std::string> one_info(num_of_files);

        while (1) {
            for (int n = 0; n < num_of_files; n++) {
                cache_index = n;

                std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) + file_name[n]);
                if (!cache_file.is_open()) {
                    cache_index = -1;
                    break;
                }
                std::string cache_info;
                std::getline(cache_file, cache_info);
                one_info[n] = cache_info;
            }

            if (cache_index == -1) {
                if (cpu_index == 0) {
                    return -1;
                } else {
                    return 0;
                }
            } else {
                system_info_table.push_back(one_info);
                cpu_index++;
            }
        }

        return 0;
    };

    auto check_valid_cpu = [&]() {
        cpu_set_t mask;
        CPU_ZERO(&mask);

        if ((_processors == 0) || (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1)) {
            return -1;
        }

        int total_proc = 0;
        std::vector<int> cores_list;
        std::vector<int> phy_core_list;
        std::vector<std::vector<int>> valid_cpu_mapping_table;

        for (int i = 0; i < _processors; i++) {
            if (CPU_ISSET(i, &mask)) {
                total_proc++;
                cores_list.emplace_back(_cpu_mapping_table[i][CPU_MAP_CORE_ID]);
                valid_cpu_mapping_table.emplace_back(_cpu_mapping_table[i]);
                if (_cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == MAIN_CORE_PROC) {
                    phy_core_list.emplace_back(_cpu_mapping_table[i][CPU_MAP_GROUP_ID]);
                }
            }
        }

        if (total_proc == 0) {
            return -1;
        } else if (total_proc == _processors) {
            return 0;
        } else {
            _processors = total_proc;
            _cpu_mapping_table.swap(valid_cpu_mapping_table);
            for (auto& row : _proc_type_table) {
                std::fill(row.begin(), row.end(), 0);
            }
            for (auto& row : _cpu_mapping_table) {
                if (row[CPU_MAP_CORE_TYPE] == HYPER_THREADING_PROC) {
                    auto iter = std::find(phy_core_list.begin(), phy_core_list.end(), row[CPU_MAP_GROUP_ID]);
                    if (iter == phy_core_list.end()) {
                        row[CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                    }
                }
                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[0][row[CPU_MAP_CORE_TYPE]]++;
                if (_proc_type_table.size() > 1) {
                    _proc_type_table[row[CPU_MAP_SOCKET_ID] + 1][ALL_PROC]++;
                    _proc_type_table[row[CPU_MAP_SOCKET_ID] + 1][row[CPU_MAP_CORE_TYPE]]++;
                }
            }

            if (_proc_type_table.size() > 1) {
                size_t n = _proc_type_table.size();

                while (n > 0) {
                    if (0 == _proc_type_table[n - 1][ALL_PROC]) {
                        _proc_type_table.erase(_proc_type_table.begin() + n - 1);
                    }
                    n--;
                }

                if ((_proc_type_table.size() > 1) && (_proc_type_table[0][ALL_PROC] == _proc_type_table[1][ALL_PROC])) {
                    _proc_type_table.pop_back();
                }
            }
            _numa_nodes = _proc_type_table.size() == 1 ? 1 : _proc_type_table.size() - 1;
            std::sort(cores_list.begin(), cores_list.end());
            auto iter = std::unique(cores_list.begin(), cores_list.end());
            cores_list.erase(iter, cores_list.end());
            _cores = cores_list.size();
            return 0;
        }
    };

    if (!get_cache_info_linux()) {
        parse_cache_info_linux(system_info_table,
                               _processors,
                               _numa_nodes,
                               _cores,
                               _proc_type_table,
                               _cpu_mapping_table);
    }

    if ((_proc_type_table.size() == 0) || (_proc_type_table[0][MAIN_CORE_PROC] == 0)) {
        if (!get_freq_info_linux()) {
            parse_freq_info_linux(system_info_table,
                                  _processors,
                                  _numa_nodes,
                                  _cores,
                                  _proc_type_table,
                                  _cpu_mapping_table);
        }
    }

    if ((_proc_type_table.size() == 0) || (_proc_type_table[0][MAIN_CORE_PROC] == 0)) {
        /*Previous CPU resource based on calculation*/
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::vector<int> processors;
        std::map<int, int> sockets;
        int socketId = 0;
        while (!cpuinfo.eof()) {
            std::string line;
            std::getline(cpuinfo, line);
            if (line.empty())
                continue;
            auto delimeter = line.find(':');
            auto key = line.substr(0, delimeter);
            auto value = line.substr(delimeter + 1);
            if (0 == key.find("processor")) {
                processors.emplace_back(std::stoi(value));
            }
            if (0 == key.find("physical id")) {
                socketId = std::stoi(value);
            }
            if (0 == key.find("cpu cores")) {
                sockets[socketId] = std::stoi(value);
            }
        }
        _processors = processors.size();
        _numa_nodes = sockets.size();
        for (auto&& socket : sockets) {
            _cores += socket.second;
        }
        if (_cores == 0) {
            _cores = _processors;
        }
    } else {
        if (check_valid_cpu() < 0) {
            OPENVINO_THROW("CPU affinity check failed. No CPU is eligible to run inference.");
        };
    }
    std::vector<std::vector<std::string>>().swap(system_info_table);
}

void parse_cache_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                            int& _processors,
                            int& _sockets,
                            int& _cores,
                            std::vector<std::vector<int>>& _proc_type_table,
                            std::vector<std::vector<int>>& _cpu_mapping_table) {
    int n_group = 0;

    _processors = system_info_table.size();
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    auto update_proc_map_info = [&](const int nproc) {
        if (-1 == _cpu_mapping_table[nproc][CPU_MAP_CORE_ID]) {
            int core_1 = 0;
            int core_2 = 0;
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
            std::string sub_str = "";

            if (((endpos = system_info_table[nproc][0].find(',', pos)) != std::string::npos) ||
                ((endpos = system_info_table[nproc][0].find('-', pos)) != std::string::npos)) {
                sub_str = system_info_table[nproc][0].substr(pos, endpos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][0].substr(endpos + 1);
                core_2 = std::stoi(sub_str);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_2][CPU_MAP_PROCESSOR_ID] = core_2;

                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_2][CPU_MAP_CORE_ID] = _cores;

                /**
                 * Processor 0 need to handle system interception on Linux. So use second processor as physical core
                 * and first processor as logic core
                 */
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                _cpu_mapping_table[core_2][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;

                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;
                _cpu_mapping_table[core_2][CPU_MAP_GROUP_ID] = n_group;

                _cores++;
                n_group++;

                _proc_type_table[0][ALL_PROC] += 2;
                _proc_type_table[0][MAIN_CORE_PROC]++;
                _proc_type_table[0][HYPER_THREADING_PROC]++;
            } else if ((endpos = system_info_table[nproc][1].find('-', pos)) != std::string::npos) {
                sub_str = system_info_table[nproc][1].substr(pos, endpos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][1].substr(endpos + 1);
                core_2 = std::stoi(sub_str);

                for (int m = core_1; m <= core_2; m++) {
                    _cpu_mapping_table[m][CPU_MAP_PROCESSOR_ID] = m;
                    _cpu_mapping_table[m][CPU_MAP_CORE_ID] = _cores;
                    _cpu_mapping_table[m][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                    _cpu_mapping_table[m][CPU_MAP_GROUP_ID] = n_group;

                    _cores++;

                    _proc_type_table[0][ALL_PROC]++;
                    _proc_type_table[0][EFFICIENT_CORE_PROC]++;
                }

                n_group++;
            } else {
                core_1 = std::stoi(system_info_table[nproc][0]);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;

                _cores++;
                n_group++;

                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[0][MAIN_CORE_PROC]++;
            }
        }
        return;
    };

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    for (int n = 0; n < _processors; n++) {
        if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
            std::string sub_str;

            int core_1;
            int core_2;

            if (0 == _sockets) {
                _proc_type_table.push_back(line_value_0);
            } else {
                _proc_type_table.push_back(_proc_type_table[0]);
                _proc_type_table[0] = line_value_0;
            }

            while (1) {
                if ((endpos = system_info_table[n][2].find('-', pos)) != std::string::npos) {
                    sub_str = system_info_table[n][2].substr(pos, endpos);
                    core_1 = std::stoi(sub_str);
                    sub_str = system_info_table[n][2].substr(endpos + 1);
                    core_2 = std::stoi(sub_str);

                    for (int m = core_1; m <= core_2; m++) {
                        _cpu_mapping_table[m][CPU_MAP_SOCKET_ID] = _sockets;
                        update_proc_map_info(m);
                    }
                } else if (pos != std::string::npos) {
                    sub_str = system_info_table[n][2].substr(pos);
                    core_1 = std::stoi(sub_str);
                    _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = _sockets;
                    update_proc_map_info(core_1);
                    endpos = pos;
                }

                if ((pos = system_info_table[n][2].find(',', endpos)) != std::string::npos) {
                    pos++;
                } else {
                    break;
                }
            }
            _sockets++;
        }
    }
    if (_sockets > 1) {
        _proc_type_table.push_back(_proc_type_table[0]);
        _proc_type_table[0] = line_value_0;

        for (int m = 1; m <= _sockets; m++) {
            for (int n = 0; n < PROC_TYPE_TABLE_SIZE; n++) {
                _proc_type_table[0][n] += _proc_type_table[m][n];
            }
        }
    }
};

void parse_freq_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                           int& _processors,
                           int& _sockets,
                           int& _cores,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table) {
    int freq_max = 0;
    bool ecore_enabled = false;
    bool ht_enabled = false;

    _processors = system_info_table.size();
    _sockets = 0;
    _cores = 0;
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    for (int n = 0; n < _processors; n++) {
        if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
            std::string::size_type pos = 0;
            std::string::size_type endpos1 = 0;
            std::string::size_type endpos2 = 0;
            std::string sub_str;

            int core_1 = 0;
            int core_2 = 0;

            if (((endpos1 = system_info_table[n][0].find(',', pos)) != std::string::npos) ||
                ((endpos2 = system_info_table[n][0].find('-', pos)) != std::string::npos)) {
                endpos1 = (endpos1 != std::string::npos) ? endpos1 : endpos2;
                sub_str = system_info_table[n][0].substr(pos, endpos1);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[n][0].substr(endpos1 + 1);
                core_2 = std::stoi(sub_str);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = std::stoi(system_info_table[core_1][1]);
                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = _cores;

                _cpu_mapping_table[core_2][CPU_MAP_PROCESSOR_ID] = core_2;
                _cpu_mapping_table[core_2][CPU_MAP_SOCKET_ID] = _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                _cpu_mapping_table[core_2][CPU_MAP_CORE_ID] = _cpu_mapping_table[core_1][CPU_MAP_CORE_ID];
                _cpu_mapping_table[core_2][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cpu_mapping_table[core_2][CPU_MAP_GROUP_ID] = _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID];

                ht_enabled = true;
                int core_freq = std::stoi(system_info_table[core_1][2]);
                freq_max = std::max(core_freq, freq_max);

            } else if (system_info_table[n][0].size() > 0) {
                core_1 = std::stoi(system_info_table[n][0]);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = std::stoi(system_info_table[core_1][1]);
                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;

                int core_freq = std::stoi(system_info_table[core_1][2]);
                if (((0 == freq_max) || (core_freq >= freq_max * 0.95)) && (!ht_enabled)) {
                    freq_max = std::max(core_freq, freq_max);
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                } else {
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                    ecore_enabled = true;
                }

                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = _cores;
            }
            _sockets = std::max(_sockets, _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID]);
            _cores++;
        }
    }

    if ((_sockets >= 1) && (ecore_enabled)) {
        _sockets = 0;
    }

    if (_sockets >= 1) {
        _proc_type_table.resize(_sockets + 2, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
        for (int n = 0; n < _processors; n++) {
            _proc_type_table[0][ALL_PROC]++;
            _proc_type_table[_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] + 1][ALL_PROC]++;

            _proc_type_table[0][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
            _proc_type_table[_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] + 1][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
        }
        _sockets++;
    } else {
        _proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
        for (int n = 0; n < _processors; n++) {
            _proc_type_table[0][ALL_PROC]++;
            _proc_type_table[0][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
            _cpu_mapping_table[n][CPU_MAP_SOCKET_ID] = 0;
        }
        _sockets = 1;
    }
};

}  // namespace ov
