// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov::reference {

template <typename T, typename T_IDX>
void sparse_fill_empty_rows(const T* values,
                            const size_t values_size,
                            const T_IDX* dense_shape,
                            const T_IDX* indices,
                            const T default_value,
                            T_IDX* output_indices,
                            T* output_values,
                            bool* empty_row_indicator) {
    const auto num_rows = dense_shape[0];

    std::unordered_set<T_IDX> existing_rows;
    for (size_t i = 0, idx = 0; i < values_size; i++, idx += 2) {
        existing_rows.insert(indices[idx]);
    }

    std::vector<T_IDX> empty_rows;
    empty_rows.reserve(num_rows - existing_rows.size());
    for (T_IDX i = 0; i < num_rows; i++) {
        const bool is_empty = (existing_rows.find(i) == existing_rows.end());
        empty_row_indicator[i] = is_empty;
        if (is_empty) {
            empty_rows.push_back(i);
        }
    }

    // Vector of pairs containing ((row, column), source_index) for
    // both existing values and new empty rows to be added
    const size_t total_rows = values_size + empty_rows.size();
    std::vector<std::pair<std::pair<T_IDX, T_IDX>, size_t>> row_col_pairs(total_rows);

    // Add existing values and then empty rows
    for (size_t i = 0, idx = 0; i < values_size; i++, idx += 2) {
        row_col_pairs[i] = {{indices[idx], indices[idx + 1]}, i};
    }
    for (size_t i = 0; i < empty_rows.size(); i++) {
        row_col_pairs[values_size + i] = {{empty_rows[i], 0}, values_size + i};
    }

    std::sort(row_col_pairs.begin(), row_col_pairs.end(), [](const auto& a, const auto& b) {
        if (a.first.first != b.first.first) {
            return a.first.first < b.first.first;
        }
        return a.first.second < b.first.second;
    });

    for (size_t i = 0, out_idx = 0; i < total_rows; i++, out_idx += 2) {
        const auto& [row_col, src_idx] = row_col_pairs[i];
        const auto& [row, col] = row_col;

        output_indices[out_idx] = row;
        output_indices[out_idx + 1] = col;

        if (src_idx < values_size) {
            output_values[i] = values[src_idx];
        } else {
            output_values[i] = default_value;
        }
    }
}

template <typename T_IDX>
void sparse_fill_empty_rows_unpacked_string(const T_IDX* begins,
                                           const T_IDX* ends,
                                           const uint8_t* symbols,
                                           const T_IDX* indices,
                                           const T_IDX* dense_shape,
                                           const uint8_t* default_value,
                                           const size_t default_value_size,
                                           const size_t num_values,
                                           const size_t symbols_size,
                                           T_IDX* output_begins,
                                           T_IDX* output_ends,
                                           T_IDX* output_indices,
                                           uint8_t* output_symbols,
                                           bool* empty_row_indicator) {
    
    const auto num_rows = dense_shape[0];

    // Find existing rows
    std::unordered_set<T_IDX> existing_rows;
    for (size_t idx = 0; idx < num_values * 2; idx += 2) {
        existing_rows.insert(indices[idx]);
    }
    
    // Find empty rows
    std::vector<T_IDX> empty_rows;
    empty_rows.reserve(num_rows - existing_rows.size());
    for (T_IDX i = 0; i < num_rows; i++) {
        const bool is_empty = (existing_rows.find(i) == existing_rows.end());
        empty_row_indicator[i] = is_empty;
        if (is_empty) {
            empty_rows.push_back(i);
        }
    }
    
    // Vector of pairs containing ((row, column), source_index) for
    // both existing values and new empty rows to be added
    const size_t total_values = num_values + empty_rows.size();
    std::vector<std::pair<std::pair<T_IDX, T_IDX>, size_t>> row_col_pairs(total_values);
    
    // Add existing values
    for (size_t i = 0; i < num_values; i++) {
        row_col_pairs[i] = {{indices[i * 2], indices[i * 2 + 1]}, i};
    }
    
    // Add empty rows with column 0
    for (size_t i = 0; i < empty_rows.size(); i++) {
        row_col_pairs[num_values + i] = {{empty_rows[i], 0}, num_values + i};
    }
    
    // Sort by row then column
    std::sort(row_col_pairs.begin(), row_col_pairs.end(), [](const auto& a, const auto& b) {
        if (a.first.first != b.first.first) {
            return a.first.first < b.first.first;
        }
        return a.first.second < b.first.second;
    });
    
    // Copy the input symbols
    std::copy(symbols, symbols + symbols_size, output_symbols);
    
    // If we have empty rows, add default_value at the end of symbols once
    T_IDX default_value_offset = symbols_size;
    if (!empty_rows.empty()) {
        std::copy(default_value, 
                  default_value + default_value_size, 
                  output_symbols + symbols_size);
    }
    
    // Process all values in sorted order
    for (size_t i = 0; i < total_values; i++) {
        const auto& [row_col, src_idx] = row_col_pairs[i];
        
        // Set output indices for this entry
        output_indices[i * 2] = row_col.first;      // row
        output_indices[i * 2 + 1] = row_col.second; // column
        
        if (src_idx < num_values) {
            // Copy existing string's begins/ends
            output_begins[i] = begins[src_idx];
            output_ends[i] = ends[src_idx];
        } else {
            // For empty rows, point to the default value at the end
            output_begins[i] = default_value_offset;
            output_ends[i] = default_value_offset + default_value_size;
        }
    }

    // Print outputs to cout
    std::cout << "output_indices: ";
    for (size_t i = 0; i < total_values * 2; ++i) {
        std::cout << output_indices[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "output_begins: ";
    for (size_t i = 0; i < total_values; ++i) {
        std::cout << output_begins[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "output_ends: ";
    for (size_t i = 0; i < total_values; ++i) {
        std::cout << output_ends[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "output_symbols: ";
    size_t total_symbols = symbols_size + (empty_rows.empty() ? 0 : default_value_size);
    for (size_t i = 0; i < total_symbols; ++i) {
        std::cout << static_cast<char>(output_symbols[i]);
    }
    std::cout << std::endl;

    std::cout << "empty_row_indicator: ";
    for (T_IDX i = 0; i < num_rows; ++i) {
        std::cout << empty_row_indicator[i] << " ";
    }
    std::cout << std::endl;
}

}  // namespace ov::reference
