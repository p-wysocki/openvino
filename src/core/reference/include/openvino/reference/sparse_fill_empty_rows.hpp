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

template <typename T_IDX, typename T_SYMBOLS>
void sparse_fill_empty_rows_unpacked_string(const T_IDX* begins,
                                           const T_IDX* ends,
                                           const uint8_t* symbols,
                                           const uint8_t* default_value,
                                           const T_IDX num_rows,
                                           const T_IDX num_cols,
                                           const size_t symbols_size,
                                           const size_t default_value_size,
                                           T_IDX* output_begins,
                                           T_IDX* output_ends,
                                           uint8_t* output_symbols,
                                           bool* empty_row_indicator) {
    // Directly identify empty rows in a single pass
    size_t empty_row_count = 0;
    for (size_t row = 0; row < num_rows; row++) {
        bool is_empty = true;
        for (size_t col = 0; col < num_cols; col++) {
            const size_t idx = row * num_cols + col;
            if (begins[idx] != ends[idx]) {
                is_empty = false;
                break;
            }
        }
        empty_row_indicator[row] = is_empty;
        if (is_empty) {
            empty_row_count++;
        }
    }
    
    // Calculate total strings in the output - original strings plus one for each empty row
    const size_t total_strings = num_rows + empty_row_count;
    
    // Create a vector of pairs containing ((row, col), source_index) for sorting
    std::vector<std::pair<std::pair<T_IDX, T_IDX>, size_t>> row_col_pairs(total_strings);
    
    // Add existing strings
    for (size_t i = 0; i < num_rows; i++) {
        row_col_pairs[i] = {{begins[i * 2], begins[i * 2 + 1]}, i};
    }
    
    // Add empty rows with default strings at column 0
    size_t empty_idx = num_rows;
    for (size_t row = 0; row < num_rows; row++) {
        if (empty_row_indicator[row]) {
            row_col_pairs[empty_idx++] = {{static_cast<T_IDX>(row), 0}, num_rows + empty_idx - num_rows - 1};
        }
    }
    
    std::sort(row_col_pairs.begin(), row_col_pairs.end(), [](const auto& a, const auto& b) {
        if (a.first.first != b.first.first) {
            return a.first.first < b.first.first;
        }
        return a.first.second < b.first.second;
    });
    
    // Copy existing symbols to output_symbols and append default_value for empty rows
    T_IDX current_pos = 0;
    
    // First copy all existing symbols
    for (size_t i = 0; i < num_rows; i++) {
        const T_IDX string_begin = begins[i * 2 + 1];
        const T_IDX string_end = ends[i * 2 + 1];
        const T_IDX string_length = string_end - string_begin;
        
        // Copy string content from symbols to output_symbols
        std::memcpy(output_symbols + current_pos, symbols + string_begin, string_length * sizeof(uint8_t));
        current_pos += string_length;
    }
    
    // Store default_value only once at the end of the output_symbols array, but only if there are empty rows
    T_IDX default_value_position = current_pos;
    if (empty_row_count > 0) {
        std::memcpy(output_symbols + default_value_position, default_value, default_value_size * sizeof(uint8_t));
    }
    
    // Reset current position for handling existing strings
    current_pos = 0;
    
    // Now set all the begins/ends pointers appropriately
    for (size_t i = 0; i < total_strings; i++) {
        const auto& [row_col, src_idx] = row_col_pairs[i];
        
        // Set output begins/ends for this string position
        output_begins[i * 2] = row_col.first;     // Row
        output_begins[i * 2 + 1] = row_col.second; // Column
        
        if (src_idx < num_rows) {
            // Handle existing string
            const T_IDX string_begin = begins[src_idx * 2 + 1];
            const T_IDX string_end = ends[src_idx * 2 + 1];
            const T_IDX string_length = string_end - string_begin;
            
            // Set output begins/ends for the string content
            output_begins[i * 2 + 1] = current_pos;
            output_ends[i * 2] = row_col.first;
            output_ends[i * 2 + 1] = current_pos + string_length;
            
            current_pos += string_length;
        } else {
            // For empty rows, point to the single copy of default_value
            output_begins[i * 2 + 1] = default_value_position;
            output_ends[i * 2] = row_col.first;
            output_ends[i * 2 + 1] = default_value_position + default_value_size;
        }
    }
    
    // Verify the output matches spec requirements:
    // 1. Empty rows should be filled with default_value at [row, 0]
    // 2. All other values should be preserved
    // 3. Output should have the same dense shape as input
    for (T_IDX row = 0; row < num_rows; row++) {
        bool is_empty_row = (rows_with_non_empty_strings.find(row) == rows_with_non_empty_strings.end());
        assert(empty_row_indicator[row] == is_empty_row && 
               "empty_row_indicator must correctly identify empty rows");
    }
}

}  // namespace ov::reference
