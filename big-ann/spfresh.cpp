/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 25-7-15 下午6:10
 * Description: ${DESCRIPTION}
 */
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"

#include <unordered_set>
#include <chrono>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
#include <string>

// 数据读取函数
template <typename T>
bool ReadVectorFile(const std::string& filename, std::vector<T>& vectors, uint32_t& num_vectors, uint32_t& dimension) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // 读取数据量和维度
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dimension), sizeof(uint32_t));

    std::cout << "File: " << filename << std::endl;
    std::cout << "Number of vectors: " << num_vectors << std::endl;
    std::cout << "Dimension: " << dimension << std::endl;

    // 读取向量数据
    size_t total_elements = static_cast<size_t>(num_vectors) * dimension;
    vectors.resize(total_elements);

    file.read(reinterpret_cast<char*>(vectors.data()), total_elements * sizeof(T));

    if (!file.good()) {
        std::cerr << "Error: Failed to read vector data from " << filename << std::endl;
        return false;
    }

    file.close();
    return true;
}

// SPANN构建函数
template <typename T>
bool BuildSPANN(std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string& output_path) {
    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::SPANN,
        SPTAG::GetEnumValueType<T>()
    );

    if (nullptr == vecIndex) {
        std::cerr << "Error: Failed to create SPANN index instance" << std::endl;
        return false;
    }

    // 设置SPANN参数
    vecIndex->SetParameter("IndexAlgoType", "BKT", "Base");
    vecIndex->SetParameter("DistCalcMethod", "L2", "Base");
    vecIndex->SetParameter("Dim","128" ,"Base");
    vecIndex->SetParameter("ValueType","Float" ,"Base");
    vecIndex->SetParameter("IndexDirectory", "./spann_index", "Base");
    // SelectHead 阶段参
    vecIndex->SetParameter("isExecute", "true", "SelectHead");
    vecIndex->SetParameter("NumberOfThreads", "4", "SelectHead");
    vecIndex->SetParameter("Ratio", "0.2", "SelectHead");

    // BuildHead 阶段参数
    vecIndex->SetParameter("isExecute", "true", "BuildHead");
    vecIndex->SetParameter("RefineIterations", "3", "BuildHead");
    vecIndex->SetParameter("NumberOfThreads", "4", "BuildHead");

    // BuildSSDIndex 阶段参数
    vecIndex->SetParameter("isExecute", "true", "BuildSSDIndex");
    vecIndex->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    vecIndex->SetParameter("NumberOfThreads", "4", "BuildSSDIndex");
    vecIndex->SetParameter("PostingPageLimit", "12", "BuildSSDIndex");
    vecIndex->SetParameter("SearchPostingPageLimit", "12", "BuildSSDIndex");
    vecIndex->SetParameter("InternalResultNum", "64", "BuildSSDIndex");
    vecIndex->SetParameter("SearchInternalResultNum", "64", "BuildSSDIndex");
    vecIndex->SetParameter("TmpDir", "./tmp/", "BuildSSDIndex");
    vecIndex->SetParameter("SearchResult", "result.txt", "BuildSSDIndex");
    vecIndex->SetParameter("SearchInternalResultNum", "32", "BuildSSDIndex");
    vecIndex->SetParameter("SearchPostingPageLimit", "3", "BuildSSDIndex");
    vecIndex->SetParameter("ResultNum", "10", "BuildSSDIndex");
    vecIndex->SetParameter("MaxDistRatio", "8.0", "BuildSSDIndex");


    std::cout << "Building SPANN index..." << std::endl;
    std::cout << "VectorSet: " << (vec ? "valid" : "nullptr") << std::endl;
    std::cout << "MetadataSet: " << (meta ? "valid" : "nullptr") << std::endl;
    std::cout << "VectorSet count = " << vec->Count() << ", dimension = " << vec->Dimension() << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    SPTAG::ErrorCode build_result = vecIndex->BuildIndex(vec, meta);
    if (SPTAG::ErrorCode::Success != build_result) {
        std::cerr << "Error: Failed to build SPANN index, error code: " << static_cast<int>(build_result) << std::endl;
        return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Index building completed in " << duration.count() << " seconds" << std::endl;

    std::cout << "Saving index to " << output_path << std::endl;
    SPTAG::ErrorCode save_result = vecIndex->SaveIndex(output_path);
    if (SPTAG::ErrorCode::Success != save_result) {
        std::cerr << "Error: Failed to save index, error code: " << static_cast<int>(save_result) << std::endl;
        return false;
    }

    std::cout << "Index saved successfully!" << std::endl;
    return true;
}

// 查询函数
template <typename T>
bool SearchIndex(const std::string& index_path, const std::vector<T>& query_vectors,
                 uint32_t num_queries, uint32_t dimension, int k = 10) {
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    SPTAG::ErrorCode load_result = SPTAG::VectorIndex::LoadIndex(index_path, vecIndex);
    if (SPTAG::ErrorCode::Success != load_result || nullptr == vecIndex) {
        std::cerr << "Error: Failed to load index from " << index_path << std::endl;
        return false;
    }

    std::cout << "Performing searches with " << num_queries << " queries..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_queries; i++) {
        const T* query_vec = query_vectors.data() + i * dimension;
        SPTAG::QueryResult res(query_vec, k, true);

        SPTAG::ErrorCode search_result = vecIndex->SearchIndex(res);
        if (SPTAG::ErrorCode::Success != search_result) {
            std::cerr << "Error: Search failed for query " << i << std::endl;
            continue;
        }

        // 打印前几个查询的结果
        if (i < 5) {
            std::cout << "Query " << i << " results: ";
            for (int j = 0; j < k; j++) {
                std::cout << "(" << res.GetResult(j)->VID << "," << res.GetResult(j)->Dist << ") ";
            }
            std::cout << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double avg_time = static_cast<double>(duration.count()) / num_queries;

    std::cout << "Search completed!" << std::endl;
    std::cout << "Average search time: " << avg_time << " ms per query" << std::endl;

    return true;
}

int main() {
    try {
        // 文件路径
        const std::string data_file = "./data/data_1000000_128";
        const std::string query_file = "./data/queries_10000_128";
        const std::string index_output = "./spann_index";  //不要改动

        // 读取数据集
        std::vector<float> data_vectors;
        uint32_t num_data_vectors, data_dimension;

        std::cout << "Reading data file..." << std::endl;
        if (!ReadVectorFile(data_file, data_vectors, num_data_vectors, data_dimension)) {
            std::cerr << "Failed to read data file" << std::endl;
            return -1;
        }

        // 读取查询集
        std::vector<float> query_vectors;
        uint32_t num_query_vectors, query_dimension;

        std::cout << "\nReading query file..." << std::endl;
        if (!ReadVectorFile(query_file, query_vectors, num_query_vectors, query_dimension)) {
            std::cerr << "Failed to read query file" << std::endl;
            return -1;
        }

        // 验证维度一致性
        if (data_dimension != query_dimension) {
            std::cerr << "Error: Data and query dimensions do not match!" << std::endl;
            return -1;
        }

        // 取前50万个数据用于构建索引
        const uint32_t build_count = std::min(500000u, num_data_vectors);
        std::cout << "\nUsing " << build_count << " vectors for index building" << std::endl;

        // 创建VectorSet
        std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(data_vectors.data()),
                           sizeof(float) * build_count * data_dimension, false),
            SPTAG::VectorValueType::Float,
            data_dimension,
            build_count
        ));

        // 创建简单的MetadataSet（使用向量ID作为metadata）
        std::vector<char> meta_data;
        std::vector<std::uint64_t> meta_offsets;

        for (uint32_t i = 0; i < build_count; i++) {
            meta_offsets.push_back(static_cast<std::uint64_t>(meta_data.size()));
            std::string meta_str = std::to_string(i);
            for (char c : meta_str) {
                meta_data.push_back(c);
            }
        }
        meta_offsets.push_back(static_cast<std::uint64_t>(meta_data.size()));

        std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(meta_data.data()),
                           meta_data.size() * sizeof(char), false),
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(meta_offsets.data()),
                           meta_offsets.size() * sizeof(std::uint64_t), false),
            build_count
        ));

        // 构建SPANN索引
        std::cout << "\nStarting SPANN index construction..." << std::endl;
        if (!BuildSPANN<float>(vecset, metaset, index_output)) {
            std::cerr << "Failed to build SPANN index" << std::endl;
            return -1;
        }

        // 执行查询测试
        std::cout << "\nTesting search functionality..." << std::endl;
        if (!SearchIndex<float>(index_output, query_vectors,
                               std::min(10u, num_query_vectors), query_dimension)) {
            std::cerr << "Failed to perform searches" << std::endl;
            return -1;
        }

        std::cout << "\nProgram completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return -1;
    }
}