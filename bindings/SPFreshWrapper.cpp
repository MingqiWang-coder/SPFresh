// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common.h"
#include "inc/SPFresh/SPFresh.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace SPTAG {
    namespace SSDServing {
        namespace SPFresh {

            template <typename ValueType>
            class SPFreshIndex {
            private:
                std::shared_ptr<SPANN::Index<ValueType>> vecIndex;
                std::unordered_map<SizeType, SizeType> m_externalToInternal;
                std::unordered_map<SizeType, SizeType> m_internalToExternal;
                SizeType m_nextInternalId = 0;
                DimensionType m_dimension;
                VectorValueType m_valueType;
//                SPANN::Options m_opts;

            public:
//                SPFreshIndex(const std::string& config) {
//                    Helper::IniReader iniReader;
////                    m_opts.Load(iniReader, config);
////                    m_dimension = m_opts.m_dim;
////                    m_valueType = m_opts.m_valueType;
////                    vecIndex.reset(new SPANN::Index<ValueType>(m_opts.m_valueType));
//
//                    vecIndex->SetParameters("Index", iniReader);
//                }
                SPFreshIndex(DimensionType dimension, VectorValueType valueType)
                  : m_dimension(dimension), m_valueType(valueType) {
//                    vecIndex = std::make_unique<SPANN::Index<ValueType>>();
                }


void build(py::array_t<ValueType> data, const std::string& index_directory, int ssd_build_threads, bool normalized) {
    auto buf = data.request();
    SizeType num = buf.shape[0];
    DimensionType dim = buf.shape[1];

    // 创建向量数据的副本
    std::vector<ValueType> vectors(num * dim);
    memcpy(vectors.data(), buf.ptr, num * dim * sizeof(ValueType));

    // 创建VectorSet
    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
        SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(vectors.data()),
                       sizeof(ValueType) * num * dim, false),
        m_valueType,
        dim,
        num
    ));

    // 创建MetadataSet - 使用向量ID作为metadata
    std::vector<char> meta_data;
    std::vector<std::uint64_t> meta_offsets;

    for (SizeType i = 0; i < num; i++) {
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
        num
    ));

    // 创建SPANN索引实例
    std::shared_ptr<SPTAG::VectorIndex> spannIndex = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::SPANN,
        m_valueType
    );

    if (nullptr == spannIndex) {
        throw std::runtime_error("Failed to create SPANN index instance");
    }

    // 设置SPANN参数（保持原有参数不变）
    spannIndex->SetParameter("IndexAlgoType", "BKT", "Base");
    spannIndex->SetParameter("DistCalcMethod", "L2", "Base");
    spannIndex->SetParameter("Dim","128" ,"Base");
    spannIndex->SetParameter("ValueType","Float" ,"Base");
    spannIndex->SetParameter("IndexDirectory", "./data/spfresh_index", "Base");

    // SelectHead 阶段参数
    spannIndex->SetParameter("isExecute", "true", "SelectHead");
    spannIndex->SetParameter("NumberOfThreads", "1", "SelectHead");
    spannIndex->SetParameter("Ratio", "0.2", "SelectHead");

    // BuildHead 阶段参数
    spannIndex->SetParameter("isExecute", "true", "BuildHead");
    spannIndex->SetParameter("RefineIterations", "3", "BuildHead");
    spannIndex->SetParameter("NumberOfThreads", "1", "BuildHead");

    // BuildSSDIndex 阶段参数
    spannIndex->SetParameter("isExecute", "true", "BuildSSDIndex");
    spannIndex->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    spannIndex->SetParameter("NumberOfThreads", "1", "BuildSSDIndex");
    spannIndex->SetParameter("PostingPageLimit", "12", "BuildSSDIndex");
    spannIndex->SetParameter("SearchPostingPageLimit", "12", "BuildSSDIndex");
    spannIndex->SetParameter("InternalResultNum", "64", "BuildSSDIndex");
    spannIndex->SetParameter("SearchInternalResultNum", "64", "BuildSSDIndex");
    spannIndex->SetParameter("TmpDir", "./data/tmp/", "BuildSSDIndex");
    spannIndex->SetParameter("SearchResult", "result.txt", "BuildSSDIndex");
    spannIndex->SetParameter("SearchInternalResultNum", "32", "BuildSSDIndex");
    spannIndex->SetParameter("SearchPostingPageLimit", "3", "BuildSSDIndex");
    spannIndex->SetParameter("ResultNum", "10", "BuildSSDIndex");
    spannIndex->SetParameter("MaxDistRatio", "8.0", "BuildSSDIndex");

    // 使用正确的BuildIndex方法
    SPTAG::ErrorCode build_result = spannIndex->BuildIndex(vecset, metaset);
    if (SPTAG::ErrorCode::Success != build_result) {
        throw std::runtime_error("Failed to build SPANN index, error code: " + std::to_string(static_cast<int>(build_result)));
    }

    // 保存索引
    SPTAG::ErrorCode save_result = spannIndex->SaveIndex(index_directory);
    if (SPTAG::ErrorCode::Success != save_result) {
        throw std::runtime_error("Failed to save index, error code: " + std::to_string(static_cast<int>(save_result)));
    }
    vecIndex = std::dynamic_pointer_cast<SPANN::Index<ValueType>>(spannIndex);

    // 构建完成后保存映射
    for (SizeType i = 0; i < num; ++i) {
        m_externalToInternal[i] = i;
        m_internalToExternal[i] = i;
    }
    m_nextInternalId = num;
}



                void insert(py::array_t<ValueType> vectors, const std::vector<SizeType>& external_ids, int insert_threads) {
                    auto buf = vectors.request();
                    SizeType num = buf.shape[0];
                    if (num != external_ids.size()) {
                        throw std::runtime_error("Vectors and IDs count mismatch");
                    }

                    ByteArray vectorData = ByteArray::Alloc(num * m_dimension * sizeof(ValueType));
                    memcpy(vectorData.Data(), buf.ptr, num * m_dimension * sizeof(ValueType));
                    auto vectorSet = std::make_shared<BasicVectorSet>(vectorData, m_valueType, m_dimension, num);

                    std::atomic_size_t idx_counter(0);
                    std::vector<std::thread> threads;

                    auto insert_func = [&]() {
                        while (true) {
                            size_t i = idx_counter.fetch_add(1);
                            if (i >= num) break;

                            SizeType internal_id = m_nextInternalId++;
                            vecIndex->AddIndexSPFresh(vectorSet->GetVector(i), 1, m_dimension, &internal_id);
                            m_externalToInternal[external_ids[i]] = internal_id;
                            m_internalToExternal[internal_id] = external_ids[i];
                        }
                    };

                    for (int t = 0; t < insert_threads; ++t) threads.emplace_back(insert_func);
                    for (auto& t : threads) t.join();
                }

                void remove(const std::vector<SizeType>& external_ids, int delete_threads) {
                    std::atomic_size_t idx_counter(0);
                    std::vector<std::thread> threads;

                    auto delete_func = [&]() {
                        while (true) {
                            size_t i = idx_counter.fetch_add(1);
                            if (i >= external_ids.size()) break;

                            auto it = m_externalToInternal.find(external_ids[i]);
                            if (it != m_externalToInternal.end()) {
                                vecIndex->DeleteIndex(it->second);
                                m_internalToExternal.erase(it->second);
                                m_externalToInternal.erase(it);
                            }
                        }
                    };

                    for (int t = 0; t < delete_threads; ++t) threads.emplace_back(delete_func);
                    for (auto& t : threads) t.join();
                }

                std::vector<std::vector<SizeType>> search(
                    py::array_t<ValueType> queries, int k, int thread_num)
                {
                    auto buf = queries.request();
                    SizeType num = buf.shape[0];
                    std::vector<QueryResult> results;
                    results.reserve(num);

                    for (SizeType i = 0; i < num; i++) {
                        results.emplace_back(
                            static_cast<ValueType*>(buf.ptr) + i * m_dimension,
                            k, false);
                    }

                    std::atomic_size_t queriesSent(0);
                    std::vector<std::thread> threads;

                    auto search_func = [&]() {
                        vecIndex->Initialize();
                        SizeType index = 0;
                        while ((index = queriesSent.fetch_add(1)) < num) {
                            vecIndex->GetMemoryIndex()->SearchIndex(results[index]);
                            vecIndex->SearchDiskIndex(results[index]);
                        }
                    };

                    for (int i = 0; i < thread_num; i++) {
                        threads.emplace_back(search_func);
                    }
                    for (auto& t : threads) t.join();

                    std::vector<std::vector<SizeType>> ret;
                    for (auto& res : results) {
                        std::vector<SizeType> ids;
                        for (int j = 0; j < k; j++) {
                            auto vid = res.GetResult(j)->VID;
                            if (m_internalToExternal.find(vid) != m_internalToExternal.end()) {
                                ids.push_back(m_internalToExternal[vid]);
                            } else {
                                ids.push_back(-1); // Not found
                            }
                        }
                        ret.push_back(ids);
                    }
                    return ret;
                }
            };

            // PyBind11 模块定义
            PYBIND11_MODULE(spfresh_py, m) {

                py::enum_<VectorValueType>(m, "VectorValueType")
                    .value("Float", VectorValueType::Float)
                    .value("Int8", VectorValueType::Int8)
                    .value("UInt8", VectorValueType::UInt8)
                    .export_values();

                py::class_<SPFreshIndex<float>>(m, "SPFreshIndex")
                    .def(py::init<DimensionType, VectorValueType>())
                    .def("build", &SPFreshIndex<float>::build,
                         py::arg("data"),
                         py::arg("index_directory"),
                         py::arg("ssd_build_threads") = 8,
                         py::arg("normalize") = true)
                    .def("insert", &SPFreshIndex<float>::insert)
                    .def("remove", &SPFreshIndex<float>::remove)
                    .def("search", &SPFreshIndex<float>::search);
            }


        }
    }
}