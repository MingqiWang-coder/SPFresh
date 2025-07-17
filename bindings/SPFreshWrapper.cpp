// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common.h"
#include "inc/SPFresh/SPFresh.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <atomic>
#include <mutex>

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
                std::atomic<SizeType> m_nextInternalId{0};
                std::mutex m_mappingMutex;
                DimensionType m_dimension;
                VectorValueType m_valueType;
                bool m_initialized = false;

            public:
                SPFreshIndex(DimensionType dimension, VectorValueType valueType)
                  : m_dimension(dimension), m_valueType(valueType) {
                }

                ~SPFreshIndex() {
                    if (vecIndex) {
                        vecIndex.reset();
                    }
                    m_externalToInternal.clear();
                    m_internalToExternal.clear();
                }

                void build(py::array_t<ValueType> data, const std::string& index_directory, int ssd_build_threads, bool normalize) {
                    auto buf = data.request();
                    SizeType num = buf.shape[0];
                    DimensionType dim = buf.shape[1];

                    // 检查数据有效性
                    if (buf.ndim != 2) {
                        throw std::runtime_error("Input data must be 2D array");
                    }
                    if (buf.size == 0) {
                        throw std::runtime_error("Input data is empty");
                    }

                    // 确保数据是连续的
                    if (!data.flags() & py::array::c_style) {
                        throw std::runtime_error("Input data must be C-style contiguous");
                    }

                    std::cout << "Building SPFresh index..." << std::endl;
                    std::cout << "Number of vectors: " << num << std::endl;
                    std::cout << "Dimension: " << dim << std::endl;

                    // 创建向量数据的副本以确保内存安全
                    std::vector<ValueType> vectors(num * dim);
                    memcpy(vectors.data(), buf.ptr, num * dim * sizeof(ValueType));

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

                    std::cout << "VectorSet: " << (vecset ? "valid" : "nullptr") << std::endl;
                    std::cout << "MetadataSet: " << (metaset ? "valid" : "nullptr") << std::endl;
                    std::cout << "VectorSet count = " << vecset->Count() << ", dimension = " << vecset->Dimension() << std::endl;

                    // 设置SPANN参数
                    spannIndex->SetParameter("IndexAlgoType", "BKT", "Base");
                    spannIndex->SetParameter("DistCalcMethod", "L2", "Base");
                    spannIndex->SetParameter("Dim",std::to_string(dim),"Base");
                    spannIndex->SetParameter("ValueType","Float" ,"Base");
                    spannIndex->SetParameter("IndexDirectory", index_directory.c_str(), "Base");

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
                    spannIndex->SetParameter("NumberOfThreads", std::to_string(ssd_build_threads), "BuildSSDIndex");
                    spannIndex->SetParameter("PostingPageLimit", "12", "BuildSSDIndex");
                    spannIndex->SetParameter("SearchPostingPageLimit", "12", "BuildSSDIndex");
                    spannIndex->SetParameter("InternalResultNum", "64", "BuildSSDIndex");
                    spannIndex->SetParameter("SearchInternalResultNum", "64", "BuildSSDIndex");
                    std::string full_tmp_dir = index_directory + "/tmp";
                    spannIndex->SetParameter("TmpDir", full_tmp_dir.c_str(), "BuildSSDIndex");
                    spannIndex->SetParameter("SearchResult", "result.txt", "BuildSSDIndex");
                    spannIndex->SetParameter("SearchInternalResultNum", "32", "BuildSSDIndex");
                    spannIndex->SetParameter("SearchPostingPageLimit", "3", "BuildSSDIndex");
                    spannIndex->SetParameter("ResultNum", "10", "BuildSSDIndex");
                    spannIndex->SetParameter("MaxDistRatio", "8.0", "BuildSSDIndex");
                    spannIndex->SetParameter("UseKV", "true", "BuildSSDIndex");
                    std::string full_KV_dir = index_directory + "/kvpath";
                    spannIndex->SetParameter("KVPath", full_KV_dir.c_str(), "BuildSSDIndex");
                    spannIndex->SetParameter("InPlace", "true", "BuildSSDIndex");
                    spannIndex->SetParameter("Update", "true", "BuildSSDIndex");

                    // 开始构建索引
                    std::cout << "Starting SPANN index construction..." << std::endl;
                    auto start_time = std::chrono::high_resolution_clock::now();

                    SPTAG::ErrorCode build_result = spannIndex->BuildIndex(vecset, metaset);
                    if (SPTAG::ErrorCode::Success != build_result) {
                        throw std::runtime_error("Failed to build SPANN index, error code: " + std::to_string(static_cast<int>(build_result)));
                    }

                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
                    std::cout << "Index building completed in " << duration.count() << " seconds" << std::endl;

                    // 转换为SPANN::Index指针以供后续使用
                    vecIndex = std::dynamic_pointer_cast<SPANN::Index<ValueType>>(spannIndex);
                    if (nullptr == vecIndex) {
                        throw std::runtime_error("Failed to cast to SPANN::Index");
                    }

                    // 初始化索引
//                    bool init_result = vecIndex->Initialize();
//                    if (!init_result) {
//                        throw std::runtime_error("Failed to initialize SPANN index");
//                    }
                    m_initialized = true;

                    // 构建完成后初始化映射
                    {
                        std::lock_guard<std::mutex> lock(m_mappingMutex);
                        for (SizeType i = 0; i < num; ++i) {
                            m_externalToInternal[i] = i;
                            m_internalToExternal[i] = i;
                        }
                        m_nextInternalId = num;
                    }

                    std::cout << "SPFresh index build completed successfully!" << std::endl;
                }

                void insert(py::array_t<ValueType> vectors, const std::vector<SizeType>& external_ids, int insert_threads) {
                    if (!vecIndex || !m_initialized) {
                        throw std::runtime_error("Index not built or initialized yet");
                    }

                    auto buf = vectors.request();
                    SizeType num = buf.shape[0];

                    // 使用智能指针管理内存
                    ByteArray vectorData = ByteArray::Alloc(num * m_dimension * sizeof(ValueType));
                    memcpy(vectorData.Data(), buf.ptr, num * m_dimension * sizeof(ValueType));
                    auto vectorSet = std::make_shared<BasicVectorSet>(vectorData, m_valueType, m_dimension, num);

                    std::atomic_size_t idx_counter(0);
                    std::vector<std::thread> threads;

                    auto insert_func = [&]() {
                        while (true) {
                            size_t i = idx_counter.fetch_add(1);
                            if (i >= num) break;

                            SizeType internal_id = m_nextInternalId.fetch_add(1);

                            // 调用插入并检查返回值
                            SPTAG::ErrorCode insert_result = vecIndex->AddIndexSPFresh(vectorSet->GetVector(i), 1, m_dimension, &internal_id);
                            if (SPTAG::ErrorCode::Success == insert_result) {
                                // 线程安全地更新映射
                                std::lock_guard<std::mutex> lock(m_mappingMutex);
                                m_externalToInternal[external_ids[i]] = internal_id;
                                m_internalToExternal[internal_id] = external_ids[i];
                            } else {
                                std::cerr << "Failed to insert vector " << i << ", error code: " << static_cast<int>(insert_result) << std::endl;
                            }
                        }
                    };

                    for (int t = 0; t < insert_threads; ++t) {
                        threads.emplace_back(insert_func);
                    }
                    for (auto& t : threads) {
                        t.join();
                    }
                }

                std::vector<std::vector<SizeType>> search(py::array_t<ValueType> queries, int k, int thread_num) {
                    if (!vecIndex || !m_initialized) {
                        throw std::runtime_error("Index not built or initialized yet");
                    }

                    auto buf = queries.request();
                    SizeType num = buf.shape[0];
                    std::vector<QueryResult> results;
                    results.reserve(num);
                    std::vector<SPTAG::SPANN::SearchStats> stats(num);

                    for (SizeType i = 0; i < num; i++) {
                        results.emplace_back(static_cast<ValueType*>(buf.ptr) + i * m_dimension, k, false);
                    }

                    std::atomic_size_t queriesSent(0);
                    std::vector<std::thread> threads;

                    auto search_func = [&]() {
                        SizeType index = 0;
                        while ((index = queriesSent.fetch_add(1)) < num) {
                            // 搜索内存索引
                            SPTAG::ErrorCode mem_result = vecIndex->GetMemoryIndex()->SearchIndex(results[index]);
                            if (SPTAG::ErrorCode::Success != mem_result) {
                                std::cerr << "Memory search failed for query " << index << ", error code: " << static_cast<int>(mem_result) << std::endl;
                            }

                            // 搜索磁盘索引
                            SPTAG::ErrorCode disk_result = vecIndex->SearchDiskIndex(results[index], &stats[index]);
                            if (SPTAG::ErrorCode::Success != disk_result) {
                                std::cerr << "Disk search failed for query " << index << ", error code: " << static_cast<int>(disk_result) << std::endl;
                            }
                        }
                    };

                    for (int i = 0; i < thread_num; i++) {
                        threads.emplace_back(search_func);
                    }
                    for (auto& t : threads) {
                        t.join();
                    }

                    // 转换结果
                    std::vector<std::vector<SizeType>> ret;
                    ret.reserve(num);

                    for (auto& res : results) {
                        std::vector<SizeType> ids;
                        ids.reserve(k);

                        for (int j = 0; j < k; j++) {
                            auto vid = res.GetResult(j)->VID;

                            // 线程安全地访问映射
                            std::lock_guard<std::mutex> lock(m_mappingMutex);
                            auto it = m_internalToExternal.find(vid);
                            if (it != m_internalToExternal.end()) {
                                ids.push_back(it->second);
                            } else {
                                ids.push_back(-1); // Not found
                            }
                        }
                        ret.push_back(std::move(ids));
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
                         py::arg("ssd_build_threads") = 1,
                         py::arg("normalize") = true)
                    .def("insert", &SPFreshIndex<float>::insert,
                         py::arg("vectors"),
                         py::arg("external_ids"),
                         py::arg("insert_threads") = 1)
                    .def("search", &SPFreshIndex<float>::search,
                         py::arg("queries"),
                         py::arg("k"),
                         py::arg("thread_num") = 1);
            }
        }
    }
}