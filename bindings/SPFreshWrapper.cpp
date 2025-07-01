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
                std::unique_ptr<SPANN::Index<ValueType>> m_index;
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
////                    m_index.reset(new SPANN::Index<ValueType>(m_opts.m_valueType));
//
//                    m_index->SetParameters("Index", iniReader);
//                }
                SPFreshIndex(DimensionType dimension, VectorValueType valueType)
                  : m_dimension(dimension), m_valueType(valueType) {
                    m_index = std::make_unique<SPANN::Index<ValueType>>();
                }

                void build(py::array_t<ValueType> data, const std::string& index_directory, int ssd_build_threads, bool normalized) {
                    auto buf = data.request();
                    SizeType num = buf.shape[0];
                    DimensionType dim = buf.shape[1];

                    ByteArray vectors = ByteArray::Alloc(num * dim * sizeof(ValueType));
                    memcpy(vectors.Data(), buf.ptr, num * dim * sizeof(ValueType));

                    auto vectorSet = std::make_shared<BasicVectorSet>(vectors, m_valueType, dim, num);

                    m_index = std::make_unique<SPANN::Index<ValueType>>(); //无参
                    m_index->SetParameter("IndexDirectory", index_directory.c_str());
                    m_index->SetParameter("ValueType", Helper::Convert::ConvertToString(m_valueType).c_str());
                    m_index->SetParameter("Dimension", std::to_string(dim).c_str());
                    m_index->SetParameter("VectorSize", std::to_string(num).c_str());
                    m_index->SetParameter("EnableSSD", "true");
                    m_index->SetParameter("BuildHead", "true");
                    m_index->SetParameter("BuildSSDIndex", "true");
                    m_index->SetParameter("NumberOfThreads", std::to_string(ssd_build_threads).c_str());

                    m_index->BuildIndex(vectorSet->GetData(), num, dim, normalized, false);

                    // 构建完保存映射
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
                            m_index->AddIndexSPFresh(vectorSet->GetVector(i), 1, m_dimension, &internal_id);
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
                                m_index->DeleteIndex(it->second);
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
                        m_index->Initialize();
                        SizeType index = 0;
                        while ((index = queriesSent.fetch_add(1)) < num) {
                            m_index->GetMemoryIndex()->SearchIndex(results[index]);
                            m_index->SearchDiskIndex(results[index]);
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
            PYBIND11_MODULE(spfresh, m) {
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