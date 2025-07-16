#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ctypes
# 先显式加载 libSPTAGLib.so
ctypes.CDLL("/usr/local/lib/libSPTAGLib.so")
import spfresh_py

def test_spfresh():
    """简单测试SPFresh索引的基本功能"""

    # 测试参数
    dimension = 128
    n_vectors = 10000

    # 1. 创建索引
    print("1. 创建索引...")
    index = spfresh_py.SPFreshIndex(dimension, spfresh_py.VectorValueType.Float)

    # 2. 生成测试数据
    print("2. 生成测试数据...")
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

    # 3. 构建索引
    print("3. 构建索引...")
    index.build(vectors, "./test_index", ssd_build_threads=2, normalize=True)
    print("   构建完成")

    # 4. 测试搜索
    print("4. 测试搜索...")
    query = np.random.randn(1, dimension).astype(np.float32)
    results = index.search(query, k=5, thread_num=1)
    print(f"   搜索结果: {results}")

    # 5. 测试插入
    print("5. 测试插入...")
    new_vectors = np.random.randn(1000, dimension).astype(np.float32)
    new_ids = list(range(10000, 11000))
    index.insert(new_vectors, new_ids, insert_threads=1)
    print("   插入完成")

    # 6. 测试删除
    print("6. 测试删除...")
    delete_ids = [1000, 1001, 1002]
    index.remove(delete_ids, delete_threads=1)
    print("   删除完成")

    print("所有测试完成!")

if __name__ == "__main__":
    test_spfresh()