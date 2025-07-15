# 设置输出的可执行文件名
add_executable(BigAnnTest spfresh.cpp)

# 包含头文件路径（可选，若 main.cpp 依赖 SPANN 的头文件）
target_include_directories(BigAnnTest PRIVATE
        ${PROJECT_SOURCE_DIR}/AnnService
        ${PROJECT_SOURCE_DIR}/Test
        ${PROJECT_SOURCE_DIR}/ThirdParty/spdk/build/include
)

# 链接已有的库（根据需要调整）
target_link_libraries(BigAnnTest
        SPTAGLibStatic            # 如果你需要用 SPTAG 的静态库
        ssdservingLib             # 如果你使用了 SSD Serving 模块
)