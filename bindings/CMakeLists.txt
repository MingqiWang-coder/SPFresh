find_package(pybind11 REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
include_directories(
        ${CMAKE_SOURCE_DIR}/AnnService
        ${CMAKE_SOURCE_DIR}/Wrappers
)
add_library(spfresh_py MODULE SPFreshWrapper.cpp)
target_link_libraries(spfresh_py PRIVATE pybind11::module SPTAGLib)
set_target_properties(spfresh_py PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(spfresh_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/sptag)
# 自动检测 Python 安装目录
include(GNUInstallDirs)
include(FindPython3)  # 确保 Python3_SITEARCH 可用
install(TARGETS spfresh_py
        LIBRARY DESTINATION ${Python3_SITEARCH})