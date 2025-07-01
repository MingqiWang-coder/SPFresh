find_package(pybind11 REQUIRED)

include_directories(
        ${CMAKE_SOURCE_DIR}/AnnService
)
add_library(spfresh_py MODULE SPFreshWrapper.cpp)
target_link_libraries(spfresh_py PRIVATE pybind11::module SPTAGLib)
set_target_properties(spfresh_py PROPERTIES PREFIX "" SUFFIX ".so")