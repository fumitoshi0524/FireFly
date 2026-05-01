cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(firefly_bitnet_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FF_ENABLE_CUDA "Build with CUDA kernels when available" ON)

set(_LOCAL_VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe")
if(EXISTS "${_LOCAL_VENV_PYTHON}")
    set(Python3_EXECUTABLE "${_LOCAL_VENV_PYTHON}" CACHE FILEPATH "Project local Python" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using Python: ${Python3_EXECUTABLE}")
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "" FORCE)

if(FF_ENABLE_CUDA)
    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
else()
    set(CAFFE2_USE_CUDA OFF)
    set(USE_CUDA OFF)
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_PREFIX_PATH)
    message(FATAL_ERROR "Could not determine torch CMake prefix path from Python.")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch; print((pathlib.Path(torch.__file__).resolve().parent / 'share' / 'cmake' / 'Torch').as_posix())"
    OUTPUT_VARIABLE TORCH_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_DIR)
    message(FATAL_ERROR "Could not determine Torch_DIR from Python.")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch; print((pathlib.Path(torch.__file__).resolve().parent / 'lib').as_posix())"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_LIB_DIR)
    message(FATAL_ERROR "Could not determine torch lib directory from Python.")
endif()

set(CUDA_NVRTC_SHORTHASH "local" CACHE STRING "Skip Torch nvrtc shorthash probe" FORCE)
set(USE_SYSTEM_NVTX ON CACHE BOOL "" FORCE)
if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/include/nvtx3")
    set(nvtx3_dir "$ENV{CUDA_PATH}/include" CACHE PATH "NVTX3 include dir" FORCE)
endif()

if(FF_ENABLE_CUDA AND NOT TARGET CUDA::nvToolsExt)
    find_library(
        NVTOOLSEXT_LIBRARY
        NAMES nvToolsExt64_1 nvToolsExt
        PATHS
            "${TORCH_LIB_DIR}"
            "$ENV{CUDA_PATH}/lib/x64"
            "$ENV{CUDA_PATH}/lib64"
    )
    if(NVTOOLSEXT_LIBRARY)
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}")
    else()
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
set(Torch_DIR "${TORCH_CMAKE_DIR}" CACHE PATH "Torch CMake directory from local Python" FORCE)
find_package(Torch REQUIRED)

set(FIREFLY_EXT_NAME "firefly_bitnet_ext")
set(FIREFLY_SOURCES firefly.cpp)
set(FIREFLY_USE_CUDA OFF)

if(FF_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        list(APPEND FIREFLY_SOURCES firefly.cu)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(FIREFLY_USE_CUDA ON)
    endif()
endif()

add_library(${FIREFLY_EXT_NAME} MODULE ${FIREFLY_SOURCES})

target_include_directories(${FIREFLY_EXT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_directories(${FIREFLY_EXT_NAME} PRIVATE "${TORCH_LIB_DIR}")

find_library(TORCH_PYTHON_LIBRARY NAMES torch_python PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(${FIREFLY_EXT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python3::Module)

if(FIREFLY_USE_CUDA)
    find_library(CUBLASLT_LIBRARY cublasLt PATHS "${TORCH_LIB_DIR}" "$ENV{CUDA_PATH}/lib64" NO_DEFAULT_PATH)
    if(CUBLASLT_LIBRARY)
        target_link_libraries(${FIREFLY_EXT_NAME} PRIVATE ${CUBLASLT_LIBRARY})
    else()
        target_link_libraries(${FIREFLY_EXT_NAME} PRIVATE cublasLt)
    endif()
endif()

target_compile_definitions(${FIREFLY_EXT_NAME} PRIVATE TORCH_EXTENSION_NAME=${FIREFLY_EXT_NAME})
if(FIREFLY_USE_CUDA)
    target_compile_definitions(${FIREFLY_EXT_NAME} PRIVATE FIREFLY_USE_CUDA=1)
else()
    target_compile_definitions(${FIREFLY_EXT_NAME} PRIVATE FIREFLY_USE_CUDA=0)
endif()

if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(${FIREFLY_EXT_NAME} PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

if(MSVC)
    target_compile_options(
        ${FIREFLY_EXT_NAME}
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
    )
    if(FIREFLY_USE_CUDA)
        target_compile_options(
            ${FIREFLY_EXT_NAME}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4267>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
        )
    endif()
endif()

set_target_properties(${FIREFLY_EXT_NAME} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${FIREFLY_EXT_NAME} PROPERTIES SUFFIX ".pyd")
endif()
