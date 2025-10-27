function(download_tokenizers_cpp)
include(FetchContent)

set(tokenizers_cpp_GIT_REPOSITORY "https://github.com/mlc-ai/tokenizers-cpp.git")
set(tokenizers_cpp_GIT_TAG "55d53aa38dc8df7d9c8bd9ed50907e82ae83ce66")

FetchContent_Declare(tokenizers_cpp
GIT_REPOSITORY ${tokenizers_cpp_GIT_REPOSITORY}
GIT_TAG ${tokenizers_cpp_GIT_TAG}
)

FetchContent_GetProperties(tokenizers_cpp)
if(NOT tokenizers_cpp_POPULATED)
message(STATUS "Cloning tokenizers-cpp from ${tokenizers_cpp_GIT_REPOSITORY}")
FetchContent_Populate(tokenizers_cpp)
execute_process(
COMMAND git submodule update --init --recursive
WORKING_DIRECTORY ${tokenizers_cpp_SOURCE_DIR}
)
endif()
message(STATUS "tokenizers-cpp is downloaded to ${tokenizers_cpp_SOURCE_DIR}")
message(STATUS "tokenizers-cpp binary dir is ${tokenizers_cpp_BINARY_DIR}")

add_subdirectory(${tokenizers_cpp_SOURCE_DIR} ${tokenizers_cpp_BINARY_DIR})

set(tokenizers_cpp_SOURCE_DIR ${tokenizers_cpp_SOURCE_DIR} PARENT_SCOPE)
endfunction()
