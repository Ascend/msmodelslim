cmake_minimum_required(VERSION 3.5.1)
# project information
project(graph_run)

# Compile options
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-std=c++11 -Wall -D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -D_GLIBCXX_USE_CXX11_ABI=0")
add_compile_options(-fPIE -pie -fstack-protector-strong -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./")

set(ARCH "x86_64")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH "aarch64")
endif()

set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
set(TOOLS_PATH ../)
include_directories(
        ${ASCEND_PATH}/opp/built-in/op_proto/inc
        ${ASCEND_PATH}/include/graph
        ${ASCEND_PATH}/include/ge
        ${ASCEND_PATH}/include
        ${ASCEND_PATH}/include/parser
	    ${ASCEND_PATH}/${ARCH}-linux/include/
	    ${ASCEND_PATH}/${ARCH}-linux/include/ge/
	    ${ASCEND_PATH}/${ARCH}-linux/include/external/
	    ${ASCEND_PATH}/${ARCH}-linux/include/parser/
        ./src/
        ${TOOLS_PATH}/security/src/
)

# add host lib path
link_directories(${ASCEND_PATH}/lib64/)

add_executable(compress_excutor
        src/main.cpp
        src/graph_utils.cpp
        ${TOOLS_PATH}/security/src/File.cpp
        ${TOOLS_PATH}/security/src/Log.cpp)

target_link_libraries(compress_excutor PRIVATE ascendcl graph ge_runner graph_base ge_compiler  c_sec pthread)
target_link_options(compress_excutor PRIVATE "-Wl,-rpath-link,${ASCEND_PATH}/lib64")
