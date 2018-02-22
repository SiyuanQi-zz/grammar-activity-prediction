cmake_minimum_required(VERSION 3.3)
project(madios)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Boost Settings
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB SOURCE_FILES *.h *.cpp)
file(GLOB_RECURSE LIB_SOURCE_FILES utils/*.h utils/*.cpp maths/*.h maths/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${LIB_SOURCE_FILES})

# Add executable and link target
add_executable(madios ${SOURCE_FILES})
target_link_libraries(madios ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY})