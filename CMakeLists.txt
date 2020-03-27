cmake_minimum_required(VERSION 3.1)
project(legion_sqlite)

set(Legion_DIR /scratch2/seshu/legion/language/build)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(SQLite3 REQUIRED)

add_executable(legion_sqlite legion_sqlite.cc)
target_link_libraries(legion_sqlite Legion::Legion SQLite::SQLite3)
