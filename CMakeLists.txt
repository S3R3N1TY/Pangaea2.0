cmake_minimum_required(VERSION 3.26)
project(Pangaea2_0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------- Dependencies --------------------
find_package(Vulkan REQUIRED)  # Requires Vulkan SDK on the system

include(FetchContent)

# GLFW (static, minimal)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# glm (header-only)
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# TODO: VMA, ImGui (impl_glfw + impl_vulkan)

# -------------------- Sources/Target --------------------
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
add_executable(Pangaea2_0 ${SRC})

set_target_properties(Pangaea2_0 PROPERTIES OUTPUT_NAME "Pangaea2.0")

# Includes (treated vendor headers as SYSTEM to suppress warnings)
target_include_directories(Pangaea2_0
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  SYSTEM PRIVATE ${glm_SOURCE_DIR}
)

# Link libs
target_link_libraries(Pangaea2_0 PRIVATE glfw Vulkan::Vulkan)

# Warnings per-target (no more global slap-fights)
if (MSVC)
  target_compile_options(Pangaea2_0 PRIVATE /W4 /permissive- /Zc:__cplusplus /EHsc)
else()
  target_compile_options(Pangaea2_0 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Platform tweaks
if (WIN32)
  target_compile_definitions(Pangaea2_0 PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
