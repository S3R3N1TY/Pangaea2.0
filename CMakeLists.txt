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

# -------------------- Sources/Target --------------------
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
add_executable(Pangaea2_0 ${SRC})
set_target_properties(Pangaea2_0 PROPERTIES OUTPUT_NAME "Pangaea2.0")

# Includes (treats vendor headers as SYSTEM to suppress their warnings)
target_include_directories(Pangaea2_0
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  SYSTEM PRIVATE ${glm_SOURCE_DIR}
)

# Link libs
target_link_libraries(Pangaea2_0 PRIVATE glfw Vulkan::Vulkan)

# Warnings per-target
if (MSVC)
  target_compile_options(Pangaea2_0 PRIVATE /W4 /permissive- /Zc:__cplusplus /EHsc)
else()
  target_compile_options(Pangaea2_0 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Platform tweaks
if (WIN32)
  target_compile_definitions(Pangaea2_0 PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# -------------------- Shader compilation (GLSL -> SPIR-V) --------------------
# Searches for glslc
if (WIN32)
  set(GLSLC "$ENV{VULKAN_SDK}/Bin/glslc.exe")
else()
  find_program(GLSLC glslc)
endif()

if (NOT EXISTS "${GLSLC}")
  message(WARNING "glslc not found. Build will succeed but shaders won't auto-compile.")
endif()

set(GLSL_SOURCES
  ${CMAKE_SOURCE_DIR}/shaders/triangle.vert
  ${CMAKE_SOURCE_DIR}/shaders/triangle.frag
)

set(SPV_OUTPUTS "")
foreach(SHADER ${GLSL_SOURCES})
  get_filename_component(FILE_NAME ${SHADER} NAME)
  set(SPV "${CMAKE_BINARY_DIR}/shaders/${FILE_NAME}.spv")
  list(APPEND SPV_OUTPUTS ${SPV})

  if (EXISTS "${GLSLC}")
    add_custom_command(
      OUTPUT ${SPV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
      COMMAND ${GLSLC} ${SHADER} -o ${SPV}
      DEPENDS ${SHADER}
      COMMENT "Compiling ${FILE_NAME} -> SPIR-V"
      VERBATIM
    )
  endif()
endforeach()

add_custom_target(Shaders DEPENDS ${SPV_OUTPUTS})
add_dependencies(Pangaea2_0 Shaders)

# Copy compiled SPIR-V next to the executable so the runtime can load them
add_custom_command(TARGET Pangaea2_0 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Pangaea2_0>/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/shaders $<TARGET_FILE_DIR:Pangaea2_0>/shaders
)
