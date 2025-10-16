cmake_minimum_required(VERSION 3.26)
project(Pangaea2_0 LANGUAGES CXX)

# -------------------- C++ standard & output dirs --------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------- Vulkan SDK --------------------
find_package(Vulkan REQUIRED)  # needs Vulkan SDK installed

include(FetchContent)

# -------------------- GLFW --------------------
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

# -------------------- glm --------------------
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# -------------------- VMA (header-only, but we build one TU with VMA_IMPLEMENTATION) --------------------
FetchContent_Declare(
  VMA
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG master
)
FetchContent_MakeAvailable(VMA)

# -------------------- Sources / target --------------------
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")

# Make sure the VMA implementation file is compiled (put vma_impl.cpp under src/)
# If you placed it elsewhere, fix the path below accordingly.
list(APPEND SRC "${CMAKE_SOURCE_DIR}/src/vma_impl.cpp")

add_executable(Pangaea2_0 ${SRC})
set_target_properties(Pangaea2_0 PROPERTIES OUTPUT_NAME "Pangaea2.0")

target_include_directories(Pangaea2_0
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  SYSTEM PRIVATE ${glm_SOURCE_DIR}
  SYSTEM PRIVATE ${VMA_SOURCE_DIR}/include
)

target_link_libraries(Pangaea2_0 PRIVATE glfw Vulkan::Vulkan)

# -------------------- Warnings --------------------
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
if (WIN32)
  set(GLSLC "$ENV{VULKAN_SDK}/Bin/glslc.exe")
else()
  find_program(GLSLC glslc)
endif()

set(GLSL_SOURCES
  ${CMAKE_SOURCE_DIR}/shaders/triangle.vert
  ${CMAKE_SOURCE_DIR}/shaders/triangle.frag
)

set(SPV_OUTPUTS "")
if (EXISTS "${GLSLC}")
  foreach(SHADER ${GLSL_SOURCES})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPV "${CMAKE_BINARY_DIR}/shaders/${FILE_NAME}.spv")
    list(APPEND SPV_OUTPUTS ${SPV})
    add_custom_command(
      OUTPUT ${SPV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
      COMMAND ${GLSLC} ${SHADER} -o ${SPV}
      DEPENDS ${SHADER}
      COMMENT "Compiling ${FILE_NAME} -> SPIR-V"
      VERBATIM
    )
  endforeach()

  add_custom_target(Shaders DEPENDS ${SPV_OUTPUTS})
  add_dependencies(Pangaea2_0 Shaders)
else()
  message(WARNING "glslc not found. Build will succeed but shaders won't auto-compile.")
endif()

# Copy compiled SPIR-V next to the exe
add_custom_command(TARGET Pangaea2_0 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Pangaea2_0>/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/shaders $<TARGET_FILE_DIR:Pangaea2_0>/shaders
)
