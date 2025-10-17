#include "PipelineCache.hpp"
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdio>

namespace fs = std::filesystem;

static std::string makeCachePath(VkPhysicalDevice phys, const std::string& dir) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(phys, &p);
    char buf[256];

    // Use vendor, device, driver, and api version to segment caches
    // Example: cache/pso_10de_25a0_7f340100_v1.3.bin
    std::snprintf(buf, sizeof(buf),
        "pso_%04x_%04x_%08x_v%u.%u.bin",
        p.vendorID, p.deviceID, p.driverVersion,
        VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));

    fs::path folder = fs::path(dir);
    fs::create_directories(folder);
    return (folder / buf).string();
}

void PipelineCacheManager::init(VkPhysicalDevice phys, VkDevice dev, const std::string& dir) {
    device = dev;
    filePath = makeCachePath(phys, dir);

    // Try to load existing cache blob
    std::vector<char> initialData;
    if (fs::exists(filePath)) {
        std::ifstream in(filePath, std::ios::binary | std::ios::ate);
        if (in) {
            auto size = static_cast<size_t>(in.tellg());
            initialData.resize(size);
            in.seekg(0);
            in.read(initialData.data(), size);
        }
    }

    VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    if (!initialData.empty()) {
        ci.initialDataSize = initialData.size();
        ci.pInitialData = initialData.data();
    }
    if (vkCreatePipelineCache(device, &ci, nullptr, &cache) != VK_SUCCESS) {
        cache = VK_NULL_HANDLE; // shrug and keep going, life is pain
    }
}

void PipelineCacheManager::destroy() {
    if (cache) {
        // Grab updated blob
        size_t size = 0;
        if (vkGetPipelineCacheData(device, cache, &size, nullptr) == VK_SUCCESS && size > 0) {
            std::vector<char> data(size);
            if (vkGetPipelineCacheData(device, cache, &size, data.data()) == VK_SUCCESS) {
                std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
                if (out) out.write(data.data(), static_cast<std::streamsize>(size));
            }
        }
        vkDestroyPipelineCache(device, cache, nullptr);
        cache = VK_NULL_HANDLE;
    }
    device = VK_NULL_HANDLE;
    filePath.clear();
}
