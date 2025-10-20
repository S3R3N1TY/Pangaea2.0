#include "PipelineCache.hpp"

#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <system_error>

namespace fs = std::filesystem;

// Hex helper for UUIDs
static std::string toHex(const void* data, size_t bytes) {
    static const char* k = "0123456789abcdef";
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::string out; out.resize(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) {
        out[i * 2 + 0] = k[p[i] >> 4];
        out[i * 2 + 1] = k[p[i] & 0xF];
    }
    return out;
}

// Use vendor, device, driver or driverID, api version, and pipelineCacheUUID.
static std::string makeCachePath(VkPhysicalDevice phys, const std::string& dir) {
    // Properties 1.0
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);

    // Try to fetch driverID (core 1.2) and pipelineCacheUUID (always in props)
    // driverID lives in VkPhysicalDeviceDriverProperties via vkGetPhysicalDeviceProperties2
    uint32_t driverId = 0;
    {
        VkPhysicalDeviceDriverProperties driverProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
        VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &driverProps;
        vkGetPhysicalDeviceProperties2(phys, &props2);
        driverId = static_cast<uint32_t>(driverProps.driverID); // 0 if the struct isn't filled by older drivers
    }

    char buf[512];
    // Added pipelineCacheUUID for strict compatibility
    const std::string uuidHex = toHex(props.pipelineCacheUUID, VK_UUID_SIZE);

    // Prefer driverID when available, else fall back to driverVersion
    if (driverId != 0) {
        std::snprintf(buf, sizeof(buf),
            "pso_%04x_%04x_drv_%04x_api_%u.%u_uuid_%s.bin",
            props.vendorID, props.deviceID, driverId,
            VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
            uuidHex.c_str());
    }
    else {
        std::snprintf(buf, sizeof(buf),
            "pso_%04x_%04x_drvver_%08x_api_%u.%u_uuid_%s.bin",
            props.vendorID, props.deviceID, props.driverVersion,
            VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
            uuidHex.c_str());
    }

    std::error_code ec;
    fs::create_directories(fs::path(dir), ec); // best-effort
    return (fs::path(dir) / buf).string();
}

void PipelineCacheManager::init(VkPhysicalDevice phys, VkDevice dev, const std::string& dir) {
    device = dev;
    filePath = makeCachePath(phys, dir);

    // Try to load existing cache blob
    std::vector<char> initialData;
    std::error_code ec;
    if (fs::exists(filePath, ec) && !ec) {
        std::ifstream in(filePath, std::ios::binary | std::ios::ate);
        if (in) {
            const auto size = static_cast<size_t>(in.tellg());
            if (size > 0) {
                initialData.resize(size);
                in.seekg(0);
                in.read(initialData.data(), static_cast<std::streamsize>(size));
                // If short read, discard
                if (!in) initialData.clear();
            }
        }
    }

    VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    if (!initialData.empty()) {
        ci.initialDataSize = initialData.size();
        ci.pInitialData = initialData.data();
    }

    if (vkCreatePipelineCache(device, &ci, nullptr, &cache) != VK_SUCCESS) {
        cache = VK_NULL_HANDLE; // keep going without cache
    }
}

void PipelineCacheManager::save() const {
    if (!cache || !device || filePath.empty()) return;

    size_t size = 0;
    if (vkGetPipelineCacheData(device, cache, &size, nullptr) != VK_SUCCESS || size == 0) return;

    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device, cache, &size, data.data()) != VK_SUCCESS || size == 0) return;

    // Atomic write: dump to temp, then rename
    fs::path finalPath(filePath);
    fs::path tempPath = finalPath; tempPath += ".tmp";

    {
        std::ofstream out(tempPath, std::ios::binary | std::ios::trunc);
        if (!out) return; // best-effort
        out.write(data.data(), static_cast<std::streamsize>(size));
        out.flush();
        if (!out) return;
    }

    std::error_code ec;
    fs::rename(tempPath, finalPath, ec);
    if (ec) {
        // On Windows, rename can fail if target exists. Remove then rename.
        fs::remove(finalPath, ec);
        fs::rename(tempPath, finalPath, ec);
        // If it still fails, gives up quietly.
    }
}

void PipelineCacheManager::destroy() {
    if (cache) {
        save();
        vkDestroyPipelineCache(device, cache, nullptr);
        cache = VK_NULL_HANDLE;
    }
    device = VK_NULL_HANDLE;
    filePath.clear();
}
