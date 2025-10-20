#pragma once
#include <vulkan/vulkan.h>
#include <string>

class PipelineCacheManager {
public:
    PipelineCacheManager() = default;
    ~PipelineCacheManager() { destroy(); }

    PipelineCacheManager(const PipelineCacheManager&) = delete;
    PipelineCacheManager& operator=(const PipelineCacheManager&) = delete;

    PipelineCacheManager(PipelineCacheManager&& other) noexcept { moveFrom(other); }
    PipelineCacheManager& operator=(PipelineCacheManager&& other) noexcept {
        if (this != &other) { destroy(); moveFrom(other); }
        return *this;
    }

    void init(VkPhysicalDevice phys, VkDevice dev, const std::string& dir = "cache");
    void destroy();

    void save() const;

    VkPipelineCache get() const { return cache; }
    const std::string& path() const { return filePath; }

private:
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::string filePath;

    void moveFrom(PipelineCacheManager& other) noexcept {
        device = other.device;   other.device = VK_NULL_HANDLE;
        cache = other.cache;    other.cache = VK_NULL_HANDLE;
        filePath = std::move(other.filePath);
    }
};
