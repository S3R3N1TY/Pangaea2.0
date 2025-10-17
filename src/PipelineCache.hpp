#pragma once
#include <vulkan/vulkan.h>
#include <string>

class PipelineCacheManager {
public:
    void init(VkPhysicalDevice phys, VkDevice dev, const std::string& dir = "cache");
    void destroy();

    VkPipelineCache get() const { return cache; }
    const std::string& path() const { return filePath; }

private:
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::string filePath;
};
