#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>

class Renderer {
public:
    void init();
    void cleanup();
    void drawFrame();

private:
    VkInstance instance{};
    VkDebugUtilsMessengerEXT debugMessenger{};

    void createInstance();
    void setupDebugMessenger();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
};
