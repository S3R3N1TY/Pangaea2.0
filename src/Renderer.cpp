#include "Renderer.hpp"
#include "PipelineBuilder.hpp"

#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <fstream>
#include <array>
#include <cstring>

// glm for MVP
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// --- Vulkan 1.3 dynamic rendering compatibility shims ---
// Works with 1.3 core, KHR-only headers, or prehistoric headers.
#ifndef VK_PIPELINE_CREATE_RENDERING_BIT
# ifdef VK_PIPELINE_CREATE_RENDERING_BIT_KHR
#  define VK_PIPELINE_CREATE_RENDERING_BIT VK_PIPELINE_CREATE_RENDERING_BIT_KHR
# else
   // Old headers have neither. It's fine to set this to 0
   // as long as VkPipelineRenderingCreateInfo is chained via pNext.
#  define VK_PIPELINE_CREATE_RENDERING_BIT 0
# endif
#endif

#ifndef VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
# ifdef VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR
#  define VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR
# else
#  define VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
# endif
#endif

// --------- file loader (safer) ----------
static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Shader open failed: " + path);
    const auto size = static_cast<size_t>(f.tellg());
    if (size == 0) throw std::runtime_error("Shader empty or unreadable: " + path);
    std::vector<char> data(size);
    f.seekg(0);
    f.read(data.data(), static_cast<std::streamsize>(size));
    if (!f) throw std::runtime_error("Shader short read: " + path);
    return data;
}

// ---------------- Debug callback ----------------
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    (void)messageSeverity; (void)messageType; (void)pUserData;
    std::cerr << "[Vulkan] " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

static bool hasInstanceExtension(const char* name) {
    uint32_t n = 0; vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
    for (auto& e : exts) if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}
static bool hasLayer(const char* name) {
    uint32_t n = 0; vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> layers(n);
    vkEnumerateInstanceLayerProperties(&n, layers.data());
    for (auto& l : layers) if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}

// Device-level debug utils function pointers for naming/markers
static PFN_vkSetDebugUtilsObjectNameEXT pSetName = nullptr;
static PFN_vkCmdBeginDebugUtilsLabelEXT pBeginLabel = nullptr;
static PFN_vkCmdEndDebugUtilsLabelEXT   pEndLabel = nullptr;

// ---------------- Vertex data ----------------
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};
static const std::vector<Vertex> gVertices = {
    {{ 0.0f, -0.5f, 0.0f}, {1.f, 0.f, 0.f}},
    {{ 0.5f,  0.5f, 0.0f}, {0.f, 1.f, 0.f}},
    {{-0.5f,  0.5f, 0.0f}, {0.f, 0.f, 1.f}},
};
static const std::vector<uint16_t> gIndices = { 0, 1, 2 };

// ---------------- Public API ----------------
void Renderer::init(GLFWwindow* window) {
    windowHandle = window;
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();       // VMA
    createCommandPool();     // needed for staging and one-shot cmds

    // Reusable staging uploader
    uploader.init(allocator, device, graphicsQueue, commandPool, 1ull << 20);
    pipelineCache.init(physicalDevice, device, "cache");

    // --- Swapchain-dependent setup (correct order so depthFormat is known) ---
    createSwapchain();
    createImageViews();
    createDepthResources();      // depth before pipeline so formats are known
    createDescriptorSetLayout(); // created once for lifetime of renderer
    createGraphicsPipeline();

    // --- Resources not tied to swapchain count ---
    createVertexBuffer();
    createIndexBuffer();

    // --- Per-swapchain-image resources ---
    createUniformBuffers();
    descriptorArena.init(device);
    createDescriptorSets();

    createCommandBuffers();
    createSyncObjects();
}

void Renderer::cleanup() {
    vkDeviceWaitIdle(device);

    // global non-swapchain resources
    if (indexBuffer) { vmaDestroyBuffer(allocator, indexBuffer, indexAlloc); indexBuffer = VK_NULL_HANDLE; }
    if (vertexBuffer) { vmaDestroyBuffer(allocator, vertexBuffer, vertexAlloc); vertexBuffer = VK_NULL_HANDLE; }

    destroySwapchainObjects();

    // Kill pipeline objects kept across resizes
    if (graphicsPipeline) {
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }

    descriptorArena.destroy();

    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    uploader.destroy();
    pipelineCache.destroy();

    if (commandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    // Per-frame sync
    for (size_t i = 0; i < inFlightFences.size(); ++i) {
        if (imageAvailableSemaphores[i]) vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        if (inFlightFences[i])          vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    imageAvailableSemaphores.clear();
    inFlightFences.clear();

    // Per-image semaphores
    for (auto sem : renderFinishedSemaphores) {
        if (sem) vkDestroySemaphore(device, sem, nullptr);
    }
    renderFinishedSemaphores.clear();
    imagesInFlight.clear();

    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
        if (uniformBuffers[i]) {
            vmaDestroyBuffer(allocator, uniformBuffers[i], uniformAllocs[i]);
            uniformBuffers[i] = VK_NULL_HANDLE;
            uniformAllocs[i] = VK_NULL_HANDLE;
        }
    }
    uniformBuffers.clear();
    uniformAllocs.clear();
    uniformMapped.clear();

    if (allocator) {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (device) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (surface) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }

    auto destroyDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroyDebugUtilsMessengerEXT && debugMessenger)
        destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

void Renderer::drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) throw std::runtime_error("Failed to acquire swapchain image");

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    updateUniformBuffer(imageIndex);

    vkResetCommandBuffer(commandBuffers[imageIndex], 0);
    recordCommandBuffer(commandBuffers[imageIndex], imageIndex);

    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphores[imageIndex];

    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("Failed to submit draw");

    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    VkResult pres = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapchain();
    }
    else if (pres != VK_SUCCESS) {
        throw std::runtime_error("Failed to present");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ---------------- Internals ----------------
void Renderer::createInstance() {
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "Pangaea 2.0";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "Custom";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t glfwExtCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtCount);

    bool wantDebug = hasLayer("VK_LAYER_KHRONOS_validation") && hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (wantDebug) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    std::vector<const char*> layers;
    if (wantDebug) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = &debugCreateInfo;
    }
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan instance");
}

void Renderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
}

void Renderer::setupDebugMessenger() {
    auto createDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (!createDebugUtilsMessengerEXT) {
        debugMessenger = VK_NULL_HANDLE;
        return;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);
    if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to set up debug messenger");
}

void Renderer::createSurface() {
    if (glfwCreateWindowSurface(instance, windowHandle, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
}

Renderer::QueueFamilyIndices Renderer::findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport)
            indices.presentFamily = i;

        if (indices.isComplete()) break;
    }
    return indices;
}

Renderer::SwapSupportDetails Renderer::querySwapSupport(VkPhysicalDevice dev) {
    SwapSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &details.capabilities);

    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, nullptr);
    details.formats.resize(count);
    if (count) vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, details.formats.data());

    count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, nullptr);
    details.presentModes.resize(count);
    if (count) vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, details.presentModes.data());

    return details;
}

bool Renderer::isDeviceSuitable(VkPhysicalDevice dev) {
    auto indices = findQueueFamilies(dev);
    if (!indices.isComplete()) return false;

    auto support = querySwapSupport(dev);
    bool swapAdequate = !support.formats.empty() && !support.presentModes.empty();
    return swapAdequate;
}

void Renderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No Vulkan-compatible GPU found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (auto dev : devices) {
        if (isDeviceSuitable(dev)) { physicalDevice = dev; break; }
    }
    if (physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("No suitable GPU found");
}

void Renderer::createLogicalDevice() {
    auto indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    float priority = 1.0f;
    std::vector<uint32_t> uniqueFamilies;
    if (indices.graphicsFamily.value() == indices.presentFamily.value())
        uniqueFamilies = { indices.graphicsFamily.value() };
    else
        uniqueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    for (uint32_t fam : uniqueFamilies) {
        VkDeviceQueueCreateInfo q{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        q.queueFamilyIndex = fam;
        q.queueCount = 1;
        q.pQueuePriorities = &priority;
        queueInfos.push_back(q);
    }

    VkPhysicalDeviceFeatures features{}; // default

    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    // --- Features chain: Dynamic Rendering + Synchronization2 (core in 1.3) ---
    VkPhysicalDeviceDynamicRenderingFeatures dyn{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
    };
    dyn.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceSynchronization2Features sync2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
    };
    sync2.synchronization2 = VK_TRUE;

    // chain head -> next
    dyn.pNext = &sync2;

    VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.pNext = &dyn;  // head of the chain
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.pEnabledFeatures = &features;
    createInfo.enabledExtensionCount = 1;
    createInfo.ppEnabledExtensionNames = deviceExtensions;
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

    // Load device-level debug utils (Safe if extension missing)
    pSetName = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
    pBeginLabel = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT");
    pEndLabel = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT");
}

void Renderer::createAllocator() {
    VmaAllocatorCreateInfo info{};
    info.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    info.instance = instance;
    info.physicalDevice = physicalDevice;
    info.device = device;
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    if (vmaCreateAllocator(&info, &allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create VMA allocator");
}


VkSurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    if (formats.empty()) throw std::runtime_error("No surface formats reported by the device");

    // Prefer sRGB formats for correct presentation gamma
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    }
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_R8G8B8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    }
    // Fallback: first available format
    return formats[0];
}

VkPresentModeKHR Renderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Renderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return caps.currentExtent;

    int w = 0, h = 0;
    glfwGetFramebufferSize(windowHandle, &w, &h);
    VkExtent2D extent{ static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
    extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return extent;
}

void Renderer::createSwapchain() {
    auto support = querySwapSupport(physicalDevice);

    auto surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    auto presentMode = chooseSwapPresentMode(support.presentModes);
    auto extent = chooseSwapExtent(support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // add TRANSFER_* later if you need screenshots/post

    auto indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swapchain");

    // Name the swapchain for sanity in RenderDoc
    if (pSetName) {
        VkDebugUtilsObjectNameInfoEXT n{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        n.objectType = VK_OBJECT_TYPE_SWAPCHAIN_KHR;
        n.objectHandle = (uint64_t)swapchain;
        n.pObjectName = "Swapchain";
        pSetName(device, &n);
    }

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;
}

void Renderer::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        VkImageViewCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        info.image = swapchainImages[i];
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = swapchainImageFormat;
        info.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
        };
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &info, nullptr, &swapchainImageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image view");

        // Name the image view
        if (pSetName) {
            VkDebugUtilsObjectNameInfoEXT n{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
            n.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
            n.objectHandle = (uint64_t)swapchainImageViews[i];
            // Tiny, readable labels: "SwapView[0]", "SwapView[1]"...
            char label[32]; std::snprintf(label, sizeof(label), "SwapView[%zu]", i);
            n.pObjectName = label;
            pSetName(device, &n);
        }
    }
}

VkFormat Renderer::findDepthFormat() {
    // Prefer stencil-less unless need stencil later?
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };
    for (VkFormat f : candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, f, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return f;
    }
    throw std::runtime_error("Failed to find supported depth format");
}

bool Renderer::hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void Renderer::createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
    VkImage& image, VmaAllocation& alloc) {
    VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = { w, h, 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.format = format;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateImage(allocator, &info, &allocInfo, &image, &alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image");
}

VkImageView Renderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo view{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    view.image = image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange.aspectMask = aspect;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;

    VkImageView imageView{};
    if (vkCreateImageView(device, &view, nullptr, &imageView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image view");
    return imageView;
}

void Renderer::createDepthResources() {
    depthFormat = findDepthFormat();
    createImage(
        swapchainExtent.width, swapchainExtent.height,
        depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT /* | VK_IMAGE_USAGE_SAMPLED_BIT */,
        depthImage, depthAlloc
    );
    depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Debug names
    if (pSetName) {
        VkDebugUtilsObjectNameInfoEXT n{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        n.objectType = VK_OBJECT_TYPE_IMAGE; n.objectHandle = (uint64_t)depthImage; n.pObjectName = "DepthImage";
        pSetName(device, &n);
        n = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        n.objectType = VK_OBJECT_TYPE_IMAGE_VIEW; n.objectHandle = (uint64_t)depthImageView; n.pObjectName = "DepthView";
        pSetName(device, &n);
    }
}

void Renderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding ubo{};
    ubo.binding = 0;
    ubo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo.descriptorCount = 1;
    ubo.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    info.bindingCount = 1;
    info.pBindings = &ubo;

    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor set layout");
}

VkShaderModule Renderer::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module{};
    if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
    return module;
}

void Renderer::createGraphicsPipeline() {
    const std::string base = "shaders/";
    auto vertCode = readFile(base + "triangle.vert.spv");
    auto fragCode = readFile(base + "triangle.frag.spv");

    VkShaderModule vertModule = createShaderModule(vertCode);
    VkShaderModule fragModule = createShaderModule(fragCode);

    // Vertex input layout
    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = offsetof(Vertex, pos);
    attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(Vertex, color);

    // Fixed states
    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.depthClampEnable = VK_FALSE;
    raster.rasterizerDiscardEnable = VK_FALSE;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable = VK_FALSE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorAttachment{};
    colorAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorAttachment.blendEnable = VK_FALSE;

    // Pipeline layout with push constants
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset = 0;
    pc.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pc;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    // Build via PipelineBuilder
    PipelineBuilder pb;
    pb.clearStages()
        .addStage(VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main")
        .addStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main")
        .setVertexInput(&bind, 1, attrs, 2)
        .setInputAssembly(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE)
        .setViewport(0.f, 0.f, (float)swapchainExtent.width, (float)swapchainExtent.height)  // ignored if dynamic
        .setScissor(0, 0, swapchainExtent.width, swapchainExtent.height)                      // ignored if dynamic
        .setRasterization(raster)
        .setMultisample(msaa)
        .setDepthStencil(depthStencil)
        .setColorBlendAttachments({ colorAttachment })
        .setLayout(pipelineLayout)
        .setRenderingFormats({ swapchainImageFormat }, depthFormat)
        .setPipelineCache(pipelineCache.get())
        .setDynamicStates({ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR });

    graphicsPipeline = pb.build(device);

    // Name pipeline & layout
    if (pSetName) {
        VkDebugUtilsObjectNameInfoEXT n{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        n.objectType = VK_OBJECT_TYPE_PIPELINE; n.objectHandle = (uint64_t)graphicsPipeline; n.pObjectName = "TrianglePipeline";
        pSetName(device, &n);
        n = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        n.objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT; n.objectHandle = (uint64_t)pipelineLayout; n.pObjectName = "MainLayout";
        pSetName(device, &n);
    }

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);
}

void Renderer::createCommandPool() {
    auto indices = findQueueFamilies(physicalDevice);
    VkCommandPoolCreateInfo info{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    info.queueFamilyIndex = indices.graphicsFamily.value();
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &info, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
}

void Renderer::createCommandBuffers() {
    commandBuffers.resize(swapchainImages.size());
    VkCommandBufferAllocateInfo info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    info.commandPool = commandPool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    if (vkAllocateCommandBuffers(device, &info, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");
}

// ==================== Renderer::recordCommandBuffer (Sync2 barriers + dynamic viewport/scissor) ====================
void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin command buffer");

    // --- Sync2: begin-of-pass image layout transitions ---
    VkImageMemoryBarrier2 colorBarrier2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    colorBarrier2.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    colorBarrier2.srcAccessMask = 0;
    colorBarrier2.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier2.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;              // discard on clear
    colorBarrier2.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBarrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBarrier2.image = swapchainImages[imageIndex];
    colorBarrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorBarrier2.subresourceRange.baseMipLevel = 0;
    colorBarrier2.subresourceRange.levelCount = 1;
    colorBarrier2.subresourceRange.baseArrayLayer = 0;
    colorBarrier2.subresourceRange.layerCount = 1;

    VkImageMemoryBarrier2 depthBarrier2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    depthBarrier2.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    depthBarrier2.srcAccessMask = 0;
    depthBarrier2.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    depthBarrier2.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier2.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; // core 1.3 name
    depthBarrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier2.image = depthImage;
    depthBarrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier2.subresourceRange.baseMipLevel = 0;
    depthBarrier2.subresourceRange.levelCount = 1;
    depthBarrier2.subresourceRange.baseArrayLayer = 0;
    depthBarrier2.subresourceRange.layerCount = 1;

    std::array<VkImageMemoryBarrier2, 2> imgBarriers{ colorBarrier2, depthBarrier2 };
    VkDependencyInfo depBegin{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depBegin.imageMemoryBarrierCount = static_cast<uint32_t>(imgBarriers.size());
    depBegin.pImageMemoryBarriers = imgBarriers.data();
    // no memory/buffer barriers in this batch
    vkCmdPipelineBarrier2(cmd, &depBegin);

    // --- Dynamic rendering begin ---
    VkClearValue clearColor{}; clearColor.color = { { 0.00f, 0.00f, 0.00f, 1.0f } };
    VkClearValue clearDepth{}; clearDepth.depthStencil = { 1.0f, 0 };

    VkRenderingAttachmentInfo colorAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    colorAtt.imageView = swapchainImageViews[imageIndex];
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue = clearColor;

    VkRenderingAttachmentInfo depthAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthAtt.imageView = depthImageView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.clearValue = clearDepth;

    VkRenderingInfo rendering{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    rendering.renderArea.offset = { 0, 0 };
    rendering.renderArea.extent = swapchainExtent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &colorAtt;
    rendering.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &rendering);

    // --- Pipeline + dynamic viewport/scissor ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport vp{};
    vp.x = 0.f; vp.y = 0.f;
    vp.width = static_cast<float>(swapchainExtent.width);
    vp.height = static_cast<float>(swapchainExtent.height);
    vp.minDepth = 0.f; vp.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D sc{ {0, 0}, swapchainExtent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // --- Bind geometry & descriptors ---
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
        &descriptorSets[currentFrame], 0, nullptr);

    // --- Push constants (per-object model) ---
    float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), t, glm::vec3(0.f, 0.f, 1.f));
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &model);

    // --- Draw ---
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(gIndices.size()), 1, 0, 0, 0);

    vkCmdEndRendering(cmd);

    // --- Sync2: transition color to PRESENT ---
    VkImageMemoryBarrier2 toPresent2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    toPresent2.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toPresent2.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toPresent2.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    toPresent2.dstAccessMask = 0;
    toPresent2.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toPresent2.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent2.image = swapchainImages[imageIndex];
    toPresent2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toPresent2.subresourceRange.baseMipLevel = 0;
    toPresent2.subresourceRange.levelCount = 1;
    toPresent2.subresourceRange.baseArrayLayer = 0;
    toPresent2.subresourceRange.layerCount = 1;

    VkDependencyInfo depEnd{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depEnd.imageMemoryBarrierCount = 1;
    depEnd.pImageMemoryBarriers = &toPresent2;

    vkCmdPipelineBarrier2(cmd, &depEnd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("Failed to record command buffer");
}


void Renderer::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    renderFinishedSemaphores.resize(swapchainImages.size());
    imagesInFlight.resize(swapchainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo sem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fence{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(device, &sem, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fence, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create per-frame sync objects");
        }

        // Names for sanity in tools
        if (pSetName) {
            VkDebugUtilsObjectNameInfoEXT n{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
            n.objectType = VK_OBJECT_TYPE_SEMAPHORE; n.objectHandle = (uint64_t)imageAvailableSemaphores[i];
            char labelA[32]; std::snprintf(labelA, sizeof(labelA), "ImgAvail[%d]", i); n.pObjectName = labelA; pSetName(device, &n);
            n = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
            n.objectType = VK_OBJECT_TYPE_FENCE; n.objectHandle = (uint64_t)inFlightFences[i];
            char labelF[32]; std::snprintf(labelF, sizeof(labelF), "InFlight[%d]", i); n.pObjectName = labelF; pSetName(device, &n);
        }
    }

    recreatePerImageSemaphores();
}

void Renderer::recreatePerImageSemaphores() {
    for (auto s : renderFinishedSemaphores) if (s) vkDestroySemaphore(device, s, nullptr);
    renderFinishedSemaphores.clear();
    renderFinishedSemaphores.resize(swapchainImages.size());

    VkSemaphoreCreateInfo sem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        if (vkCreateSemaphore(device, &sem, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create per-image renderFinished semaphore");

        // Name
        if (pSetName) {
            VkDebugUtilsObjectNameInfoEXT n{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
            n.objectType = VK_OBJECT_TYPE_SEMAPHORE;
            n.objectHandle = (uint64_t)renderFinishedSemaphores[i];
            char label[32]; std::snprintf(label, sizeof(label), "RenderDone[%zu]", i);
            n.pObjectName = label;
            pSetName(device, &n);
        }
    }
    imagesInFlight.assign(swapchainImages.size(), VK_NULL_HANDLE);
}

void Renderer::destroySwapchainObjects() {
    if (!commandBuffers.empty()) {
        vkFreeCommandBuffers(device, commandPool,
            static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        commandBuffers.clear();
    }

    if (depthImageView) { vkDestroyImageView(device, depthImageView, nullptr); depthImageView = VK_NULL_HANDLE; }
    if (depthImage) { vmaDestroyImage(allocator, depthImage, depthAlloc); depthImage = VK_NULL_HANDLE; depthAlloc = VK_NULL_HANDLE; }

    for (auto view : swapchainImageViews) vkDestroyImageView(device, view, nullptr);
    swapchainImageViews.clear();
    swapchainImages.clear();

    if (swapchain) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

void Renderer::recreateSwapchain() {
    int width = 0, height = 0;
    do {
        glfwGetFramebufferSize(windowHandle, &width, &height);
        if (width == 0 || height == 0) glfwWaitEvents();
    } while (width == 0 || height == 0);

    vkDeviceWaitIdle(device);

    destroySwapchainObjects();

    // --- Rebuild in the correct order ---
    createSwapchain();
    createImageViews();
    createDepthResources();      // depth before pipeline if rebuild here

    createCommandBuffers();
    recreatePerImageSemaphores();
}


// ---------------- Buffer helpers (VMA) ----------------
void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkBuffer& buffer, VmaAllocation& alloc, void** mapped) {
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (mapped) {
        aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }

    VmaAllocationInfo out{};
    if (vmaCreateBuffer(allocator, &bi, &aci, &buffer, &alloc, &out) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");
    if (mapped) *mapped = out.pMappedData;
}

VkCommandBuffer Renderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo alloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc.commandPool = commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd{};
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

// ================== StagingUploader implementation ==================
void Renderer::StagingUploader::init(VmaAllocator alloc, VkDevice dev, VkQueue q, VkCommandPool pool, VkDeviceSize initialCapacity) {
    allocator = alloc;
    device = dev;
    queue = q;
    cmdPool = pool;

    // Create fence once
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(device, &fi, nullptr, &copyFence) != VK_SUCCESS) {
        throw std::runtime_error("StagingUploader: failed to create fence");
    }

    // Allocate initial staging buffer
    ensureCapacity(initialCapacity);
}

void Renderer::StagingUploader::destroy() {
    if (stagingBuffer) {
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
        stagingBuffer = VK_NULL_HANDLE;
        stagingAlloc = VK_NULL_HANDLE;
        mapped = nullptr;
        capacity = 0;
    }
    if (copyFence) {
        vkDestroyFence(device, copyFence, nullptr);
        copyFence = VK_NULL_HANDLE;
    }
    allocator = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    cmdPool = VK_NULL_HANDLE;
}

void Renderer::StagingUploader::ensureCapacity(VkDeviceSize requiredBytes) {
    if (requiredBytes <= capacity && stagingBuffer != VK_NULL_HANDLE) return;

    // Destroy old
    if (stagingBuffer) {
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
        stagingBuffer = VK_NULL_HANDLE;
        stagingAlloc = VK_NULL_HANDLE;
        mapped = nullptr;
    }

    // Grow with a little headroom to reduce reallocs
    capacity = std::max<VkDeviceSize>(requiredBytes, capacity ? capacity * 2 : (1ull << 20)); // min 1 MB

    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = capacity;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocationInfo out{};
    if (vmaCreateBuffer(allocator, &bi, &aci, &stagingBuffer, &stagingAlloc, &out) != VK_SUCCESS) {
        throw std::runtime_error("StagingUploader: failed to create staging buffer");
    }
    mapped = out.pMappedData;
}

VkCommandBuffer Renderer::StagingUploader::allocateOneShotCmd() const {
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd{};
    if (vkAllocateCommandBuffers(device, &ai, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("StagingUploader: failed to allocate command buffer");
    }
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        throw std::runtime_error("StagingUploader: failed to begin command buffer");
    }
    return cmd;
}

void Renderer::StagingUploader::submitAndWait(VkCommandBuffer cmd) {
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        throw std::runtime_error("StagingUploader: failed to end command buffer");
    }
    // Reset fence to unsignaled for this submit
    vkResetFences(device, 1, &copyFence);

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    if (vkQueueSubmit(queue, 1, &si, copyFence) != VK_SUCCESS) {
        throw std::runtime_error("StagingUploader: queue submit failed");
    }

    // Wait only for this upload instead of stalling the whole queue with vkQueueWaitIdle
    vkWaitForFences(device, 1, &copyFence, VK_TRUE, UINT64_MAX);

    vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
}

void Renderer::StagingUploader::upload(const void* src, VkDeviceSize sizeBytes, VkBuffer dst, VkDeviceSize dstOffset) {
    if (sizeBytes == 0) return;
    ensureCapacity(sizeBytes);

    // Copy to mapped staging, then flush the written range
    std::memcpy(mapped, src, static_cast<size_t>(sizeBytes));
    vmaFlushAllocation(allocator, stagingAlloc, 0, sizeBytes);

    // Record copy
    VkCommandBuffer cmd = allocateOneShotCmd();
    VkBufferCopy region{ 0, dstOffset, sizeBytes };
    vkCmdCopyBuffer(cmd, stagingBuffer, dst, 1, &region);

    // Submit and wait for completion
    submitAndWait(cmd);
}

void Renderer::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copy{ 0, 0, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);
    endSingleTimeCommands(cmd);
}

void Renderer::createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkBuffer& buffer, VmaAllocation& alloc) {
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator, &bi, &aci, &buffer, &alloc, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create device-local buffer");
    }
}

// ---------------- Resource creation ----------------
void Renderer::createVertexBuffer() {
    const VkDeviceSize size = sizeof(decltype(gVertices)::value_type) * gVertices.size();

    // Device-local vertex buffer
    createDeviceLocalBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vertexBuffer, vertexAlloc);

    // Upload from reusable staging
    uploader.upload(gVertices.data(), size, vertexBuffer, 0);
}

void Renderer::createIndexBuffer() {
    const VkDeviceSize size = sizeof(uint16_t) * gIndices.size();

    // Device-local index buffer
    createDeviceLocalBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        indexBuffer, indexAlloc);

    // Upload from reusable staging
    uploader.upload(gIndices.data(), size, indexBuffer, 0);
}

void Renderer::createUniformBuffers() {
    const VkDeviceSize size = sizeof(UniformBufferObject);
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformAllocs.resize(MAX_FRAMES_IN_FLIGHT);
    uniformMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        void* mapped = nullptr;
        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniformBuffers[i], uniformAllocs[i], &mapped);
        uniformMapped[i] = mapped;
    }
}

void Renderer::updateUniformBuffer(uint32_t /*imageIndex*/) {
    // Use currentFrame for per-frame UBO indexing
    const uint32_t idx = currentFrame;

    // Time
    float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

    // Camera
    glm::mat4 view = glm::lookAt(glm::vec3(0.f, 0.f, 1.5f),
        glm::vec3(0.f, 0.f, 0.f),
        glm::vec3(0.f, 1.f, 0.f));
    float aspect = swapchainExtent.width / static_cast<float>(std::max(1u, swapchainExtent.height));
    glm::mat4 proj = glm::perspective(glm::radians(60.f), aspect, 0.01f, 10.f);
    proj[1][1] *= -1.f;

    glm::mat4 vp = proj * view;

    UniformBufferObject u{};
    std::memcpy(u.vp, &vp[0][0], sizeof(u.vp));

    std::memcpy(uniformMapped[idx], &u, sizeof(u));
    vmaFlushAllocation(allocator, uniformAllocs[idx], 0, sizeof(u));
}

void Renderer::createDescriptorSets() {
    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        descriptorSets[i] = descriptorArena.allocate(descriptorSetLayout);

        VkDescriptorBufferInfo buf{};
        buf.buffer = uniformBuffers[i];
        buf.offset = 0;
        buf.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = descriptorSets[i];
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &buf;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

