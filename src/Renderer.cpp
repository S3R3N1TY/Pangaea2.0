#include "Renderer.hpp"

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
#ifdef VK_PIPELINE_CREATE_RENDERING_BIT_KHR
#define VK_PIPELINE_CREATE_RENDERING_BIT VK_PIPELINE_CREATE_RENDERING_BIT_KHR
#else
  // Old headers have neither. It's fine to set this to 0
  // as long as VkPipelineRenderingCreateInfo is chained via pNext.
#define VK_PIPELINE_CREATE_RENDERING_BIT 0
#endif
#endif

#ifndef VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
#ifdef VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR
#define VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR
#else
#define VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
#endif
#endif

// --------- file loader ----------
static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> data(size);
    file.seekg(0);
    file.read(data.data(), size);
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

    // Reusable staging uploader (1 MB initial)
    uploader.init(allocator, device, graphicsQueue, commandPool, 1ull << 20);


    // --- Swapchain-dependent setup (correct order so depthFormat is known) ---
    createSwapchain();
    createImageViews();
    createDepthResources();      // <-- moved before render pass
    createDescriptorSetLayout(); // created once for lifetime of renderer
    createGraphicsPipeline();

    // --- Resources not tied to swapchain count ---
    createVertexBuffer();
    createIndexBuffer();

    // --- Per-swapchain-image resources ---
    createUniformBuffers();
    createDescriptorPool();
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

    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    uploader.destroy();

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

    VkPhysicalDeviceFeatures features{}; // leave default for now

    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    // Request dynamic rendering (core in 1.3)
    VkPhysicalDeviceDynamicRenderingFeatures dyn{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES };
    dyn.dynamicRendering = VK_TRUE;

    VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.pNext = &dyn; // chain features
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
    // Perfect match: sRGB + nonlinear color space
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    }
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_R8G8B8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    }
    // Fallback: takes the first available.
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
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

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
        info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &info, nullptr, &swapchainImageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image view");
    }
}

VkFormat Renderer::findDepthFormat() {
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };
    for (VkFormat f : candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, f, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return f;
        }
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
    if (vkCreateImageView(device, &view, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view");
    }
    return imageView;
}

void Renderer::createDepthResources() {
    depthFormat = findDepthFormat();
    createImage(
        swapchainExtent.width, swapchainExtent.height,
        depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        depthImage, depthAlloc
    );
    depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
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

    VkPipelineShaderStageCreateInfo vertStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    // Vertex input
    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = offsetof(Vertex, pos);
    attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bind;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f; viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapchainExtent.width);
    viewport.height = static_cast<float>(swapchainExtent.height);
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;

    VkRect2D scissor{ {0,0}, swapchainExtent };

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;  viewportState.pScissors = &scissor;

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

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    // Dynamic rendering attachment formats
    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &swapchainImageFormat;
    rendering.depthAttachmentFormat = depthFormat;
    rendering.stencilAttachmentFormat = VK_FORMAT_UNDEFINED; // not using stencil

    VkGraphicsPipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeInfo.pNext = &rendering; // tie formats to pipeline
    pipeInfo.flags = VK_PIPELINE_CREATE_RENDERING_BIT;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &raster;
    pipeInfo.pMultisampleState = &msaa;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlend;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = VK_NULL_HANDLE; // no render pass
    pipeInfo.subpass = 0;              // ignored in dynamic rendering

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create graphics pipeline");

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

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin command buffer");

    // --- Layout transitions (classic barrier for compatibility) ---
    // Swapchain image: UNDEFINED or PRESENT_SRC -> COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier colorBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    colorBarrier.srcAccessMask = 0;
    colorBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;          // ok each frame with discard/clear
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBarrier.image = swapchainImages[imageIndex];
    colorBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorBarrier.subresourceRange.baseMipLevel = 0;
    colorBarrier.subresourceRange.levelCount = 1;
    colorBarrier.subresourceRange.baseArrayLayer = 0;
    colorBarrier.subresourceRange.layerCount = 1;

    // Depth image: UNDEFINED or whatever -> DEPTH_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier depthBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    depthBarrier.srcAccessMask = 0;
    depthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; // 1.3 name, maps to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.image = depthImage;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier.subresourceRange.baseMipLevel = 0;
    depthBarrier.subresourceRange.levelCount = 1;
    depthBarrier.subresourceRange.baseArrayLayer = 0;
    depthBarrier.subresourceRange.layerCount = 1;

    std::array<VkImageMemoryBarrier, 2> barriers = { colorBarrier, depthBarrier };
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0,
        0, nullptr,
        0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data()
    );

    // --- Dynamic rendering begin ---
    VkClearValue clearColor{};     clearColor.color = { { 0.07f, 0.17f, 0.33f, 1.0f } };
    VkClearValue clearDepth{};     clearDepth.depthStencil = { 1.0f, 0 };

    VkRenderingAttachmentInfo colorAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    colorAtt.imageView = swapchainImageViews[imageIndex];
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue = clearColor;

    VkRenderingAttachmentInfo depthAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthAtt.imageView = depthImageView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; // 1.3 name
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
    rendering.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &rendering);

    // Pipeline + descriptors + draws
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
        &descriptorSets[imageIndex], 0, nullptr);

    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(gIndices.size()), 1, 0, 0, 0);

    vkCmdEndRendering(cmd);

    // Transition swapchain image to PRESENT for presentation
    VkImageMemoryBarrier toPresent{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toPresent.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toPresent.dstAccessMask = 0;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.image = swapchainImages[imageIndex];
    toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toPresent.subresourceRange.baseMipLevel = 0;
    toPresent.subresourceRange.levelCount = 1;
    toPresent.subresourceRange.baseArrayLayer = 0;
    toPresent.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toPresent
    );

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
    }
    imagesInFlight.assign(swapchainImages.size(), VK_NULL_HANDLE);
}

void Renderer::destroySwapchainObjects() {
    if (!commandBuffers.empty()) {
        vkFreeCommandBuffers(device, commandPool,
            static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        commandBuffers.clear();
        
    }
    if (descriptorPool) { vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE; }

    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
        if (uniformBuffers[i])  vmaDestroyBuffer(allocator, uniformBuffers[i], uniformAllocs[i]);
    }
    uniformBuffers.clear();
    uniformAllocs.clear();
    uniformMapped.clear();
    descriptorSets.clear();

    if (graphicsPipeline) { vkDestroyPipeline(device, graphicsPipeline, nullptr); graphicsPipeline = VK_NULL_HANDLE; }
    if (pipelineLayout) { vkDestroyPipelineLayout(device, pipelineLayout, nullptr); pipelineLayout = VK_NULL_HANDLE; }

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
    createDepthResources();      // <-- before render pass
    createGraphicsPipeline();

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();

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
    uniformBuffers.resize(swapchainImages.size());
    uniformAllocs.resize(swapchainImages.size());
    uniformMapped.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        void* mapped = nullptr;
        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniformBuffers[i], uniformAllocs[i], &mapped);
        uniformMapped[i] = mapped;
    }
}

void Renderer::updateUniformBuffer(uint32_t imageIndex) {
    float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), t, glm::vec3(0.f, 0.f, 1.f));
    glm::mat4 view = glm::lookAt(glm::vec3(0.f, 0.f, 1.5f),
        glm::vec3(0.f, 0.f, 0.f),
        glm::vec3(0.f, 1.f, 0.f));
    float aspect = swapchainExtent.width / static_cast<float>(std::max(1u, swapchainExtent.height));
    glm::mat4 proj = glm::perspective(glm::radians(60.f), aspect, 0.01f, 10.f);
    proj[1][1] *= -1.f;

    glm::mat4 mvp = proj * view * model;

    UniformBufferObject u{};
    std::memcpy(u.mvp, &mvp[0][0], sizeof(u.mvp));

    // Persistently mapped UBO: write then FLUSH
    std::memcpy(uniformMapped[imageIndex], &u, sizeof(u));
    vmaFlushAllocation(allocator, uniformAllocs[imageIndex], 0, sizeof(u));
}


void Renderer::createDescriptorPool() {
    VkDescriptorPoolSize pool{};
    pool.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool.descriptorCount = static_cast<uint32_t>(swapchainImages.size());

    VkDescriptorPoolCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    info.poolSizeCount = 1;
    info.pPoolSizes = &pool;
    info.maxSets = static_cast<uint32_t>(swapchainImages.size());

    if (vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool");
}

void Renderer::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(swapchainImages.size(), descriptorSetLayout);
    VkDescriptorSetAllocateInfo alloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc.descriptorPool = descriptorPool;
    alloc.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    alloc.pSetLayouts = layouts.data();

    descriptorSets.resize(layouts.size());
    if (vkAllocateDescriptorSets(device, &alloc, descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor sets");

    for (size_t i = 0; i < descriptorSets.size(); ++i) {
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
