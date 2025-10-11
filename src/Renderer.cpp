#include "Renderer.hpp"
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>

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

// ---------------- Public API ----------------
void Renderer::init(GLFWwindow* window) {
    windowHandle = window;
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

void Renderer::cleanup() {
    vkDeviceWaitIdle(device);

    destroySwapchainObjects();

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
    // Wait for the current frame to finish
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Acquire next image
    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    else if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    // Ensure the image isn't still in flight
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    // Re-record for this image
    vkResetCommandBuffer(commandBuffers[imageIndex], 0);
    recordCommandBuffer(commandBuffers[imageIndex], imageIndex);

    // Submit, signaling the per-image semaphore
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
        throw std::runtime_error("Failed to submit draw command buffer");

    // Present waits on that same per-image semaphore
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
        throw std::runtime_error("Failed to present swapchain image");
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
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo createInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = &debugCreateInfo;

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
        if (isDeviceSuitable(dev)) {
            physicalDevice = dev;
            break;
        }
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

    VkPhysicalDeviceFeatures features{};
    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.pEnabledFeatures = &features;
    createInfo.enabledExtensionCount = 1;
    createInfo.ppEnabledExtensionNames = deviceExtensions;

    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = layers;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

VkSurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    }
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

void Renderer::createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo info{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    info.attachmentCount = 1;
    info.pAttachments = &colorAttachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.pDependencies = &dep;
    info.dependencyCount = 1;

    if (vkCreateRenderPass(device, &info, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create render pass");
}

void Renderer::createFramebuffers() {
    framebuffers.resize(swapchainImageViews.size());
    for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
        VkImageView attachments[] = { swapchainImageViews[i] };
        VkFramebufferCreateInfo info{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        info.renderPass = renderPass;
        info.attachmentCount = 1;
        info.pAttachments = attachments;
        info.width = swapchainExtent.width;
        info.height = swapchainExtent.height;
        info.layers = 1;

        if (vkCreateFramebuffer(device, &info, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer");
    }
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
    commandBuffers.resize(framebuffers.size());
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

    VkClearValue clear{};
    clear.color = { { 0.07f, 0.17f, 0.33f, 1.0f } };

    VkRenderPassBeginInfo rp{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rp.renderPass = renderPass;
    rp.framebuffer = framebuffers[imageIndex];
    rp.renderArea.offset = { 0, 0 };
    rp.renderArea.extent = swapchainExtent;
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(cmd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("Failed to record command buffer");
}

void Renderer::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    // Per-image sync for present waits
    renderFinishedSemaphores.resize(swapchainImages.size());
    imagesInFlight.resize(swapchainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo sem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fence{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Per-frame
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(device, &sem, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fence, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create per-frame sync objects");
        }
    }

    recreatePerImageSemaphores();
}

// Create per-image semaphores; call on first init and after swapchain recreation
void Renderer::recreatePerImageSemaphores() {
    // destroy old if any
    for (auto s : renderFinishedSemaphores)
        if (s) vkDestroySemaphore(device, s, nullptr);

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
    for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();

    if (renderPass) {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }

    for (auto view : swapchainImageViews) vkDestroyImageView(device, view, nullptr);
    swapchainImageViews.clear();
    swapchainImages.clear();

    if (swapchain) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

// Full swapchain recreation on resize or OUT_OF_DATE
void Renderer::recreateSwapchain() {
    // Handle minimized windows: wait until there is a non-zero framebuffer
    int width = 0, height = 0;
    do {
        glfwGetFramebufferSize(windowHandle, &width, &height);
        if (width == 0 || height == 0)
            glfwWaitEvents();
    } while (width == 0 || height == 0);

    vkDeviceWaitIdle(device);

    destroySwapchainObjects();

    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    // command buffers depend on framebuffer count
    createCommandBuffers();

    // per-image semaphores depend on swapchain image count
    recreatePerImageSemaphores();
}
