#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>

struct GLFWwindow;

class Renderer {
public:
    void init(GLFWwindow* window);
    void cleanup();
    void drawFrame();

    // Call from the GLFW resize callback
    void setFramebufferResized(bool v) { framebufferResized = v; }

private:
    // Core
    VkInstance instance{};
    VkDebugUtilsMessengerEXT debugMessenger{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device{};
    VkQueue graphicsQueue{};
    VkQueue presentQueue{};
    GLFWwindow* windowHandle = nullptr;

    // Swapchain
    VkSwapchainKHR swapchain{};
    VkFormat swapchainImageFormat{};
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    // Render pass, pipeline & framebuffers
    VkRenderPass renderPass{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline graphicsPipeline{};
    std::vector<VkFramebuffer> framebuffers;

    // Commands
    VkCommandPool commandPool{};
    std::vector<VkCommandBuffer> commandBuffers;

    // Sync
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<VkSemaphore> imageAvailableSemaphores;  // per-frame
    std::vector<VkSemaphore> renderFinishedSemaphores;  // per-swapchain-image
    std::vector<VkFence>     inFlightFences;            // per-frame
    std::vector<VkFence>     imagesInFlight;            // per-swapchain-image
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

private:
    // Setup
    void createInstance();
    void setupDebugMessenger();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();

    // Swapchain pipeline
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();

    // Commands
    void createCommandPool();
    void createCommandBuffers();

    // Sync
    void createSyncObjects();
    void recreatePerImageSemaphores();

    // Helpers
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
    };
    struct SwapSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };
    QueueFamilyIndices   findQueueFamilies(VkPhysicalDevice dev);
    bool                 isDeviceSuitable(VkPhysicalDevice dev);
    SwapSupportDetails   querySwapSupport(VkPhysicalDevice dev);
    VkSurfaceFormatKHR   chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&);
    VkPresentModeKHR     chooseSwapPresentMode(const std::vector<VkPresentModeKHR>&);
    VkExtent2D           chooseSwapExtent(const VkSurfaceCapabilitiesKHR&);
    void                 recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    VkShaderModule       createShaderModule(const std::vector<char>& code);

    // Resize handling
    void recreateSwapchain();
    void destroySwapchainObjects();
};
