#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <chrono>

struct GLFWwindow;

class Renderer {
public:
    void init(GLFWwindow* window);
    void cleanup();
    void drawFrame();

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

    // Depth
    VkImage        depthImage{};
    VkDeviceMemory depthImageMemory{};
    VkImageView    depthImageView{};
    VkFormat       depthFormat{};

    // Render pass, pipeline & framebuffers
    VkRenderPass renderPass{};
    VkDescriptorSetLayout descriptorSetLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline graphicsPipeline{};
    std::vector<VkFramebuffer> framebuffers;

    // Buffers: vertex/index
    VkBuffer vertexBuffer{};
    VkDeviceMemory vertexBufferMemory{};
    VkBuffer indexBuffer{};
    VkDeviceMemory indexBufferMemory{};

    // Uniforms: per swapchain image
    struct UniformBufferObject {
        float mvp[16]; // raw 4x4 column-major
    };
    std::vector<VkBuffer>       uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*>          uniformBuffersMapped;

    VkDescriptorPool descriptorPool{};
    std::vector<VkDescriptorSet> descriptorSets;

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

    // Time
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

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
    void createDescriptorSetLayout();
    void createDepthResources();
    void createGraphicsPipeline();
    void createFramebuffers();

    // Resources
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void updateUniformBuffer(uint32_t imageIndex);
    void createDescriptorPool();
    void createDescriptorSets();

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

    // Images & buffers
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);
    void createImage(uint32_t w, uint32_t h, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags props,
        VkImage& image, VkDeviceMemory& memory);
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect);
    VkFormat findDepthFormat();
    bool hasStencilComponent(VkFormat format);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
        VkBuffer& buffer, VkDeviceMemory& memory);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);

    // Resize handling
    void recreateSwapchain();
    void destroySwapchainObjects();
};
