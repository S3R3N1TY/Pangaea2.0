#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <optional>
#include <chrono>
#include <array>
#include <stdexcept>

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
    VmaAllocator allocator = VK_NULL_HANDLE;
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
    VmaAllocation  depthAlloc{};
    VkImageView    depthImageView{};
    VkFormat       depthFormat{};

    // Render pass, pipeline & framebuffers
    VkDescriptorSetLayout descriptorSetLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline graphicsPipeline{};

    // Buffers: vertex/index
    VkBuffer       vertexBuffer{};
    VmaAllocation  vertexAlloc{};
    VkBuffer       indexBuffer{};
    VmaAllocation  indexAlloc{};

    // Uniforms: per swapchain image
    struct UniformBufferObject { float vp[16]; };
    std::vector<VkBuffer>      uniformBuffers;
    std::vector<VmaAllocation> uniformAllocs;
    std::vector<void*>         uniformMapped;

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
    void createAllocator(); // VMA

    // Swapchain pipeline
    void createSwapchain();
    void createImageViews();
    void createDescriptorSetLayout();
    void createDepthResources();
    void createGraphicsPipeline();

    // Resources
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void updateUniformBuffer(uint32_t imageIndex);
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

    // ---------------- Staging uploader ----------------
    struct StagingUploader {
        // External deps
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool cmdPool = VK_NULL_HANDLE;

        // Reusable staging buffer
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAlloc = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize capacity = 0;

        // Per-upload fence
        VkFence copyFence = VK_NULL_HANDLE;

        void init(VmaAllocator alloc, VkDevice dev, VkQueue q, VkCommandPool pool, VkDeviceSize initialCapacity);
        void destroy();

        // Ensures capacity >= requiredBytes, reallocating if needed which keeps mapping persistent
        void ensureCapacity(VkDeviceSize requiredBytes);

        // Blocking upload: memcpy to staging, flush, record copy, submit, wait on fence
        void upload(const void* src, VkDeviceSize sizeBytes, VkBuffer dst, VkDeviceSize dstOffset);

    private:
        VkCommandBuffer allocateOneShotCmd() const;
        void submitAndWait(VkCommandBuffer cmd);
    };

    StagingUploader uploader;

    // Device-local buffer creation helper (no mapping)
    void createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkBuffer& buffer, VmaAllocation& alloc);

    // ---------------- Descriptor arena (auto-growing pools) ----------------
    struct DescriptorArena {
        VkDevice device = VK_NULL_HANDLE;
        uint32_t maxSetsPerPool = 256; // grow in chunks

        std::vector<VkDescriptorPool> pools;

        void init(VkDevice dev) { device = dev; }
        void destroy() {
            for (auto p : pools) if (p) vkDestroyDescriptorPool(device, p, nullptr);
            pools.clear();
            device = VK_NULL_HANDLE;
        }
        // Reset all pools so their sets become invalid, ready for re-allocation.
        // Use this on swapchain rebuild, or at frame-begin if you switch to per-frame sets.
        void reset() {
            for (auto p : pools) if (p) vkResetDescriptorPool(device, p, 0);
        }

        VkDescriptorSet allocate(VkDescriptorSetLayout layout) {
            if (pools.empty()) pools.push_back(createPool());

            VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ai.descriptorPool = pools.back();
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;

            VkDescriptorSet set{};
            VkResult r = vkAllocateDescriptorSets(device, &ai, &set);
            if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
                pools.push_back(createPool());
                ai.descriptorPool = pools.back();
                VK_CHECK(vkAllocateDescriptorSets(device, &ai, &set));
            }
            else {
                VK_CHECK(r);
            }
            return set;
        }

    private:
        // Tiny helper so we can write VK_CHECK above without littering the class
        static void VK_CHECK(VkResult res) {
            if (res != VK_SUCCESS) throw std::runtime_error("DescriptorArena: VkResult != VK_SUCCESS");
        }

        VkDescriptorPool createPool() {
            // Generous defaults so you don’t babysit counts. Tune later if needed.
            std::array<VkDescriptorPoolSize, 5> sizes{ {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         maxSetsPerPool },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxSetsPerPool },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         maxSetsPerPool / 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   maxSetsPerPool / 4 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          maxSetsPerPool / 4 },
            } };

            VkDescriptorPoolCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; // nice to have
            info.maxSets = maxSetsPerPool * 2; // headroom
            info.poolSizeCount = static_cast<uint32_t>(sizes.size());
            info.pPoolSizes = sizes.data();

            VkDescriptorPool pool{};
            if (vkCreateDescriptorPool(device, &info, nullptr, &pool) != VK_SUCCESS) {
                throw std::runtime_error("DescriptorArena: failed to create descriptor pool");
            }
            return pool;
        }
    };

    DescriptorArena descriptorArena;


    // Images & buffers
    VkFormat findDepthFormat();
    bool hasStencilComponent(VkFormat format);

    // VMA versions
    void createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
        VkImage& image, VmaAllocation& alloc);
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkBuffer& buffer, VmaAllocation& alloc, void** mapped = nullptr);

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);

    // Resize handling
    void recreateSwapchain();
    void destroySwapchainObjects();
};
