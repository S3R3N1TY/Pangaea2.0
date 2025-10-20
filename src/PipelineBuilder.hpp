#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <cstdint>

// Compatibility shim thingy (handles older headers without VK_PIPELINE_CREATE_RENDERING_BIT)
#ifndef VK_PIPELINE_CREATE_RENDERING_BIT
# ifdef VK_PIPELINE_CREATE_RENDERING_BIT_KHR
#  define VK_PIPELINE_CREATE_RENDERING_BIT VK_PIPELINE_CREATE_RENDERING_BIT_KHR
# else
#  define VK_PIPELINE_CREATE_RENDERING_BIT 0
# endif
#endif


class PipelineBuilder {
public:
    // ----- Lifecycle helpers -----
    PipelineBuilder& reset();                      // clear all state
    PipelineBuilder& clearStages() { stages.clear(); return *this; }

    // ----- Shader Stages -----
    PipelineBuilder& addStage(VkShaderStageFlagBits stage, VkShaderModule module, const char* entry = "main");

    // ----- Vertex Input & Assembly -----
    PipelineBuilder& setVertexInput(const VkVertexInputBindingDescription* bindings, uint32_t bindingCount,
        const VkVertexInputAttributeDescription* attrs, uint32_t attrCount);
    PipelineBuilder& setInputAssembly(VkPrimitiveTopology topo, VkBool32 primitiveRestart = VK_FALSE);

    // ----- Fixed Function States -----
    PipelineBuilder& setViewport(float x, float y, float w, float h,
        float minDepth = 0.0f, float maxDepth = 1.0f);
    PipelineBuilder& setScissor(int32_t x, int32_t y, uint32_t w, uint32_t h);
    PipelineBuilder& setRasterization(const VkPipelineRasterizationStateCreateInfo& r);
    PipelineBuilder& setMultisample(const VkPipelineMultisampleStateCreateInfo& m);
    PipelineBuilder& setDepthStencil(const VkPipelineDepthStencilStateCreateInfo& d);
    PipelineBuilder& enableDepth(bool on) { useDepth = on; return *this; } // explicit toggle
    PipelineBuilder& setColorBlendAttachments(const std::vector<VkPipelineColorBlendAttachmentState>& atts);

    // No blending by default
    PipelineBuilder& setColorWriteMask(
        VkColorComponentFlags mask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        VkBool32 enableBlend = VK_FALSE);

    PipelineBuilder& setDynamicStates(const std::vector<VkDynamicState>& states);

    // ----- Layout / Rendering Formats -----
    PipelineBuilder& setLayout(VkPipelineLayout layout_);
    PipelineBuilder& setRenderingFormats(const std::vector<VkFormat>& colorFormats,
        VkFormat depthFormat = VK_FORMAT_UNDEFINED);
    PipelineBuilder& setPipelineCache(VkPipelineCache cache_);

    // ----- Final Build -----
    VkPipeline build(VkDevice device) const;

private:
    // Shader stages
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    // Vertex input & assembly
    VkPipelineVertexInputStateCreateInfo   vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };

    // Fixed-function bits
    VkViewport viewport{};
    VkRect2D   scissor{ {0,0}, {0,0} };
    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo   msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo  depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    bool useDepth = false;

    std::vector<VkPipelineColorBlendAttachmentState> colorAttachments;
    std::vector<VkDynamicState> dynamicStates;

    // Layout / rendering info
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipelineCache  cache = VK_NULL_HANDLE;

    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    std::vector<VkFormat> colorFormatsOwned;
};

// ---------------- Inline definitions ----------------

inline PipelineBuilder& PipelineBuilder::reset() {
    stages.clear();

    vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };

    viewport = {};
    scissor = { {0,0}, {0,0} };

    raster = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    msaa = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    depth = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    useDepth = false;

    colorAttachments.clear();
    dynamicStates.clear();

    layout = VK_NULL_HANDLE;
    cache = VK_NULL_HANDLE;

    rendering = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    colorFormatsOwned.clear();
    return *this;
}

inline PipelineBuilder& PipelineBuilder::addStage(VkShaderStageFlagBits stage, VkShaderModule module, const char* entry) {
    VkPipelineShaderStageCreateInfo s{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    s.stage = stage; s.module = module; s.pName = entry;
    stages.push_back(s);
    return *this;
}

inline PipelineBuilder& PipelineBuilder::setVertexInput(const VkVertexInputBindingDescription* bindings, uint32_t bindingCount,
    const VkVertexInputAttributeDescription* attrs, uint32_t attrCount) {
    vertexInput.vertexBindingDescriptionCount = bindingCount;
    vertexInput.pVertexBindingDescriptions = bindings;
    vertexInput.vertexAttributeDescriptionCount = attrCount;
    vertexInput.pVertexAttributeDescriptions = attrs;
    return *this;
}

inline PipelineBuilder& PipelineBuilder::setInputAssembly(VkPrimitiveTopology topo, VkBool32 primitiveRestart) {
    inputAssembly.topology = topo;
    inputAssembly.primitiveRestartEnable = primitiveRestart;
    return *this;
}

inline PipelineBuilder& PipelineBuilder::setViewport(float x, float y, float w, float h, float minDepth, float maxDepth) {
    viewport = { x, y, w, h, minDepth, maxDepth };
    return *this;
}

inline PipelineBuilder& PipelineBuilder::setScissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
    scissor = { {x, y}, {w, h} };
    return *this;
}

inline PipelineBuilder& PipelineBuilder::setRasterization(const VkPipelineRasterizationStateCreateInfo& r) { raster = r; return *this; }
inline PipelineBuilder& PipelineBuilder::setMultisample(const VkPipelineMultisampleStateCreateInfo& m) { msaa = m; return *this; }
inline PipelineBuilder& PipelineBuilder::setDepthStencil(const VkPipelineDepthStencilStateCreateInfo& d) { depth = d; useDepth = true; return *this; }
inline PipelineBuilder& PipelineBuilder::setColorBlendAttachments(const std::vector<VkPipelineColorBlendAttachmentState>& atts) { colorAttachments = atts; return *this; }

inline PipelineBuilder& PipelineBuilder::setColorWriteMask(VkColorComponentFlags mask, VkBool32 enableBlend) {
    VkPipelineColorBlendAttachmentState a{};
    a.colorWriteMask = mask;
    a.blendEnable = enableBlend;
    colorAttachments = { a };
    return *this;
}

inline PipelineBuilder& PipelineBuilder::setDynamicStates(const std::vector<VkDynamicState>& states) { dynamicStates = states; return *this; }

inline PipelineBuilder& PipelineBuilder::setLayout(VkPipelineLayout layout_) { layout = layout_; return *this; }
inline PipelineBuilder& PipelineBuilder::setRenderingFormats(const std::vector<VkFormat>& colorFormats, VkFormat depthFormat) {
    rendering.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
    colorFormatsOwned = colorFormats;
    rendering.pColorAttachmentFormats = colorFormatsOwned.empty() ? nullptr : colorFormatsOwned.data();
    rendering.depthAttachmentFormat = depthFormat;
    rendering.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    return *this;
}
inline PipelineBuilder& PipelineBuilder::setPipelineCache(VkPipelineCache cache_) { cache = cache_; return *this; }

inline VkPipeline PipelineBuilder::build(VkDevice device) const {
    if (stages.empty())   throw std::runtime_error("PipelineBuilder: no shader stages added (addStage())");
    if (layout == VK_NULL_HANDLE) throw std::runtime_error("PipelineBuilder: missing layout (setLayout())");
    if (rendering.colorAttachmentCount == 0 && rendering.depthAttachmentFormat == VK_FORMAT_UNDEFINED)
        throw std::runtime_error("PipelineBuilder: no color or depth formats set (setRenderingFormats())");

    // Fixed-function groups
    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1; viewportState.pScissors = &scissor;

    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = static_cast<uint32_t>(colorAttachments.size());
    colorBlend.pAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data();

    VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    if (!dynamicStates.empty()) {
        dyn.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dyn.pDynamicStates = dynamicStates.data();
    }

    VkGraphicsPipelineCreateInfo info{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    info.pNext = &rendering;
    info.flags = VK_PIPELINE_CREATE_RENDERING_BIT;
    info.stageCount = static_cast<uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &msaa;
    info.pDepthStencilState = useDepth ? &depth : nullptr;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = dynamicStates.empty() ? nullptr : &dyn;
    info.layout = layout;
    info.renderPass = VK_NULL_HANDLE;
    info.subpass = 0;

    VkPipeline pipe{};
    if (vkCreateGraphicsPipelines(device, cache, 1, &info, nullptr, &pipe) != VK_SUCCESS)
        throw std::runtime_error("PipelineBuilder: vkCreateGraphicsPipelines failed");
    return pipe;
}
