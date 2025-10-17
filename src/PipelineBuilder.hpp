#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <cstdint>

// --------- Dynamic rendering compatibility shim (same spirit as your Renderer.cpp) ---------
#ifndef VK_PIPELINE_CREATE_RENDERING_BIT
# ifdef VK_PIPELINE_CREATE_RENDERING_BIT_KHR
#  define VK_PIPELINE_CREATE_RENDERING_BIT VK_PIPELINE_CREATE_RENDERING_BIT_KHR
# else
#  define VK_PIPELINE_CREATE_RENDERING_BIT 0
# endif
#endif

class PipelineBuilder {
public:
    // Shader stages
    PipelineBuilder& clearStages() { stages.clear(); return *this; }
    PipelineBuilder& addStage(VkShaderStageFlagBits stage, VkShaderModule module, const char* entry = "main") {
        VkPipelineShaderStageCreateInfo s{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        s.stage = stage; s.module = module; s.pName = entry;
        stages.push_back(s);
        return *this;
    }

    // Vertex input
    PipelineBuilder& setVertexInput(const VkVertexInputBindingDescription* bindings, uint32_t bindingCount,
        const VkVertexInputAttributeDescription* attrs, uint32_t attrCount) {
        vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInput.vertexBindingDescriptionCount = bindingCount;
        vertexInput.pVertexBindingDescriptions = bindings;
        vertexInput.vertexAttributeDescriptionCount = attrCount;
        vertexInput.pVertexAttributeDescriptions = attrs;
        return *this;
    }

    // IA
    PipelineBuilder& setInputAssembly(VkPrimitiveTopology topo, VkBool32 primitiveRestart = VK_FALSE) {
        inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = topo;
        inputAssembly.primitiveRestartEnable = primitiveRestart;
        return *this;
    }

    // Viewport/scissor (fixed for now; enable dynamic states later if you like suffering less)
    PipelineBuilder& setViewport(float x, float y, float w, float h, float minDepth = 0.0f, float maxDepth = 1.0f) {
        viewport = {};
        viewport.x = x; viewport.y = y; viewport.width = w; viewport.height = h;
        viewport.minDepth = minDepth; viewport.maxDepth = maxDepth;
        return *this;
    }
    PipelineBuilder& setScissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
        scissor = { {x, y}, {w, h} };
        return *this;
    }

    // Raster, MSAA, depth
    PipelineBuilder& setRasterization(const VkPipelineRasterizationStateCreateInfo& r) { raster = r; return *this; }
    PipelineBuilder& setMultisample(const VkPipelineMultisampleStateCreateInfo& m) { msaa = m; return *this; }
    PipelineBuilder& setDepthStencil(const VkPipelineDepthStencilStateCreateInfo& d) { depth = d; useDepth = true; return *this; }

    // Color blend
    PipelineBuilder& setColorBlendAttachments(const std::vector<VkPipelineColorBlendAttachmentState>& atts) {
        colorAttachments = atts;
        return *this;
    }

    // Layout + formats (dynamic rendering)
    PipelineBuilder& setLayout(VkPipelineLayout layout_) { layout = layout_; return *this; }
    PipelineBuilder& setRenderingFormats(const std::vector<VkFormat>& colorFormats, VkFormat depthFormat = VK_FORMAT_UNDEFINED) {
        rendering = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        rendering.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
        colorFormatsOwned = colorFormats; // keep storage alive
        rendering.pColorAttachmentFormats = colorFormatsOwned.empty() ? nullptr : colorFormatsOwned.data();
        rendering.depthAttachmentFormat = depthFormat;
        rendering.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
        return *this;
    }

    // Bake
    VkPipeline build(VkDevice device) {
        if (stages.empty()) throw std::runtime_error("PipelineBuilder: no shader stages");
        if (layout == VK_NULL_HANDLE) throw std::runtime_error("PipelineBuilder: pipeline layout not set");
        if (rendering.colorAttachmentCount == 0 && rendering.depthAttachmentFormat == VK_FORMAT_UNDEFINED)
            throw std::runtime_error("PipelineBuilder: no rendering formats set");

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;  viewportState.pScissors = &scissor;

        VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlend.attachmentCount = static_cast<uint32_t>(colorAttachments.size());
        colorBlend.pAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data();

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
        info.layout = layout;
        info.renderPass = VK_NULL_HANDLE;
        info.subpass = 0;

        VkPipeline pipe{};
        if (vkCreateGraphicsPipelines(device, cache, 1, &info, nullptr, &pipe) != VK_SUCCESS) {
            throw std::runtime_error("PipelineBuilder: vkCreateGraphicsPipelines failed");
        }
        return pipe;
    }

private:
    // Owned storage for create infos
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    VkPipelineVertexInputStateCreateInfo   vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkViewport                             viewport{};
    VkRect2D                               scissor{ {0,0}, {0,0} };

    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo   msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo  depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    bool                                   useDepth = false;

    std::vector<VkPipelineColorBlendAttachmentState> colorAttachments;

    VkPipelineLayout layout = VK_NULL_HANDLE;

    // dynamic rendering formats
    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    std::vector<VkFormat> colorFormatsOwned;

public:
	PipelineBuilder& setPipelineCache(VkPipelineCache cache_) { cache = cache_; return *this; }

private:
    VkPipelineCache cache = VK_NULL_HANDLE;
};

