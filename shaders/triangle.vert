#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;

// Per-frame UBO: view-projection
layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 vp;
} ubo;

// Per-object push constant: model matrix
layout(push_constant) uniform PushConst {
    mat4 model;
} pc;

void main() {
    vColor = inColor;
    gl_Position = ubo.vp * pc.model * vec4(inPos, 1.0);
}
