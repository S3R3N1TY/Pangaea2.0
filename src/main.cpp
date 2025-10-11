#include <GLFW/glfw3.h>
#include "Renderer.hpp"
#include <iostream>

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    (void)width; (void)height;
    // Store a pointer to Renderer in the window user pointer
    auto renderer = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (renderer) renderer->setFramebufferResized(true);
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Pangaea 2.0", nullptr, nullptr);

    Renderer renderer;
    glfwSetWindowUserPointer(window, &renderer);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    try {
        renderer.init(window);
    }
    catch (const std::exception& e) {
        std::cerr << "Init error: " << e.what() << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        renderer.drawFrame();
    }

    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Exit cleanly.\n";
    return 0;
}
