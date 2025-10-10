#include <GLFW/glfw3.h>
#include "Renderer.hpp"
#include <iostream>

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Pangaea 2.0", nullptr, nullptr);

    Renderer renderer;
    renderer.init();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        renderer.drawFrame();
    }

    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Exit cleanly.\n";
}
