#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
using namespace std;

const float PI = 3.14159265358979323846f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

const char* vertexShaderSrc = R"(
    #version 330 core
    layout(location = 0) in vec2 aPos;
    uniform vec2 uOffset;
    void main() {
        gl_Position = vec4(aPos + uOffset, 0.0, 1.0);
    }
)";

const char* fragmentShaderSrc = R"(
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0); // white
    }
)";

vector<float> buildCircle(float radius, int segments) {
    vector<float> verts;
    for (int i = 0; i < segments; i++) {
        float angle1 = 2.0f * PI * i       / segments;
        float angle2 = 2.0f * PI * (i + 1) / segments;
        verts.push_back(0.0f);
        verts.push_back(0.0f);
        verts.push_back(radius * cos(angle1));
        verts.push_back(radius * sin(angle1));
        verts.push_back(radius * cos(angle2));
        verts.push_back(radius * sin(angle2));
    }
    return verts;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Bouncing Circle", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // --- Circle mesh ---
    const float RADIUS   = 0.08f;
    const int   SEGMENTS = 64;
    vector<float> circleVerts = buildCircle(RADIUS, SEGMENTS);

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 circleVerts.size() * sizeof(float),
                 circleVerts.data(),
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // --- Shaders ---
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSrc, NULL);
    glCompileShader(vs);

    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSrc, NULL);
    glCompileShader(fs);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    int offsetLoc = glGetUniformLocation(program, "uOffset");

    // --- Physics setup ---
    const float SCREEN_HEIGHT_METERS = 5.0f;
    const float SCALE = 2.0f / SCREEN_HEIGHT_METERS;   // NDC per meter

    float posY_m  =  SCREEN_HEIGHT_METERS / 2.0f;      // start at top
    float posX    =  0.0f;
    float velY_m  =  0.0f;
    float gravity = -9.8f;                              // real m/s²
    float bounce  =  0.6f;                              // 60% energy on bounce
    float floor_m = -(SCREEN_HEIGHT_METERS / 2.0f) + (RADIUS / SCALE);

    float lastTime = (float)glfwGetTime();

    // --- Render loop ---
    while (!glfwWindowShouldClose(window)) {
        float now = (float)glfwGetTime();
        float dt  = now - lastTime;
        lastTime  = now;

        // clamp dt so a window drag doesn't cause a huge jump
        if (dt > 0.05f) dt = 0.05f;

        // gravity
        velY_m += gravity * dt;
        posY_m += velY_m  * dt;

        // bounce off floor
        if (posY_m <= floor_m) {
            posY_m = floor_m;
            velY_m = -velY_m * bounce;
            if (fabs(velY_m) < 0.05f) velY_m = 0.0f;
        }

        // convert meters → NDC
        float posY_ndc = posY_m * SCALE;

        // draw
        glClearColor(0.08f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glUniform2f(offsetLoc, posX, posY_ndc);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, SEGMENTS * 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --- Cleanup ---
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}