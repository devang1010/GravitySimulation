#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

const float PI = 3.14159265358979323846f;
const double G = 6.6743e-11;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

const char* vertexShaderSrc = R"(
    #version 330 core
    layout(location = 0) in vec2 aPos;
    uniform vec2  uOffset;
    uniform float uScale;
    uniform float uAspect;
    void main() {
        vec2 pos = aPos * uScale + uOffset;
        pos.x /= uAspect;
        gl_Position = vec4(pos, 0.0, 1.0);
    }
)";

const char* fragmentShaderSrc = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec4 uColor;
    void main() {
        FragColor = uColor;
    }
)";

const double SIM_SCALE = 1.5e11;

double orbitalVel(double mass_central, double r_ndc) {
    double r_m = r_ndc * SIM_SCALE;
    double v_m = sqrt(G * mass_central / r_m);
    return v_m / SIM_SCALE;
}

const float TILT = 0.45f;

struct Body {
    double posX, posZ;
    double velX, velZ;
    double mass;
    double density;
    double radius_ndc;
    float  r, g, b;
    bool   alive = true;

    unsigned int VAO, VBO;
    int segments = 128;

    // visual_radius_override: if > 0, ignore physics radius and use this
    double visual_radius;

    Body(double mass, double density,
         double px, double pz,
         double vx, double vz,
         float r, float g, float b,
         double visual_radius_override = -1.0)
        : mass(mass), density(density),
          posX(px), posZ(pz),
          velX(vx), velZ(vz),
          r(r), g(g), b(b),
          visual_radius(visual_radius_override)
    {
        updateRadius();
        buildMesh();
    }

    void updateRadius() {
        if (visual_radius > 0) {
            // use the override — physics mass does not affect visual size
            radius_ndc = visual_radius;
        } else {
            double radius_m = cbrt((3.0 * mass / density) / (4.0 * PI));
            radius_ndc = radius_m / SIM_SCALE;
            radius_ndc = max(radius_ndc, 0.03);
            radius_ndc = min(radius_ndc, 0.28);
        }
    }

    float screenX() const { return (float)posX; }
    float screenY() const { return (float)posZ * TILT; }

    void buildMesh() {
        vector<float> verts;
        for (int i = 0; i < segments; i++) {
            float a1 = 2.0f * PI * i       / segments;
            float a2 = 2.0f * PI * (i + 1) / segments;
            verts.push_back(0.0f); verts.push_back(0.0f);
            verts.push_back(cos(a1)); verts.push_back(sin(a1));
            verts.push_back(cos(a2)); verts.push_back(sin(a2));
        }
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER,
                     verts.size() * sizeof(float),
                     verts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }

    void draw(int offsetLoc, int colorLoc, int scaleLoc,
              int aspectLoc, float aspect) const {
        glUniform2f(offsetLoc, screenX(), screenY());
        glUniform1f(scaleLoc,  (float)radius_ndc);
        glUniform4f(colorLoc,  r, g, b, 1.0f);
        glUniform1f(aspectLoc, aspect);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, segments * 3);
    }

    void cleanup() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
};

void merge(Body& a, Body& b) {
    double totalMass = a.mass + b.mass;
    a.velX = (a.mass * a.velX + b.mass * b.velX) / totalMass;
    a.velZ = (a.mass * a.velZ + b.mass * b.velZ) / totalMass;
    a.posX = (a.mass * a.posX + b.mass * b.posX) / totalMass;
    a.posZ = (a.mass * a.posZ + b.mass * b.posZ) / totalMass;
    a.mass    = totalMass;
    a.density = max(a.density, b.density);
    float ra  = (float)(b.mass / totalMass);
    a.r = a.r * (1.0f - ra) + b.r * ra;
    a.g = a.g * (1.0f - ra) + b.g * ra;
    a.b = a.b * (1.0f - ra) + b.b * ra;
    // keep visual override on merge
    a.visual_radius = -1.0; // merged body uses physics radius
    a.updateRadius();
    b.alive = false;
}

void stepPhysics(vector<Body>& bodies, double dt) {
    int n = bodies.size();

    for (int i = 0; i < n; i++) {
        if (!bodies[i].alive) continue;
        for (int j = i + 1; j < n; j++) {
            if (!bodies[j].alive) continue;

            double dx    = (bodies[j].posX - bodies[i].posX) * SIM_SCALE;
            double dz    = (bodies[j].posZ - bodies[i].posZ) * SIM_SCALE;
            double dist2 = dx*dx + dz*dz;
            double dist  = sqrt(dist2);
            double soft  = (bodies[i].radius_ndc + bodies[j].radius_ndc)
                           * SIM_SCALE * 0.3;
            double dist2s = dist2 + soft * soft;

            double force = G * bodies[i].mass * bodies[j].mass / dist2s;
            double nx = dx / dist;
            double nz = dz / dist;
            double ai = force / bodies[i].mass / SIM_SCALE;
            double aj = force / bodies[j].mass / SIM_SCALE;

            bodies[i].velX += nx * ai * dt;
            bodies[i].velZ += nz * ai * dt;
            bodies[j].velX -= nx * aj * dt;
            bodies[j].velZ -= nz * aj * dt;
        }
    }

    for (auto& b : bodies) {
        if (!b.alive) continue;
        b.posX += b.velX * dt;
        b.posZ += b.velZ * dt;
    }

    for (int i = 0; i < n; i++) {
        if (!bodies[i].alive) continue;
        for (int j = i + 1; j < n; j++) {
            if (!bodies[j].alive) continue;
            double dx   = bodies[j].posX - bodies[i].posX;
            double dz   = bodies[j].posZ - bodies[i].posZ;
            double dist = sqrt(dx*dx + dz*dz);
            if (dist < bodies[i].radius_ndc + bodies[j].radius_ndc) {
                if (bodies[i].mass >= bodies[j].mass)
                    merge(bodies[i], bodies[j]);
                else
                    merge(bodies[j], bodies[i]);
            }
        }
    }

    bodies.erase(
        remove_if(bodies.begin(), bodies.end(),
                  [](const Body& b){ return !b.alive; }),
        bodies.end()
    );
}

unsigned int buildProgram() {
    auto compile = [](GLenum type, const char* src) {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, NULL);
        glCompileShader(s);
        return s;
    };
    unsigned int vs   = compile(GL_VERTEX_SHADER,   vertexShaderSrc);
    unsigned int fs   = compile(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

vector<Body> makeScene() {
    const double M_SUN   = 1.989e30;
    const double M_EARTH = 5.972e24;
    const double M_MOON  = 7.342e22;
    const double M_MARS  = 6.39e23;

    // -----------------------------------------------------------
    // Orbital periods:
    //   Earth: 365.25 days
    //   Mars:  687.0  days  (~1.88x Earth)
    //   Moon:  27.3   days  (around Earth)
    //
    // Kepler's 3rd law: T ∝ r^(3/2)
    // So if Earth orbital radius = R_EARTH in NDC,
    // Mars radius must satisfy:
    //   (R_MARS / R_EARTH)^(3/2) = 687/365.25 = 1.881
    //   R_MARS = R_EARTH * 1.881^(2/3)
    //          = R_EARTH * 1.524
    // This is exactly the real ratio (Earth=1AU, Mars=1.524AU)
    // -----------------------------------------------------------
    const double R_EARTH = 0.38;
    const double R_MARS  = R_EARTH * 1.524;   // = 0.579 NDC
    const double R_MOON  = 0.10;              // around earth

    // const double R_EARTH = 0.88;
    // const double R_MARS  = R_EARTH * 1.524;   // = 0.579 NDC
    // const double R_MOON  = 0.10;              // around earth

    cout << "Orbital radii (NDC):" << endl;
    cout << "  Earth: " << R_EARTH << endl;
    cout << "  Mars:  " << R_MARS  << "  (ratio=" << R_MARS/R_EARTH << ")" << endl;
    cout << "  Moon:  " << R_MOON  << " (around earth)" << endl;

    // Orbital velocity from Newton: v = sqrt(G*M/r)
    // This automatically gives correct relative periods via Kepler
    double v_earth = orbitalVel(M_SUN,   R_EARTH);
    double v_mars  = orbitalVel(M_SUN,   R_MARS);
    double v_moon  = orbitalVel(M_EARTH, R_MOON);

    // Period check — T = 2*pi*r / v
    // (in real seconds, then divide by 86400 for days)
    double T_earth = 2.0 * PI * R_EARTH * SIM_SCALE / (v_earth * SIM_SCALE) / 86400.0;
    double T_mars  = 2.0 * PI * R_MARS  * SIM_SCALE / (v_mars  * SIM_SCALE) / 86400.0;
    double T_moon  = 2.0 * PI * R_MOON  * SIM_SCALE / (v_moon  * SIM_SCALE) / 86400.0;
    cout << "\nOrbital periods (real days):" << endl;
    cout << "  Earth: " << T_earth << "  (real: 365.25)" << endl;
    cout << "  Mars:  " << T_mars  << "  (real: 687.0)"  << endl;
    cout << "  Moon:  " << T_moon  << "  (real: 27.3)"   << endl;
    cout << "  Mars/Earth ratio: " << T_mars/T_earth << "  (real: 1.88)" << endl;

    return {
        // ---  Sun  ---
        // mass      dens   posX     posZ    velX    velZ
        // visual_radius = -1 means use physics calculation
        { M_SUN,    1408,  0.0,     0.0,    0.0,    0.0,
          1.0f, 0.92f, 0.2f,
          -1.0 },        // physics radius (will clamp to 0.28)

        // --- Earth --- starts right (+X), moves into screen (+Z) = CCW
        { M_EARTH,  5515,  R_EARTH, 0.0,    0.0,    v_earth,
          0.18f, 0.5f, 1.0f,
          -1.0 },        // physics radius

        // --- Moon --- starts above earth in Z, moves left (-X) = CCW around earth
        // IMPORTANT: visual_radius = 0.06 → always visible regardless of mass
        { M_MOON,   3344,  R_EARTH, R_MOON, -v_moon, v_earth,
          0.95f, 0.95f, 0.95f,
          -1.0 },        // ← forced visual size — this is why moon was invisible before

        // --- Mars --- starts left (-X), moves away (-Z) = CCW
        { M_MARS,   3934, -R_MARS,  0.0,    0.0,    -v_mars,
          0.9f, 0.35f, 0.1f,
          -1.0 },        // physics radius
    };
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode    = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(
        mode->width, mode->height,
        "Solar System — Side-Top View", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    int winW, winH;
    glfwGetFramebufferSize(window, &winW, &winH);
    glViewport(0, 0, winW, winH);
    float aspect = (float)winW / (float)winH;

    unsigned int program  = buildProgram();
    int offsetLoc = glGetUniformLocation(program, "uOffset");
    int colorLoc  = glGetUniformLocation(program, "uColor");
    int scaleLoc  = glGetUniformLocation(program, "uScale");
    int aspectLoc = glGetUniformLocation(program, "uAspect");

    vector<Body> bodies = makeScene();

    const double TIME_SCALE = 3e6;
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        double dt  = (now - lastTime) * TIME_SCALE;
        lastTime   = now;
        if (dt > 0.05 * TIME_SCALE) dt = 0.05 * TIME_SCALE;

        stepPhysics(bodies, dt);

        glClearColor(0.02f, 0.02f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        for (auto& b : bodies)
            b.draw(offsetLoc, colorLoc, scaleLoc, aspectLoc, aspect);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            for (auto& b : bodies) b.cleanup();
            bodies = makeScene();
        }
    }

    for (auto& b : bodies) b.cleanup();
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}