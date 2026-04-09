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

struct OrbitRing {
    unsigned int VAO, VBO;
    int pointCount;
    float r, g, b, a;

    OrbitRing(double radius_ndc, float r, float g, float b,
              float alpha = 0.22f, int points = 256)
        : r(r), g(g), b(b), a(alpha), pointCount(points)
    {
        vector<float> verts;
        for (int i = 0; i < points; i++) {
            float angle = 2.0f * PI * i / points;
            verts.push_back((float)radius_ndc * cos(angle));
            verts.push_back((float)radius_ndc * sin(angle) * TILT);
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
        glUniform2f(offsetLoc, 0.0f, 0.0f);
        glUniform1f(scaleLoc,  1.0f);
        glUniform4f(colorLoc,  r, g, b, a);
        glUniform1f(aspectLoc, aspect);
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINE_LOOP, 0, pointCount);
    }

    void cleanup() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
};

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
            radius_ndc = visual_radius;
        } else {
            double radius_m = cbrt((3.0 * mass / density) / (4.0 * PI));
            radius_ndc = radius_m / SIM_SCALE;
            radius_ndc = max(radius_ndc, 0.012);
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
            verts.push_back(0.0f);    verts.push_back(0.0f);
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

void bounce(Body& a, Body& b, double restitution = 0.8) {
    double dx   = b.posX - a.posX;
    double dz   = b.posZ - a.posZ;
    double dist = sqrt(dx*dx + dz*dz);
    if (dist == 0) return;
    double nx = dx / dist;
    double nz = dz / dist;
    double dvx = a.velX - b.velX;
    double dvz = a.velZ - b.velZ;
    double dvn = dvx * nx + dvz * nz;
    if (dvn < 0) return;
    double impulse   = (1.0 + restitution) * dvn / (1.0/a.mass + 1.0/b.mass);
    a.velX -= impulse / a.mass * nx;
    a.velZ -= impulse / a.mass * nz;
    b.velX += impulse / b.mass * nx;
    b.velZ += impulse / b.mass * nz;
    double overlap   = (a.radius_ndc + b.radius_ndc) - dist;
    double totalMass = a.mass + b.mass;
    a.posX -= nx * overlap * (b.mass / totalMass);
    a.posZ -= nz * overlap * (b.mass / totalMass);
    b.posX += nx * overlap * (a.mass / totalMass);
    b.posZ += nz * overlap * (a.mass / totalMass);
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
            double force  = G * bodies[i].mass * bodies[j].mass / dist2s;
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
            if (dist < bodies[i].radius_ndc + bodies[j].radius_ndc)
                bounce(bodies[i], bodies[j], 0.6);
        }
    }
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

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
    GLFWwindow* window = glfwCreateWindow(
        mode->width, mode->height, "Solar System", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    int winW, winH;
    glfwGetFramebufferSize(window, &winW, &winH);
    glViewport(0, 0, winW, winH);
    float aspect = (float)winW / (float)winH;

    unsigned int program  = buildProgram();
    int offsetLoc = glGetUniformLocation(program, "uOffset");
    int colorLoc  = glGetUniformLocation(program, "uColor");
    int scaleLoc  = glGetUniformLocation(program, "uScale");
    int aspectLoc = glGetUniformLocation(program, "uAspect");

    // -------------------------------------------------------
    // Real AU ratios (Earth = 1 AU):
    //   Mercury = 0.387 AU
    //   Venus   = 0.723 AU
    //   Earth   = 1.000 AU
    //   Mars    = 1.524 AU
    //
    // We set R_EARTH as our base NDC radius and scale others
    // from it using real AU ratios.
    // -------------------------------------------------------
    const double M_SUN     = 1.989e30;
    const double M_MERCURY = 3.285e23;
    const double M_VENUS   = 4.867e24;
    const double M_EARTH   = 5.972e24;
    const double M_MARS    = 6.39e23;

    const double R_EARTH   = 0.42;               // base NDC radius
    const double R_MERCURY = R_EARTH * 0.387;    // 0.163 NDC
    const double R_VENUS   = R_EARTH * 0.723;    // 0.303 NDC
    const double R_MARS    = R_EARTH * 1.524;    // 0.640 NDC

    double v_mercury = orbitalVel(M_SUN, R_MERCURY);
    double v_venus   = orbitalVel(M_SUN, R_VENUS);
    double v_earth   = orbitalVel(M_SUN, R_EARTH);
    double v_mars    = orbitalVel(M_SUN, R_MARS);

    // Period check: T = 2*pi*r / v (in days)
    auto periodDays = [&](double r, double v) {
        return 2.0 * PI * r * SIM_SCALE / (v * SIM_SCALE) / 86400.0;
    };
    cout << "Orbital periods:" << endl;
    cout << "  Mercury: " << periodDays(R_MERCURY, v_mercury) << " d  (real: 88)"   << endl;
    cout << "  Venus:   " << periodDays(R_VENUS,   v_venus)   << " d  (real: 224.7)"<< endl;
    cout << "  Earth:   " << periodDays(R_EARTH,   v_earth)   << " d  (real: 365.25)"<< endl;
    cout << "  Mars:    " << periodDays(R_MARS,    v_mars)    << " d  (real: 687)"   << endl;

    // -------------------------------------------------------
    // Each planet starts at a different angle so they are
    // spread around the sun at launch — looks nicer than
    // all bunched up on the same side.
    //
    // Position on orbit at angle theta:
    //   posX = R * cos(theta)
    //   posZ = R * sin(theta)
    // Velocity perpendicular CCW at angle theta:
    //   velX = -v * sin(theta)
    //   velZ =  v * cos(theta)
    // -------------------------------------------------------
    auto startPos = [](double R, double theta, double v,
                       double& px, double& pz,
                       double& vx, double& vz) {
        px = R * cos(theta);
        pz = R * sin(theta);
        vx = -v * sin(theta);  // perpendicular CCW
        vz =  v * cos(theta);
    };

    double px, pz, vx, vz;

    // orbit rings — drawn behind everything
    vector<OrbitRing> rings = {
        OrbitRing(R_MERCURY, 0.7f,  0.7f,  0.7f,  0.2f), // grey
        OrbitRing(R_VENUS,   0.9f,  0.7f,  0.3f,  0.2f), // warm yellow
        OrbitRing(R_EARTH,   0.3f,  0.5f,  1.0f,  0.2f), // blue
        OrbitRing(R_MARS,    0.9f,  0.4f,  0.2f,  0.2f), // orange
    };

    // build bodies
    vector<Body> bodies;

    // Sun
    bodies.push_back({ M_SUN, 1408,
                       0.0, 0.0, 0.0, 0.0,
                       1.0f, 0.92f, 0.2f, 0.035 });

    // Mercury — starts at 0° (right)
    startPos(R_MERCURY, 0.0, v_mercury, px, pz, vx, vz);
    bodies.push_back({ M_MERCURY, 5427,
                       px, pz, vx, vz,
                       0.72f, 0.70f, 0.68f,  // grey
                       0.020 });              // tiny but visible

    // Venus — starts at 60°
    startPos(R_VENUS, PI * 0.33, v_venus, px, pz, vx, vz);
    bodies.push_back({ M_VENUS, 5243,
                       px, pz, vx, vz,
                       0.90f, 0.75f, 0.35f,  // pale orange-yellow
                       0.023 });

    // Earth — starts at 150°
    startPos(R_EARTH, PI * 0.83, v_earth, px, pz, vx, vz);
    bodies.push_back({ M_EARTH, 5515,
                       px, pz, vx, vz,
                       0.18f, 0.50f, 1.00f,  // blue
                       0.025 });

    // Mars — starts at 240°
    startPos(R_MARS, PI * 1.33, v_mars, px, pz, vx, vz);
    bodies.push_back({ M_MARS, 3934,
                       px, pz, vx, vz,
                       0.90f, 0.35f, 0.10f,  // red-orange
                       0.035 });

    // lambda to rebuild scene
    auto resetBodies = [&]() -> vector<Body> {
        vector<Body> b;
        b.push_back({ M_SUN, 1408, 0.0, 0.0, 0.0, 0.0,
                      1.0f, 0.92f, 0.2f, -1.0 });

        startPos(R_MERCURY, 0.0, v_mercury, px, pz, vx, vz);
        b.push_back({ M_MERCURY, 5427, px, pz, vx, vz,
                      0.72f, 0.70f, 0.68f, 0.018 });

        startPos(R_VENUS, PI * 0.33, v_venus, px, pz, vx, vz);
        b.push_back({ M_VENUS, 5243, px, pz, vx, vz,
                      0.90f, 0.75f, 0.35f, -1.0 });

        startPos(R_EARTH, PI * 0.83, v_earth, px, pz, vx, vz);
        b.push_back({ M_EARTH, 5515, px, pz, vx, vz,
                      0.18f, 0.50f, 1.00f, -1.0 });

        startPos(R_MARS, PI * 1.33, v_mars, px, pz, vx, vz);
        b.push_back({ M_MARS, 3934, px, pz, vx, vz,
                      0.90f, 0.35f, 0.10f, -1.0 });
        return b;
    };

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

        // rings first — behind planets
        for (auto& ring : rings)
            ring.draw(offsetLoc, colorLoc, scaleLoc, aspectLoc, aspect);

        // planets on top
        for (auto& b : bodies)
            b.draw(offsetLoc, colorLoc, scaleLoc, aspectLoc, aspect);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            for (auto& b : bodies) b.cleanup();
            bodies = resetBodies();
        }
    }

    for (auto& b  : bodies) b.cleanup();
    for (auto& rg : rings)  rg.cleanup();
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}