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
    bool   alive     = true;
    bool   draggable = false; // true = player controlled, gravity still applied

    unsigned int VAO, VBO;
    int segments = 128;
    double visual_radius;

    Body(double mass, double density,
         double px, double pz,
         double vx, double vz,
         float r, float g, float b,
         double visual_radius_override = -1.0,
         bool draggable = false)
        : mass(mass), density(density),
          posX(px), posZ(pz),
          velX(vx), velZ(vz),
          r(r), g(g), b(b),
          visual_radius(visual_radius_override),
          draggable(draggable)
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

    // draw with a pulsing white outline to show it's selected/draggable
    void drawSelected(int offsetLoc, int colorLoc, int scaleLoc,
                      int aspectLoc, float aspect, float time) const {
        // draw the body itself
        draw(offsetLoc, colorLoc, scaleLoc, aspectLoc, aspect);

        // draw a slightly larger ring around it to highlight
        float pulse = 1.05f + 0.03f * sin(time * 4.0f);
        glUniform2f(offsetLoc, screenX(), screenY());
        glUniform1f(scaleLoc,  (float)radius_ndc * pulse);
        glUniform4f(colorLoc,  1.0f, 1.0f, 1.0f, 0.5f);
        glUniform1f(aspectLoc, aspect);
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINE_LOOP, 0, segments);
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

    // use effective mass: draggable body acts as if infinite mass (immovable)
    double invMassA = a.draggable ? 0.0 : 1.0 / a.mass;
    double invMassB = b.draggable ? 0.0 : 1.0 / b.mass;
    double invMassSum = invMassA + invMassB;
    if (invMassSum == 0.0) return; // both draggable, skip

    double impulse = (1.0 + restitution) * dvn / invMassSum;

    // only apply velocity change to non-draggable bodies
    if (!a.draggable) {
        a.velX -= impulse * invMassA * nx;
        a.velZ -= impulse * invMassA * nz;
    }
    if (!b.draggable) {
        b.velX += impulse * invMassB * nx;
        b.velZ += impulse * invMassB * nz;
    }

    // only push non-draggable bodies away from overlap
    double overlap = (a.radius_ndc + b.radius_ndc) - dist;
    if (!a.draggable) {
        a.posX -= nx * overlap * (invMassA / invMassSum);
        a.posZ -= nz * overlap * (invMassA / invMassSum);
    }
    if (!b.draggable) {
        b.posX += nx * overlap * (invMassB / invMassSum);
        b.posZ += nz * overlap * (invMassB / invMassSum);
    }
}

void stepPhysics(vector<Body>& bodies, double dt) {
    int n = bodies.size();

    // gravity between all pairs
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

            // draggable body still exerts gravity on others
            // but we don't update its velocity from gravity
            double ai = force / bodies[i].mass / SIM_SCALE;
            double aj = force / bodies[j].mass / SIM_SCALE;

            if (!bodies[i].draggable) {
                bodies[i].velX += nx * ai * dt;
                bodies[i].velZ += nz * ai * dt;
            }
            if (!bodies[j].draggable) {
                bodies[j].velX -= nx * aj * dt;
                bodies[j].velZ -= nz * aj * dt;
            }
        }
    }

    // integrate — skip draggable (position set by input)
    for (auto& b : bodies) {
        if (!b.alive || b.draggable) continue;
        b.posX += b.velX * dt;
        b.posZ += b.velZ * dt;
    }

    // collision bounce — draggable still participates
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
        mode->width, mode->height, "Solar System — Rogue Star", NULL, NULL);

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

    const double M_SUN     = 1.989e30;
    const double M_MERCURY = 3.285e23;
    const double M_VENUS   = 4.867e24;
    const double M_EARTH   = 5.972e24;
    const double M_MARS    = 6.39e23;
    const double M_ROGUE   = 1.989e30; // same mass as sun

    const double R_EARTH   = 0.42;
    const double R_MERCURY = R_EARTH * 0.387;
    const double R_VENUS   = R_EARTH * 0.723;
    const double R_MARS    = R_EARTH * 1.524;

    double v_mercury = orbitalVel(M_SUN, R_MERCURY);
    double v_venus   = orbitalVel(M_SUN, R_VENUS);
    double v_earth   = orbitalVel(M_SUN, R_EARTH);
    double v_mars    = orbitalVel(M_SUN, R_MARS);

    auto startPos = [](double R, double theta, double v,
                       double& px, double& pz,
                       double& vx, double& vz) {
        px = R * cos(theta);
        pz = R * sin(theta);
        vx = -v * sin(theta);
        vz =  v * cos(theta);
    };

    double px, pz, vx, vz;

    vector<OrbitRing> rings = {
        OrbitRing(R_MERCURY, 0.7f, 0.7f, 0.7f, 0.2f),
        OrbitRing(R_VENUS,   0.9f, 0.7f, 0.3f, 0.2f),
        OrbitRing(R_EARTH,   0.3f, 0.5f, 1.0f, 0.2f),
        OrbitRing(R_MARS,    0.9f, 0.4f, 0.2f, 0.2f),
    };

    // build the scene including the rogue star
    auto buildBodies = [&]() -> vector<Body> {
        vector<Body> b;

        // Sun — fixed at center (draggable=false, mass holds everything)
        b.push_back({ M_SUN, 1408, 0.0, 0.0, 0.0, 0.0,
                      1.0f, 0.92f, 0.2f, 0.05, false });

        // Mercury
        startPos(R_MERCURY, 0.0, v_mercury, px, pz, vx, vz);
        b.push_back({ M_MERCURY, 5427, px, pz, vx, vz,
                      0.72f, 0.70f, 0.68f, 0.020, false });

        // Venus
        startPos(R_VENUS, PI * 0.33, v_venus, px, pz, vx, vz);
        b.push_back({ M_VENUS, 5243, px, pz, vx, vz,
                      0.90f, 0.75f, 0.35f, 0.028, false });

        // Earth
        startPos(R_EARTH, PI * 0.83, v_earth, px, pz, vx, vz);
        b.push_back({ M_EARTH, 5515, px, pz, vx, vz,
                      0.18f, 0.50f, 1.00f, 0.030, false });

        // Mars
        startPos(R_MARS, PI * 1.33, v_mars, px, pz, vx, vz);
        b.push_back({ M_MARS, 3934, px, pz, vx, vz,
                      0.90f, 0.35f, 0.10f, 0.022, false });

        // Rogue star — starts far above, player controlled
        // draggable=true means WASD moves it, gravity doesn't pull it
        b.push_back({ M_ROGUE, 1408, 0.7, 0.6, 0.0, 0.0,
              0.8f, 0.3f, 1.0f,
              0.05, true });

        return b;
    };

    vector<Body> bodies = buildBodies();
    bodies.back().mass = 0.0; // gravity fully OFF at start

    // rogue star is always last in the vector
    // speed of rogue star movement (NDC per second, real time)
    const float ROGUE_SPEED = 0.4f;

    // controls hint
    cout << "\n=== ROGUE STAR CONTROLS ===" << endl;
    cout << "  W / S     — move rogue star up / down" << endl;
    cout << "  A / D     — move rogue star left / right" << endl;
    cout << "  R         — reset simulation" << endl;
    cout << "  ESC       — quit" << endl;
    cout << "  F         — toggle rogue star gravity (freeze/unfreeze)" << endl;
    cout << "==========================\n" << endl;

    bool rogueActive = false; // whether rogue star affects gravity

    const double TIME_SCALE = 3e6;
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now    = glfwGetTime();
        double dt     = (now - lastTime) * TIME_SCALE;
        float  realDt = (float)(now - lastTime); // real seconds for movement
        lastTime = now;
        if (dt > 0.05 * TIME_SCALE) dt = 0.05 * TIME_SCALE;

        // --- rogue star movement (WASD, real time not sim time) ---
        Body& rogue = bodies.back();
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            // W moves up on screen = -Z in sim (because screen_y = posZ * TILT)
            rogue.posZ += ROGUE_SPEED * realDt / TILT;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            rogue.posZ -= ROGUE_SPEED * realDt / TILT;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            rogue.posX -= ROGUE_SPEED * realDt * aspect; // correct for aspect
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            rogue.posX += ROGUE_SPEED * realDt * aspect;

        // F — toggle: first press ramps up gravity, second press disables it
        static bool fWasPressed = false;
        bool fPressed = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
        if (fPressed && !fWasPressed) {
            // toggle rogueActive on each press
            rogueActive = !rogueActive;
            if (!rogueActive) {
                rogue.mass = 0.0; // instantly disable gravity
                cout << "Rogue star gravity: OFF" << endl;
            } else {
                cout << "Rogue star gravity: ramping up..." << endl;
            }
        }
        // while active, keep ramping up toward M_SUN
        if (rogueActive) {
            rogue.mass += (M_SUN / 3.0) * realDt;
            if (rogue.mass > M_SUN) rogue.mass = M_SUN;
        }
        fWasPressed = fPressed;

        // physics
        stepPhysics(bodies, dt);

        // draw
        glClearColor(0.02f, 0.02f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);

        // orbit rings
        for (auto& ring : rings)
            ring.draw(offsetLoc, colorLoc, scaleLoc, aspectLoc, aspect);

        // all bodies except rogue
        for (int i = 0; i < (int)bodies.size() - 1; i++)
            bodies[i].draw(offsetLoc, colorLoc, scaleLoc, aspectLoc, aspect);

        // rogue star with pulsing highlight
        rogue.drawSelected(offsetLoc, colorLoc, scaleLoc,
                           aspectLoc, aspect, (float)now);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            for (auto& b : bodies) b.cleanup();
            bodies      = buildBodies();
            rogueActive = true;
        }
    }

    for (auto& b  : bodies) b.cleanup();
    for (auto& rg : rings)  rg.cleanup();
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}