#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace std;

const float PI = 3.14159265358979323846f;
const double G = 6.6743e-11;
const double SIM_SCALE = 1.5e11;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// ─── SHADERS ────────────────────────────────────────────────────────────────
const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
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

// ─── SHADER BUILD ───────────────────────────────────────────────────────────
unsigned int buildProgram() {
    auto compile = [](GLenum type, const char* src) {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, NULL);
        glCompileShader(s);
        int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[512]; glGetShaderInfoLog(s,512,NULL,log); cerr<<log; }
        return s;
    };
    unsigned int vs = compile(GL_VERTEX_SHADER,   vertexShaderSrc);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    unsigned int p  = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// ─── ORBITAL VELOCITY ───────────────────────────────────────────────────────
double orbitalVel(double mass_central, double r_ndc) {
    double r_m = r_ndc * SIM_SCALE;
    return sqrt(G * mass_central / r_m) / SIM_SCALE;
}

// ─── SPHERE MESH ────────────────────────────────────────────────────────────
struct SphereMesh {
    unsigned int VAO, VBO, EBO;
    int indexCount;

    void build(int stacks = 24, int slices = 24) {
        vector<float>        verts;
        vector<unsigned int> idx;

        for (int i = 0; i <= stacks; i++) {
            float phi = PI * i / stacks;
            for (int j = 0; j <= slices; j++) {
                float theta = 2.0f * PI * j / slices;
                verts.push_back(sin(phi)*cos(theta));
                verts.push_back(cos(phi));
                verts.push_back(sin(phi)*sin(theta));
            }
        }
        for (int i = 0; i < stacks; i++)
            for (int j = 0; j < slices; j++) {
                int a = i*(slices+1)+j, b = a+slices+1;
                idx.push_back(a);   idx.push_back(b);   idx.push_back(a+1);
                idx.push_back(b);   idx.push_back(b+1); idx.push_back(a+1);
            }
        indexCount = (int)idx.size();

        glGenVertexArrays(1,&VAO); glGenBuffers(1,&VBO); glGenBuffers(1,&EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned int), idx.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
    }
    void draw() const {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    }
    void cleanup() {
        glDeleteVertexArrays(1,&VAO);
        glDeleteBuffers(1,&VBO);
        glDeleteBuffers(1,&EBO);
    }
};

// ─── CIRCLE (orbit ring) ────────────────────────────────────────────────────
struct CircleMesh {
    unsigned int VAO, VBO;
    int pts;
    void build(int points = 256) {
        pts = points;
        vector<float> v;
        for (int i = 0; i < points; i++) {
            float a = 2.0f*PI*i/points;
            v.push_back(cos(a)); v.push_back(0.0f); v.push_back(sin(a));
        }
        glGenVertexArrays(1,&VAO); glGenBuffers(1,&VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER,v.size()*sizeof(float),v.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
    }
    void draw() const { glBindVertexArray(VAO); glDrawArrays(GL_LINE_LOOP,0,pts); }
    void cleanup(){ glDeleteVertexArrays(1,&VAO); glDeleteBuffers(1,&VBO); }
};

// ─── SPACETIME GRID ─────────────────────────────────────────────────────────
struct SpacetimeGrid {
    static const int COLS = 50;
    static const int ROWS = 30;

    // base flat positions on the XZ plane at Y = GRID_Y
    float baseX[ROWS][COLS], baseZ[ROWS][COLS];
    // deformed positions
    float bentX[ROWS][COLS], bentY[ROWS][COLS], bentZ[ROWS][COLS];

    static constexpr float GRID_Y   = -0.1f; // sits below the solar system
    static constexpr float EXTENT_X =  2.00f;
    static constexpr float EXTENT_Z =  1.40f;

    unsigned int VAO, VBO;
    static const int MAX_LINES = ROWS*(COLS-1) + COLS*(ROWS-1);
    vector<float> verts; // x,y,z per vertex, 2 verts per segment

    void init() {
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                baseX[r][c] = -EXTENT_X + 2.0f*EXTENT_X * c / (COLS-1);
                baseZ[r][c] = -EXTENT_Z + 2.0f*EXTENT_Z * r / (ROWS-1);
            }
        glGenVertexArrays(1,&VAO); glGenBuffers(1,&VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, MAX_LINES*2*3*sizeof(float),
                     nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
    }

    struct Influencer { float x, z, mass; };

    void update(const vector<Influencer>& inf) {
        const float M_REF   = 1.989e30f;
        const float MAX_DIP = 0.55f;
        const float FALLOFF = 0.50f;

        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                float bx = baseX[r][c], bz = baseZ[r][c];
                float dip = 0.0f, sx = 0.0f, sz = 0.0f;

                for (auto& body : inf) {
                    if (body.mass <= 0.0f) continue;
                    float dx = bx - body.x;
                    float dz = bz - body.z;
                    float d2 = dx*dx + dz*dz + 1e-5f;
                    float d  = sqrt(d2);
                    float s  = (body.mass / M_REF) * MAX_DIP
                               / (1.0f + (d/FALLOFF)*(d/FALLOFF));
                    dip += s;
                    float pull = s * 0.12f / (d + 0.01f);
                    sx -= dx * pull;
                    sz -= dz * pull;
                }

                bentX[r][c] = bx + sx;
                bentY[r][c] = GRID_Y - dip;
                bentZ[r][c] = bz + sz;
            }
        }
    }

    void buildAndUpload() {
        verts.clear();
        auto push = [&](int r, int c) {
            verts.push_back(bentX[r][c]);
            verts.push_back(bentY[r][c]);
            verts.push_back(bentZ[r][c]);
        };
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS-1; c++) { push(r,c); push(r,c+1); }
        for (int c = 0; c < COLS; c++)
            for (int r = 0; r < ROWS-1; r++) { push(r,c); push(r+1,c); }

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferSubData(GL_ARRAY_BUFFER,0,verts.size()*sizeof(float),verts.data());
    }

    void draw(int mvpLoc, int colorLoc, const glm::mat4& vp) {
        buildAndUpload();
        glm::mat4 mvp = vp * glm::mat4(1.0f);
        glUniformMatrix4fv(mvpLoc,1,GL_FALSE,glm::value_ptr(mvp));
        glUniform4f(colorLoc, 0.30f, 0.60f, 1.00f, 0.35f);
        glBindVertexArray(VAO);
        int lineVerts = (ROWS*(COLS-1) + COLS*(ROWS-1)) * 2;
        glDrawArrays(GL_LINES, 0, lineVerts);
    }

    void cleanup(){ glDeleteVertexArrays(1,&VAO); glDeleteBuffers(1,&VBO); }
};

// ─── BODY ───────────────────────────────────────────────────────────────────
struct Body {
    double posX, posY, posZ;   // 3-D position (Y is up)
    double velX, velY, velZ;
    double mass, density;
    float  r, g, b;
    double radius_ndc;
    double visual_radius;
    bool   alive     = true;
    bool   draggable = false;

    Body(double mass, double density,
         double px, double py, double pz,
         double vx, double vy, double vz,
         float r, float g, float b,
         double vis_r = -1.0, bool drag = false)
        : mass(mass), density(density),
          posX(px), posY(py), posZ(pz),
          velX(vx), velY(vy), velZ(vz),
          r(r), g(g), b(b),
          visual_radius(vis_r), draggable(drag)
    { updateRadius(); }

    void updateRadius() {
        if (visual_radius > 0) { radius_ndc = visual_radius; return; }
        double rm = cbrt((3.0*mass/density)/(4.0*PI));
        radius_ndc = rm / SIM_SCALE;
        radius_ndc = max(radius_ndc, 0.012);
        radius_ndc = min(radius_ndc, 0.28);
    }

    glm::vec3 pos() const { return {(float)posX,(float)posY,(float)posZ}; }

    void draw(const SphereMesh& sphere, int mvpLoc, int colorLoc,
              const glm::mat4& vp, float pulse = 1.0f) const {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), pos());
        model = glm::scale(model, glm::vec3((float)radius_ndc * pulse));
        glUniformMatrix4fv(mvpLoc,1,GL_FALSE,glm::value_ptr(vp*model));
        glUniform4f(colorLoc, r, g, b, 1.0f);
        sphere.draw();
    }
};

// ─── PHYSICS ────────────────────────────────────────────────────────────────
void stepPhysics(vector<Body>& bodies, double dt) {
    int n = (int)bodies.size();
    for (int i = 0; i < n; i++) {
        if (!bodies[i].alive) continue;
        for (int j = i+1; j < n; j++) {
            if (!bodies[j].alive) continue;
            double dx = (bodies[j].posX-bodies[i].posX)*SIM_SCALE;
            double dy = (bodies[j].posY-bodies[i].posY)*SIM_SCALE;
            double dz = (bodies[j].posZ-bodies[i].posZ)*SIM_SCALE;
            double d2 = dx*dx+dy*dy+dz*dz;
            double d  = sqrt(d2);
            double soft = (bodies[i].radius_ndc+bodies[j].radius_ndc)*SIM_SCALE*0.3;
            double d2s  = d2 + soft*soft;
            double F    = G*bodies[i].mass*bodies[j].mass/d2s;
            double nx=dx/d, ny=dy/d, nz=dz/d;
            double ai = F/bodies[i].mass/SIM_SCALE;
            double aj = F/bodies[j].mass/SIM_SCALE;
            if (!bodies[i].draggable) {
                bodies[i].velX+=nx*ai*dt; bodies[i].velY+=ny*ai*dt; bodies[i].velZ+=nz*ai*dt;
            }
            if (!bodies[j].draggable) {
                bodies[j].velX-=nx*aj*dt; bodies[j].velY-=ny*aj*dt; bodies[j].velZ-=nz*aj*dt;
            }
        }
    }
    for (auto& b : bodies) {
        if (!b.alive || b.draggable) continue;
        b.posX+=b.velX*dt; b.posY+=b.velY*dt; b.posZ+=b.velZ*dt;
    }
    // elastic bounce
    for (int i = 0; i < n; i++) {
        if (!bodies[i].alive) continue;
        for (int j = i+1; j < n; j++) {
            if (!bodies[j].alive) continue;
            glm::dvec3 d = {bodies[j].posX-bodies[i].posX,
                            bodies[j].posY-bodies[i].posY,
                            bodies[j].posZ-bodies[i].posZ};
            double dist = glm::length(d);
            double minD = bodies[i].radius_ndc+bodies[j].radius_ndc;
            if (dist < minD && dist > 0) {
                glm::dvec3 n = d/dist;
                double dvn = (bodies[i].velX-bodies[j].velX)*n.x
                            +(bodies[i].velY-bodies[j].velY)*n.y
                            +(bodies[i].velZ-bodies[j].velZ)*n.z;
                if (dvn <= 0) continue;
                double iA = bodies[i].draggable?0.0:1.0/bodies[i].mass;
                double iB = bodies[j].draggable?0.0:1.0/bodies[j].mass;
                double iS = iA+iB; if(iS==0) continue;
                double imp = 1.6*dvn/iS;
                if (!bodies[i].draggable){ bodies[i].velX-=imp*iA*n.x; bodies[i].velY-=imp*iA*n.y; bodies[i].velZ-=imp*iA*n.z; }
                if (!bodies[j].draggable){ bodies[j].velX+=imp*iB*n.x; bodies[j].velY+=imp*iB*n.y; bodies[j].velZ+=imp*iB*n.z; }
                double ov = minD-dist;
                if (!bodies[i].draggable){ bodies[i].posX-=n.x*ov*(iA/iS); bodies[i].posY-=n.y*ov*(iA/iS); bodies[i].posZ-=n.z*ov*(iA/iS); }
                if (!bodies[j].draggable){ bodies[j].posX+=n.x*ov*(iB/iS); bodies[j].posY+=n.y*ov*(iB/iS); bodies[j].posZ+=n.z*ov*(iB/iS); }
            }
        }
    }
}

// ─── MAIN ───────────────────────────────────────────────────────────────────
int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
    GLFWwindow* window = glfwCreateWindow(
        mode->width, mode->height, "Solar System 3D — Rogue Star", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    int winW, winH;
    glfwGetFramebufferSize(window, &winW, &winH);
    float aspect = (float)winW/(float)winH;

    unsigned int prog = buildProgram();
    int mvpLoc   = glGetUniformLocation(prog, "uMVP");
    int colorLoc = glGetUniformLocation(prog, "uColor");

    // ── shared meshes ──
    SphereMesh sphere; sphere.build(32,32);
    CircleMesh circle; circle.build(256);
    SpacetimeGrid grid; grid.init();

    // ── camera (orbit camera) ──
    float camYaw   = 30.0f;   // degrees around Y
    float camPitch = 28.0f;   // degrees above XZ
    float camDist  = 2.8f;
    glm::vec3 camTarget = {0,0,0};

    // ── constants ──
    const double M_SUN     = 1.989e30;
    const double M_MERCURY = 3.285e23;
    const double M_VENUS   = 4.867e24;
    const double M_EARTH   = 5.972e24;
    const double M_MARS    = 6.39e23;
    const double M_ROGUE   = 1.989e30;

    const double R_EARTH   = 0.55;
    const double R_MERCURY = R_EARTH * 0.387;
    const double R_VENUS   = R_EARTH * 0.723;
    const double R_MARS    = R_EARTH * 1.524;

    double v_mercury = orbitalVel(M_SUN, R_MERCURY);
    double v_venus   = orbitalVel(M_SUN, R_VENUS);
    double v_earth   = orbitalVel(M_SUN, R_EARTH);
    double v_mars    = orbitalVel(M_SUN, R_MARS);

    // orbit ring radii stored for drawing
    vector<double> ringRadii = { R_MERCURY, R_VENUS, R_EARTH, R_MARS };
    vector<glm::vec4> ringColors = {
        {0.7f,0.7f,0.7f,0.25f},
        {0.9f,0.7f,0.3f,0.25f},
        {0.3f,0.5f,1.0f,0.25f},
        {0.9f,0.4f,0.2f,0.25f},
    };

    auto buildBodies = [&]() -> vector<Body> {
        vector<Body> b;
        // Sun
        b.push_back({M_SUN,1408, 0,0,0, 0,0,0, 1.0f,0.92f,0.2f, 0.07,false});
        // Mercury — orbits in XZ plane
        b.push_back({M_MERCURY,5427, R_MERCURY,0,0, 0,0,v_mercury, 0.72f,0.70f,0.68f, 0.022,false});
        // Venus
        double vvx=-v_venus*sin(PI*0.33), vvz=v_venus*cos(PI*0.33);
        b.push_back({M_VENUS,5243, R_VENUS*cos(PI*0.33),0,R_VENUS*sin(PI*0.33), vvx,0,vvz, 0.90f,0.75f,0.35f, 0.030,false});
        // Earth
        double vex=-v_earth*sin(PI*0.83), vez=v_earth*cos(PI*0.83);
        b.push_back({M_EARTH,5515, R_EARTH*cos(PI*0.83),0,R_EARTH*sin(PI*0.83), vex,0,vez, 0.18f,0.50f,1.00f, 0.032,false});
        // Mars
        double vmx=-v_mars*sin(PI*1.33), vmz=v_mars*cos(PI*1.33);
        b.push_back({M_MARS,3934, R_MARS*cos(PI*1.33),0,R_MARS*sin(PI*1.33), vmx,0,vmz, 0.90f,0.35f,0.10f, 0.024,false});
        // Rogue star — starts to the side, player controlled
        b.push_back({M_ROGUE,1408, 1.2,0.5,0.0, 0,0,0, 0.8f,0.3f,1.0f, 0.07,true});
        return b;
    };

    vector<Body> bodies = buildBodies();
    bodies.back().mass = 0.0;

    const float ROGUE_SPEED = 0.8f;
    bool rogueActive = false;

    // mouse state for camera orbit
    double lastMouseX = 0, lastMouseY = 0;
    bool   mouseDown  = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    cout << "\n=== SOLAR SYSTEM 3D — ROGUE STAR ===" << endl;
    cout << "  W/S/A/D       — move rogue star (XZ plane)"  << endl;
    cout << "  SPACE / Y     — move rogue star down / up"   << endl;
    cout << "  F             — toggle rogue gravity"        << endl;
    cout << "  R             — reset"                       << endl;
    cout << "  Mouse drag    — orbit camera"                << endl;
    cout << "  Scroll        — zoom"                        << endl;
    cout << "  ESC           — quit"                        << endl;
    cout << "=====================================\n"       << endl;

    // scroll callback for zoom
    glfwSetWindowUserPointer(window, &camDist);
    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy){
        float& d = *(float*)glfwGetWindowUserPointer(w);
        d -= (float)dy * 0.15f;
        d  = max(0.8f, min(d, 8.0f));
    });

    const double TIME_SCALE = 3e6;
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now    = glfwGetTime();
        double dt     = (now - lastTime) * TIME_SCALE;
        float  realDt = (float)(now - lastTime);
        lastTime = now;
        if (dt > 0.05 * TIME_SCALE) dt = 0.05 * TIME_SCALE;

        // ── mouse camera orbit ──────────────────────────────────
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        int mb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        if (mb == GLFW_PRESS) {
            if (mouseDown) {
                camYaw   += (float)(mx - lastMouseX) * 0.35f;
                camPitch -= (float)(my - lastMouseY) * 0.35f;
                camPitch  = max(-89.0f, min(89.0f, camPitch));
            }
            mouseDown = true;
        } else { mouseDown = false; }
        lastMouseX = mx; lastMouseY = my;

        // ── rogue star movement ─────────────────────────────────
        Body& rogue = bodies.back();
        // build camera-relative right/forward vectors for intuitive controls
        float yawRad = glm::radians(camYaw);
        glm::vec3 camRight   = { cos(yawRad), 0, sin(yawRad)};
        glm::vec3 camForward = {-sin(yawRad), 0, cos(yawRad)};

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            rogue.posX += camForward.x*ROGUE_SPEED*realDt;
            rogue.posZ += camForward.z*ROGUE_SPEED*realDt;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            rogue.posX -= camForward.x*ROGUE_SPEED*realDt;
            rogue.posZ -= camForward.z*ROGUE_SPEED*realDt;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            rogue.posX -= camRight.x*ROGUE_SPEED*realDt;
            rogue.posZ -= camRight.z*ROGUE_SPEED*realDt;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            rogue.posX += camRight.x*ROGUE_SPEED*realDt;
            rogue.posZ += camRight.z*ROGUE_SPEED*realDt;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            rogue.posY -= ROGUE_SPEED * realDt;
        if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
            rogue.posY += ROGUE_SPEED * realDt;

        // ── F toggle gravity ────────────────────────────────────
        static bool fWas = false;
        bool fNow = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
        if (fNow && !fWas) {
            rogueActive = !rogueActive;
            if (!rogueActive) { rogue.mass = 0.0; cout<<"Rogue gravity: OFF\n"; }
            else               cout<<"Rogue gravity: ramping...\n";
        }
        if (rogueActive) { rogue.mass += (M_SUN/3.0)*realDt; if(rogue.mass>M_SUN) rogue.mass=M_SUN; }
        fWas = fNow;

        stepPhysics(bodies, dt);

        // ── spacetime grid update ───────────────────────────────
        {
            vector<SpacetimeGrid::Influencer> infl;
            for (auto& b : bodies)
                if (b.alive && b.mass > 0.0)
                    infl.push_back({(float)b.posX, (float)b.posZ, (float)b.mass});
            grid.update(infl);
        }

        // ── build VP matrix ─────────────────────────────────────
        float pitchRad = glm::radians(camPitch);
        float yawR     = glm::radians(camYaw);
        glm::vec3 camPos = camTarget + glm::vec3(
            camDist * cos(pitchRad) * sin(yawR),
            camDist * sin(pitchRad),
            camDist * cos(pitchRad) * cos(yawR));
        glm::mat4 view = glm::lookAt(camPos, camTarget, {0,1,0});
        glm::mat4 proj = glm::perspective(glm::radians(50.0f), aspect, 0.01f, 50.0f);
        glm::mat4 VP   = proj * view;

        // ── render ──────────────────────────────────────────────
        glClearColor(0.02f,0.02f,0.07f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);

        // 1. spacetime grid (depth-write off so it doesn't occlude planets)
        glDepthMask(GL_FALSE);
        grid.draw(mvpLoc, colorLoc, VP);
        glDepthMask(GL_TRUE);

        // 2. orbit rings (in XZ plane, scale by radius)
        for (int i = 0; i < (int)ringRadii.size(); i++) {
            glm::mat4 model = glm::scale(glm::mat4(1.0f),
                                         glm::vec3((float)ringRadii[i]));
            glUniformMatrix4fv(mvpLoc,1,GL_FALSE,glm::value_ptr(VP*model));
            auto& c = ringColors[i];
            glUniform4f(colorLoc, c.r, c.g, c.b, c.a);
            circle.draw();
        }

        // 3. planets + sun
        for (int i = 0; i < (int)bodies.size()-1; i++)
            bodies[i].draw(sphere, mvpLoc, colorLoc, VP);

        // 4. rogue star with pulse
        float pulse = 1.0f + 0.04f*sin((float)now*4.0f);
        rogue.draw(sphere, mvpLoc, colorLoc, VP, pulse);
        // white ring
        glm::mat4 rm = glm::translate(glm::mat4(1.0f), rogue.pos());
        rm = glm::scale(rm, glm::vec3((float)rogue.radius_ndc * pulse * 1.12f));
        glUniformMatrix4fv(mvpLoc,1,GL_FALSE,glm::value_ptr(VP*rm));
        glUniform4f(colorLoc,1,1,1,0.45f);
        circle.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            bodies      = buildBodies();
            rogueActive = false;
            bodies.back().mass = 0.0;
        }
    }

    sphere.cleanup(); circle.cleanup(); grid.cleanup();
    glDeleteProgram(prog);
    glfwTerminate();
    return 0;
}