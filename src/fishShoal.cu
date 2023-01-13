/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

#include <vector_types.h>

#include "config.h"
#include "boid.h"
#include "draw_boids_cpu.h"
#include "update_boids_cpu.h"
#include "draw_boids_gpu.h"
#include "update_boids_gpu.h"

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD 0.30f
#define REFRESH_DELAY 10 // ms

////////////////////////////////////////////////////////////////////////////////
// constants

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0; // FPS count for averaging
int fpsLimit = 1; // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

const char *sSDKsample = "simpleGL (VBO)";
char *window_title = "Fish shoal simulation on %s (%d fishes): fps: %3.1f, separtion: %f, alignment: %f, cohision: %f, inertia: %f, interaction radius: %f, velocity: %f";

Boid *boids;

#define MAX(a, b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
bool StartRender(int argc, char **argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
// run cuda computation
void runCuda(struct cudaGraphicsResource **vbo_resource);

int main(int argc, char **argv)
{

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    printf("\n");

    StartRender(argc, argv);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, window_title,gpu_render?"GPU":"CPU", num_boids, avgFPS, factor_separation, factor_alignment, factor_cohesion, factor_intertia, radius, velocity);
    glutSetWindowTitle(fps);
}
// Initialize GL
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow(window_title);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 100.0);

    SDK_CHECK_ERROR_GL();

    return true;
}
// Start rendering simulation
bool StartRender(int argc, char **argv)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutCloseFunc(cleanup);

    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    // init boids
    boids = init_boids();

    // start rendering mainloop
    glutMainLoop();
    return true;
}
void launch_kernel(float3 *pos)
{
    // todo:
    //  1. drawing arrow in 3d
    const int num_threads = std::min(1024, num_boids);
    const int num_blocks = std::ceil((float)num_boids / (float)num_threads);
    update_boids_position<<<num_blocks, num_threads>>>(boids, interaction_radius_2, velocity, factor_separation, factor_alignment, factor_cohesion, factor_intertia);
    draw_boids<<<num_blocks, num_threads>>>(pos, boids, num_boids);
}
// Run the Cuda part of the computation
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));

    // launch cuda computation
    launch_kernel(dptr);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void runCpu()
{
    float3 *pos = (float3 *)malloc(sizeof(float3) * 3 * num_boids);
    if(pos==NULL){
        printf("malloc failed");
        exit(1);
    }
    update_boids_cpu(boids);
    draw_boids_cpu(pos, boids);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 3 * num_boids, pos, GL_STATIC_DRAW);
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif
#endif
// Create VBO
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = num_boids * 9 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}
// Delete VBO
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// Display callback
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    if (animate)
    {
        if (gpu_render)
        {
            runCuda(&cuda_vbo_resource);
        }
        else
        {
            runCpu();
        }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, num_boids * 3);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    // g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}
void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}
void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}
// Keyboard events handler
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (27): // esc
        glutDestroyWindow(glutGetWindow());
        free_boids(boids);
        return;
    case 'q':
    {
        factor_separation += change_force_factor_step;
        return;
    }
    case 'a':
    {
        factor_separation -= change_force_factor_step;
        if (factor_separation < 0)
        {
            factor_separation = 0;
        }
        return;
    }
    case 'w':
    {
        factor_alignment += change_force_factor_step;
        return;
    }
    case 's':
    {
        factor_alignment -= change_force_factor_step;
        if (factor_alignment < 0)
        {
            factor_alignment = 0;
        }
        return;
    }
    case 'e':
    {
        factor_cohesion += change_force_factor_step;
        return;
    }

    case 'd':
    {
        factor_cohesion -= change_force_factor_step;
        if (factor_cohesion < 0)
        {
            factor_cohesion = 0;
        }
        return;
    }
    case 'r':
    {
        factor_intertia += change_force_factor_step;
        return;
    }
    case 'f':
    {
        factor_intertia -= change_force_factor_step;
        if (factor_intertia < 0)
        {
            factor_intertia = 0;
        }
        return;
    }
    case 't':
    {
        radius += chanage_radious_step;
        return;
    }
    case 'g':
    {
        radius -= chanage_radious_step;
        if (radius < 0)
        {
            radius = 0;
        }
        return;
    }
    case 32: // space
    {
        animate = !animate;
        return;
    }
    case 'n':
    {
        boids = init_boids();
        return;
    }
    case 'y':
    {
        velocity += change_velocity_step;
        return;
    }
    case 'h':
    {
        velocity -= change_velocity_step;
        return;
    }
      case 9: // tab
    {
        gpu_render=!gpu_render;
        return;
    }
    }
}
// Mouse event handlers
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
