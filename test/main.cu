#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>



// Constants
const int NUM_BOIDS = 1000;
const int NUM_THREADS = 256;
const float TIME_STEP = 0.01f;
const float BOID_RADIUS = 0.01f;
const float NEIGHBOR_RADIUS = 0.1f;
const float SEPARATION_WEIGHT = 0.1f;
const float COHESION_WEIGHT = 0.1f;
const float ALIGNMENT_WEIGHT = 0.1f;

// Boid data
struct Boid {
  float3 position;
  float3 velocity;
};

// CUDA device memory pointers
Boid* d_boids;

// OpenGL buffers and VAO
GLuint vbo, vao;

// CUDA-OpenGL interop resources
cudaGraphicsResource* cudaVBO;

// Kernel to update the position and velocity of the boids
__global__ void updateBoidPositions(Boid* boids, int numBoids, float timeStep, float R, float separationWeight, float cohesionWeight, float alignmentWeight) {
  int boidIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (boidIdx < numBoids) {
    // Compute the average position and velocity of the neighbors within radius R
    float3 avgPos = make_float3(0, 0, 0);
    float3 avgVel = make_float3(0, 0, 0);
    int numNeighbors = 0;
    for (int i = 0; i < numBoids; i++) {
      if (i == boidIdx) continue;  // Skip the current boid
      float dist = distance(boids[boidIdx].position, boids[i].position);
      if (dist < R) {
        avgPos += boids[i].position;
        avgVel += boids[i].velocity;
        numNeighbors++;
      }
    }
    if (numNeighbors > 0) {
      avgPos /= numNeighbors;
      avgVel /= numNeighbors;
    }

    // Compute the separation, cohesion, and alignment forces
    float3 separationForce = separationWeight * (boids[boidIdx].position - avgPos);
    float3 cohesionForce = cohesionWeight * (avgPos - boids[boidIdx].position);
    float3 alignmentForce = alignmentWeight * (avgVel - boids[boidIdx].velocity);

    // Update the velocity and position of the current boid based on the forces
    boids[boidIdx].velocity += separationForce + cohesionForce + alignmentForce;
    boids[boidIdx].position += boids[boidIdx].velocity * timeStep;
  }
}

// Initializes the boids with random positions and velocities
void initBoids(Boid* boids, int numBoids) {
  for (int i = 0; i < numBoids; i++) {
    boids[i].position.x = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    boids[i].position.y = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    boids[i].position.z = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    boids[i].velocity.x = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    boids[i].velocity.y = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    boids[i].velocity.z = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
  }
}

// Renders the boids
void render(Boid* boids, int numBoids) {
  // Bind the VAO and VBO
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  // Map the CUDA device memory to the VBO
  cudaGraphicsMapResources(1, &cudaVBO);
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&boids, &num_bytes, cudaVBO);

  // Set the vertex attrib pointers
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Boid), (void*)0);
  glEnableVertexAttribArray(0);

  // Draw the boids as points
  glPointSize(3.0f);
  glDrawArrays(GL_POINTS, 0, numBoids);

  // Unbind the VAO and VBO
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Unmap the CUDA device memory from the VBO
  cudaGraphicsUnmapResources(1, &cudaVBO);
}

int main() {
  // Initialize GLFW
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create the window
  GLFWwindow* window = glfwCreateWindow(800, 600, "Boids Simulation", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // Generate the VAO and VBO
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  // Allocate device memory for the boids
  cudaMalloc((void**)&d_boids, NUM_BOIDS * sizeof(Boid));

  // Initialize the boids
  initBoids(d_boids, NUM_BOIDS);

  // Register the VBO with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsNone);

  // Set the viewport
  glViewport(0, 0, 800, 600);

  // Set the clear color
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  // Run the simulation loop
  while (!glfwWindowShouldClose(window)) {
    // Poll for events
    glfwPollEvents();

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Execute the kernel to update the boids
    dim3 gridDim((NUM_BOIDS + NUM_THREADS - 1) / NUM_THREADS);
    updateBoidPositions<<<gridDim, NUM_THREADS>>>(d_boids, NUM_BOIDS, TIME_STEP, NEIGHBOR_RADIUS, SEPARATION_WEIGHT, COHESION_WEIGHT, ALIGNMENT_WEIGHT);

    // Render the boids
    render(d_boids, NUM_BOIDS);

    // Swap the buffers
    glfwSwapBuffers(window);
  }

  // Free the device memory and OpenGL resources
  cudaFree(d_boids);
  cudaGraphicsUnregisterResource(cudaVBO);
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);

  // Terminate GLFW
  glfwTerminate();

  return 0;
}
``
