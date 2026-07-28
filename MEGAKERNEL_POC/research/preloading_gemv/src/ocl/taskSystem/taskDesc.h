#pragma once

typedef enum TaskType { GEMV } TaskType;

// TODO __alignas(16) ?
typedef struct TaskDesc {
  __global const void* restrict weights;
  __global const void* restrict input;
  __global void* restrict output;
  int id;
  TaskType taskType;
} TaskDesc;