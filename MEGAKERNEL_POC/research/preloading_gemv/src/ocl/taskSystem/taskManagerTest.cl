#include "taskManager.h"

__kernel void taskManagerTest(__constant const TaskManager* taskManager,
                              __global int* taskIds) {
  if (get_local_id(0) == 0) {
    const __global TaskDesc* task = GetNextTask(taskManager);

    while (task != NULL) {
      // printf("Task with id <%d> is executed by work-group <%lu>\n", task->id,
      //        get_group_id(0));
      const int executedID = taskIds[task->id];

      if (executedID == -1) {
        taskIds[task->id] = get_group_id(0);
      } else {
        taskIds[task->id] = -2;
      }

      task = GetNextTask(taskManager);
    }
  }
}