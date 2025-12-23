# Local WorkerGroup implementation
# Mimics RayWorkerGroup but executes everything locally in the same process (or handled via simple threading if needed, but synchronous for now)

import logging
from copy import deepcopy
from typing import Any, List

from verl.protocol import DataProto, _padding_size_key
from verl.single_controller.base import WorkerGroup, ClassWithInitArgs

def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            # Dispatch: split args for workers (if needed)
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            
            # Execute: run on workers (locally)
            output = execute_fn(method_name, *args, **kwargs)
            
            # Since we are local/sync, 'output' is already the result, not a future.
            
            # Collect: aggregate results
            output = collect_fn(self, output)
            
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    return type(method_name, (Functor,), {})()

class LocalWorkerGroup(WorkerGroup):
    """
    A group of local workers (objects) running in the same process.
    Replaces RayWorkerGroup for single-node refactoring.
    """
    def __init__(self, resource_pool=None, cls_with_init: ClassWithInitArgs = None, **kwargs):
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.cls_with_init = cls_with_init
        
        # In local mode, we assume a single "worker" effectively, OR we actually instantiate multiple objects
        # if the code logic expects distinct worker instances (e.g. holding different state).
        # For simplicity, if world_size is implied to be 1, we just have one.
        # But if the logic splits data across "workers", we should support N distinct objects.
        
        # For now, let's support N=1 by default for simplicity, unless we strictly need more.
        # However, to be "correct", we should instantiate as many as requested.
        # RayWorkerGroup typically uses resource_pool to decide count.
        
        # For the simplifying refactor, we likely want just 1 worker if possible, 
        # but let's see if we can support N.
        
        self._workers = []
        # If resource pool is provided, use it. If not, maybe just 1.
        if resource_pool:
            count = resource_pool.world_size
        else:
            count = 1
            
        self._world_size = count
        
        if cls_with_init:
            # Instantiate workers
            for i in range(count):
                # We need to handle 'rank' and 'world_size' env vars potentially?
                # or just pass them if the worker class accepts them.
                # The 'ClassWithInitArgs' stores args/kwargs.
                # We might need to inject rank/world_size if the worker expects it in __init__ 
                # or if it reads env vars. 
                # NOTE: The original Worker class reads ENV VARS in __init__.
                # We might need to mock os.environ for each instantiation if they check it.
                # For now, let's assume they might.
                
                # Check if we can just instantiate.
                # If the Worker class is 'Worker', it reads os.environ.
                # We might need to set them temporarily?
                
                # Doing a hacky temp env var set for each init?
                # Or we bypass that if we can.
                worker_instance = cls_with_init.cls(*cls_with_init.args, **cls_with_init.kwargs)
                self._workers.append(worker_instance)

            self._bind_worker_method(self.cls_with_init.cls, func_generator)

    def _is_worker_alive(self, worker):
        return True

    def execute_rank_zero_async(self, method_name, *args, **kwargs):
        # execute on worker 0 synchronously
        worker = self._workers[0]
        method = getattr(worker, method_name)
        return method(*args, **kwargs)

    def execute_rank_zero_sync(self, method_name, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all_async(self, method_name, *args, **kwargs):
        # execute on all workers synchronously and return list
        # logic similar to RayWorkerGroup.execute_all_async but looping
        
        results = []
        length = len(self._workers)
        
        # Check if args are sharded
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
             if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                for i, worker in enumerate(self._workers):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    method = getattr(worker, method_name)
                    results.append(method(*sliced_args, **sliced_kwargs))
                return results

        # Broadcast args
        for worker in self._workers:
            method = getattr(worker, method_name)
            results.append(method(*args, **kwargs))
        return results

    def execute_all_sync(self, method_name, *args, **kwargs):
        return self.execute_all_async(method_name, *args, **kwargs)

