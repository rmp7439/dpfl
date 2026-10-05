import time
import json
import os
import torch
from contextlib import contextmanager

class RuntimeProfiler:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.timings = {
            "rounds": {},
            "clients": {}
        }
        self.current_round = 0
        self.round_start_time = 0
        self.global_start_time = time.time()
        
    @contextmanager
    def measure(self, category, name, round_num=None, client_id=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.time() - start
        
        if category == "round" and round_num is not None:
            if round_num not in self.timings["rounds"]:
                self.timings["rounds"][round_num] = {}
            if name not in self.timings["rounds"][round_num]:
                self.timings["rounds"][round_num][name] = 0
            self.timings["rounds"][round_num][name] += duration
            
        elif category == "client" and client_id is not None:
            if client_id not in self.timings["clients"]:
                self.timings["clients"][client_id] = {}
            if name not in self.timings["clients"][client_id]:
                self.timings["clients"][client_id][name] = []
            self.timings["clients"][client_id][name].append(duration)
            
        elif category == "global":
            if name not in self.timings:
                self.timings[name] = 0
            self.timings[name] += duration

    def set_round(self, round_num):
        self.current_round = round_num

    def save(self, metadata):
        os.makedirs(self.output_dir, exist_ok=True)
        self.timings["total_runtime"] = time.time() - self.global_start_time
        self.timings["metadata"] = metadata
        
        if torch.cuda.is_available():
            self.timings["metadata"]["gpu_name"] = torch.cuda.get_device_name(0)
            
        out_path = os.path.join(self.output_dir, "runtime_profile.json")
        with open(out_path, "w") as f:
            json.dump(self.timings, f, indent=4)
        print(f"Saved runtime profile to {out_path}")

# Global profiler instance for easy access
profiler = None

def init_profiler(output_dir):
    global profiler
    profiler = RuntimeProfiler(output_dir)
    return profiler

def get_profiler():
    global profiler
    return profiler
