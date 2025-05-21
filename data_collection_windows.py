import threading
import psutil
import platform
import time
import csv
import socket
import os
import ctypes
from ctypes import wintypes

# Setup
hostname = socket.gethostname()
machine_id = f"{hostname}_{platform.node()}"
system_os = platform.system()
log_file = f"multithreaded_memory_log_{machine_id}.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "OS", "Machine_ID", "Hostname", "RAM_Usage_MB", "Swap_Usage_MB", "CPU_Usage", "Page_Faults_Delta"])

system_data = {
    "ram_usage": 0,
    "swap_usage": 0,
    "cpu_usage": 0,
    "page_faults": 0
}

data_lock = threading.Lock()

def monitor_memory_cpu():
    while True:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        total_swap_usage_mb = 0
        for proc in psutil.process_iter(['memory_info']):
            try:
                swap_usage = getattr(proc.info['memory_info'], 'vms', 0)
                total_swap_usage_mb += swap_usage / (1024 * 1024)
            except:
                continue
        with data_lock:
            system_data["ram_usage"] = mem.used / (1024 * 1024)
            system_data["swap_usage"] = total_swap_usage_mb
            system_data["cpu_usage"] = cpu
        time.sleep(0.1)

# Windows ctypes structs for Page Faults
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
OpenProcess = ctypes.windll.kernel32.OpenProcess
CloseHandle = ctypes.windll.kernel32.CloseHandle
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

def get_process_pagefaults(pid):
    h_process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h_process:
        return 0
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    success = GetProcessMemoryInfo(h_process, ctypes.byref(counters), counters.cb)
    CloseHandle(h_process)
    if not success:
        return 0
    return counters.PageFaultCount

def monitor_page_faults():
    previous_total_faults = 0
    while True:
        total_faults = 0
        for proc in psutil.process_iter(['pid']):
            try:
                total_faults += get_process_pagefaults(proc.pid)
            except:
                continue
        with data_lock:
            delta = total_faults - previous_total_faults
            system_data["page_faults"] = max(0, delta)
            previous_total_faults = total_faults
        time.sleep(1)

def log_data():
    while True:
        timestamp = time.time()
        with data_lock:
            ram = system_data["ram_usage"]
            swap = system_data["swap_usage"]
            cpu = system_data["cpu_usage"]
            faults = system_data["page_faults"]
        with open(log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, system_os, machine_id, hostname, ram, swap, cpu, faults])
            file.flush()
        print(f"[Logger] RAM: {ram:.2f}MB, Swap: {swap:.2f}MB, CPU: {cpu:.2f}%, Page Faults Delta: {faults}")
        time.sleep(0.5)

if __name__ == "__main__":
    print(f"[System] Starting multithreaded system monitoring on {system_os}...")

    threading.Thread(target=monitor_memory_cpu, daemon=True).start()
    threading.Thread(target=monitor_page_faults, daemon=True).start()
    threading.Thread(target=log_data, daemon=True).start()

    while True:
        time.sleep(1)
