import os
import time
import csv
import psutil
import socket
import platform

hostname = socket.gethostname()  
machine_id = f"{hostname}_{platform.node()}"  

log_file = f"memory_log_{machine_id}.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Machine_ID", "Hostname", "Process_ID", "Memory_Usage_MB", "Swap_Usage_MB", "CPU_Usage", "Page_Faults"])

def get_memory_stats():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S") 
    process_list = []

    for proc in psutil.process_iter(['pid', 'memory_info', 'cpu_percent']):
        try:
            mem_usage = proc.info['memory_info'].rss / (1024 * 1024)  
            swap_usage = proc.info['memory_info'].vms / (1024 * 1024)  
            cpu_usage = proc.info['cpu_percent']
            page_faults = proc.info['memory_info'].pfaults if hasattr(proc.info['memory_info'], 'pfaults') else 0

            process_list.append([timestamp, machine_id, hostname, proc.info['pid'], mem_usage, swap_usage, cpu_usage, page_faults])

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return process_list

while True:
    data = get_memory_stats()
    if data:
        with open(log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
    time.sleep(1) 