import subprocess
import re
import argparse
import sys


def run_bench(args, device, save_ncu_report="test"):
    bench_args = [
        f"--num-nodes={args.num_nodes}",
        f"--feat-dim={args.feat_dim}",
        f"--batch-size={args.batch_size}",
        f"--device={device}",
    ]
    ncu_args = [
        f"--section=regex:MemoryWorkloadAnalysis",
        f"--nvtx",
        f"--nvtx-include=loading]",
        f"--export={save_ncu_report}",
        f"--force-overwrite",
    ]
    p1 = subprocess.Popen(
        [args.ncu, *ncu_args, sys.executable, args.bench, *bench_args],
        stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "Nids_num"],
                          stdin=p1.stdout,
                          stdout=subprocess.PIPE)
    p1.stdout.close()
    out, _ = p2.communicate()
    return out.strip().decode("gbk")


def get_cpu_read_sectors(ncu_path, report_file):
    args = [
        f"--import={report_file}",
        f"--page=raw",
    ]
    p1 = subprocess.Popen([ncu_path, *args], stdout=subprocess.PIPE)
    p2 = subprocess.Popen([
        "grep",
        "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum"
    ],
                          stdin=p1.stdout,
                          stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["grep", "sector "],
                          stdin=p2.stdout,
                          stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    out, _ = p3.communicate()
    return out.strip().decode("gbk").splitlines()


def get_gpu_read_sectors(ncu_path, report_file):
    args = [
        f"--import={report_file}",
        f"--page=raw",
    ]
    p1 = subprocess.Popen([ncu_path, *args], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "dram__sectors_read.sum"],
                          stdin=p1.stdout,
                          stdout=subprocess.PIPE)
    p1.stdout.close()
    out, _ = p2.communicate()
    return out.strip().decode("gbk").splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10000000)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--ncu", type=str, default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--sector-size", type=int, default=32)
    parser.add_argument("--bench", type=str, default="ncu_feature.py")
    args = parser.parse_args()

    result = run_bench(args, device="cpu", save_ncu_report="feature_cpu")
    loading_nids_num = int(result.split()[-1])
    run_bench(args, device="gpu", save_ncu_report="feature_gpu")

    result = get_cpu_read_sectors(args.ncu, "feature_cpu.ncu-rep")
    host_read_sectors_on_host = int(re.sub(",", "", result[0].split()[-1]))

    result = get_gpu_read_sectors(args.ncu, "feature_cpu.ncu-rep")
    gpu_read_sectors_on_host = int(re.sub(",", "", result[0].split()[-1]))

    result = get_cpu_read_sectors(args.ncu, "feature_gpu.ncu-rep")
    host_read_sectors_on_gpu = int(re.sub(",", "", result[0].split()[-1]))

    result = get_gpu_read_sectors(args.ncu, "feature_gpu.ncu-rep")
    gpu_read_sectors_on_gpu = int(re.sub(",", "", result[0].split()[-1]))

    print("[Feature loading] GPU read bytes per node = {:.0f}".format(
        (gpu_read_sectors_on_gpu - gpu_read_sectors_on_host) *
        args.sector_size / loading_nids_num))
    print("[Feature loading] CPU read bytes per node = {:.0f}".format(
        (host_read_sectors_on_host - host_read_sectors_on_gpu) *
        args.sector_size / loading_nids_num))

    subprocess.run(["rm", "feature_cpu.ncu-rep"])
    subprocess.run(["rm", "feature_gpu.ncu-rep"])
