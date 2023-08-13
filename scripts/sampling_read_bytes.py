import subprocess
import re
import argparse
import sys

NCU_REPORT_NUM_KERNELS = 6
INDPTR_KERNEL_IDX = 0
SAMPLING_KERNEL_IDX = 5


def run_bench(args,
              indptr_device,
              indices_device,
              probs_device=None,
              save_ncu_report="test"):
    bench_args = [
        f"--dataset={args.dataset}",
        f"--root={args.root}",
        f"--batch-size={args.batch_size}",
        f"--fan-out={args.fan_out}",
        f"--indptr-device={indptr_device}",
        f"--indices-device={indices_device}",
        f"--probs-device={probs_device}",
    ]
    if args.bias:
        bench_args.append("--bias")
    ncu_args = [
        f"--section=regex:MemoryWorkloadAnalysis",
        f"--nvtx",
        f"--nvtx-include=sampling]",
        f"--export={save_ncu_report}",
        f"--force-overwrite",
    ]
    p1 = subprocess.Popen(
        [args.ncu, *ncu_args, sys.executable, args.bench, *bench_args],
        stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "Seeds_num"],
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
    parser.add_argument("--dataset",
                        default=None,
                        choices=["ogbn-products", "ogbn-papers100M"])
    parser.add_argument("--root", default="dataset/")
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--fan-out", type=str, default='10,10,10')
    parser.add_argument("--batch-size", default="5000", type=int)
    parser.add_argument("--ncu", type=str, default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--sector-size", type=int, default=32)
    parser.add_argument("--bench", type=str, default="ncu_sampling.py")
    args = parser.parse_args()

    result = run_bench(args,
                       indptr_device="cpu",
                       indices_device="cpu",
                       probs_device="cpu",
                       save_ncu_report="sampling_cpu")
    seeds_num = int(result.split()[-1])

    run_bench(args,
              indptr_device="gpu",
              indices_device="cpu",
              probs_device="cpu",
              save_ncu_report="sampling_indptr_gpu")

    run_bench(args,
              indptr_device="cpu",
              indices_device="gpu",
              probs_device="cpu",
              save_ncu_report="sampling_indices_gpu")

    if args.bias:
        run_bench(args,
                  indptr_device="cpu",
                  indices_device="cpu",
                  probs_device="gpu",
                  save_ncu_report="sampling_probs_gpu")

    cpu_result_cpu = get_cpu_read_sectors(args.ncu, "sampling_cpu.ncu-rep")
    gpu_result_cpu = get_gpu_read_sectors(args.ncu, "sampling_cpu.ncu-rep")
    cpu_result_indptr_gpu = get_cpu_read_sectors(
        args.ncu, "sampling_indptr_gpu.ncu-rep")
    gpu_result_indptr_gpu = get_gpu_read_sectors(
        args.ncu, "sampling_indptr_gpu.ncu-rep")
    cpu_result_indices_gpu = get_cpu_read_sectors(
        args.ncu, "sampling_indices_gpu.ncu-rep")
    gpu_result_indices_gpu = get_gpu_read_sectors(
        args.ncu, "sampling_indices_gpu.ncu-rep")
    if args.bias:
        cpu_result_probs_gpu = get_cpu_read_sectors(
            args.ncu, "sampling_probs_gpu.ncu-rep")
        gpu_result_probs_gpu = get_gpu_read_sectors(
            args.ncu, "sampling_probs_gpu.ncu-rep")

    num_layers = len(args.fan_out.split(','))
    indptr_host_read_sectors = 0
    indptr_gpu_read_sectors = 0
    indices_host_read_sectors = 0
    indices_gpu_read_sectors = 0
    if args.bias:
        probs_host_read_sectors = 0
        probs_gpu_read_sectors = 0

    for i in range(num_layers):
        indptr_host_read_sectors += int(
            re.sub(
                ",", "", cpu_result_cpu[i * NCU_REPORT_NUM_KERNELS +
                                        INDPTR_KERNEL_IDX].split()[-1])
        ) - int(
            re.sub(
                ",", "", cpu_result_indptr_gpu[i * NCU_REPORT_NUM_KERNELS +
                                               INDPTR_KERNEL_IDX].split()[-1]))
        indptr_gpu_read_sectors += int(
            re.sub(
                ",", "", gpu_result_indptr_gpu[i * NCU_REPORT_NUM_KERNELS +
                                               INDPTR_KERNEL_IDX].split()[-1])
        ) - int(
            re.sub(
                ",", "", gpu_result_cpu[i * NCU_REPORT_NUM_KERNELS +
                                        INDPTR_KERNEL_IDX].split()[-1]))

        indices_host_read_sectors += int(
            re.sub(
                ",", "",
                cpu_result_cpu[i * NCU_REPORT_NUM_KERNELS +
                               SAMPLING_KERNEL_IDX].split()[-1])) - int(
                                   re.sub(
                                       ",", "", cpu_result_indices_gpu[
                                           i * NCU_REPORT_NUM_KERNELS +
                                           SAMPLING_KERNEL_IDX].split()[-1]))
        indices_gpu_read_sectors += int(
            re.sub(
                ",", "", gpu_result_indices_gpu[i * NCU_REPORT_NUM_KERNELS +
                                                SAMPLING_KERNEL_IDX].split()
                [-1])) - int(
                    re.sub(
                        ",", "",
                        gpu_result_cpu[i * NCU_REPORT_NUM_KERNELS +
                                       SAMPLING_KERNEL_IDX].split()[-1]))

        if args.bias:
            probs_host_read_sectors += int(
                re.sub(
                    ",", "", cpu_result_cpu[i * NCU_REPORT_NUM_KERNELS +
                                            SAMPLING_KERNEL_IDX].split()[-1])
            ) - int(
                re.sub(
                    ",", "",
                    cpu_result_probs_gpu[i * NCU_REPORT_NUM_KERNELS +
                                         SAMPLING_KERNEL_IDX].split()[-1]))
            probs_gpu_read_sectors += int(
                re.sub(
                    ",", "", gpu_result_probs_gpu[i * NCU_REPORT_NUM_KERNELS +
                                                  SAMPLING_KERNEL_IDX].split()
                    [-1])) - int(
                        re.sub(
                            ",", "",
                            gpu_result_cpu[i * NCU_REPORT_NUM_KERNELS +
                                           SAMPLING_KERNEL_IDX].split()[-1]))

    print("[indptr] GPU read bytes per seed = {:.0f}".format(
        indptr_gpu_read_sectors * args.sector_size / seeds_num))
    print("[indptr] CPU read bytes per seed = {:.0f}".format(
        indptr_host_read_sectors * args.sector_size / seeds_num))

    print("[indices] GPU read bytes per seed = {:.0f}".format(
        indices_gpu_read_sectors * args.sector_size / seeds_num))
    print("[indices] CPU read bytes per seed = {:.0f}".format(
        indices_host_read_sectors * args.sector_size / seeds_num))

    if args.bias:
        print("[probs] GPU read bytes per seed = {:.0f}".format(
            probs_gpu_read_sectors * args.sector_size / seeds_num))
        print("[probs] CPU read bytes per seed = {:.0f}".format(
            probs_host_read_sectors * args.sector_size / seeds_num))

    subprocess.run(["rm", "sampling_indptr_gpu.ncu-rep"])
    subprocess.run(["rm", "sampling_indices_gpu.ncu-rep"])
    subprocess.run(["rm", "sampling_probs_gpu.ncu-rep"])
    subprocess.run(["rm", "sampling_cpu.ncu-rep"])
