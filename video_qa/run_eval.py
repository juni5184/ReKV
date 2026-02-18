import os
import argparse
import subprocess
import multiprocessing


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)


def eval_mlvu(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/mlvu/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        anno_path = "data/mlvu/dev_debug_mc.json" if not args.sample else "data/mlvu/dev_debug_mc_sample.json"
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/qaego4d/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    anno_path = "data/qaego4d/test_mc.json" if not args.sample else "data/qaego4d/test_mc_sample.json"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_videomme(args):   
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/videomme/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    anno_path = "data/videomme/test.json" if not args.sample else "data/videomme/test_long.json"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_0.5b_vanilla', 'llava_ov_7b_vanilla'])
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'qaego4d', 'videomme'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if args.debug == 'true':
        import debugpy
        def connect_debugpy():
            if not debugpy.is_client_connected():
                debugpy.listen(("0.0.0.0", 2345))
                print("Waiting for debugger to attach...")
                debugpy.wait_for_client()
            debugpy.configure(subProcess=True)
        connect_debugpy()
    
    func_dic = {
        'mlvu': eval_mlvu,
        'qaego4d': eval_qaego4d,
        'videomme': eval_videomme,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
