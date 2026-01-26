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


def eval_qaego4d(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/qaego4d/{args.solver}/{args.retrieve_size}-{args.sample_fps}"
    os.makedirs(save_dir, exist_ok=True)
    solver = f'{args.solver}_offline_vqa'
    if args.sample:
        # anno_path = "data/qaego4d/test_mc_sample.json" # sample 55
        anno_path = "data/qaego4d/test_mc_sample_2.json" # sample 20
    else:
        anno_path = "data/qaego4d/test_mc.json"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = [
                "python", f"video_qa/{solver}.py",
                "--model", args.model,
                "--sample_fps", str(args.sample_fps),
                "--n_local", str(args.n_local),
                "--retrieve_size", str(args.retrieve_size),
                "--save_dir", save_dir,
                "--anno_path", anno_path,
                "--debug", str(args.debug),
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--solver", str(args.solver),
            ]
            p = multiprocessing.Process(
                target=exec,
                args=(
                    cmd,
                    True,
                    f'{4*idx},{4*idx+1},{4*idx+2},{4*idx+3}' if args.model == 'llava_ov_72b' else str(idx)
                )
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


def eval_mlvu(args):
    num_chunks = args.num_chunks
    if num_chunks == 1:
        save_dir = f"results/debug/{args.model}/mlvu/{args.retrieve_size}-{args.sample_fps}"
    else:   
        save_dir = f"results/{args.model}/mlvu/{args.retrieve_size}-{args.sample_fps}"
    os.makedirs(save_dir, exist_ok=True)
    solver = f'{args.solver}_offline_vqa'
    if args.sample:
        anno_path = "data/mlvu/dev_debug_mc_sample.json"
    else:
        anno_path = "data/mlvu/dev_debug_mc.json"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = [
                "python", f"video_qa/{solver}.py",
                "--model", args.model,
                "--sample_fps", str(args.sample_fps),
                "--n_local", str(args.n_local),
                "--retrieve_size", str(args.retrieve_size),
                "--save_dir", save_dir,
                "--anno_path", anno_path,
                "--debug", str(args.debug),
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--solver", str(args.solver),
            ]
            p = multiprocessing.Process(
                target=exec,
                args=(
                    cmd,
                    True,
                    f'{4*idx},{4*idx+1},{4*idx+2},{4*idx+3}' if args.model == 'llava_ov_72b' else str(idx)
                )
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


def eval_videomme(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/videomme/{args.retrieve_size}-{args.sample_fps}"
    os.makedirs(save_dir, exist_ok=True)
    solver = f'{args.solver}_offline_vqa'
    if args.sample:
        anno_path = "data/videomme/test_short_sample.json"
    else:
        anno_path = "data/videomme/test_short.json" # now in tmux 0
        # anno_path = "data/videomme/test_medium.json"
        # anno_path = "data/videomme/test_long.json"
        # anno_path = "data/videomme/test.json" # now in tmux 1
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = [
                "python", f"video_qa/{solver}.py",
                "--model", args.model,
                "--sample_fps", str(args.sample_fps),
                "--n_local", str(args.n_local),
                "--retrieve_size", str(args.retrieve_size),
                "--save_dir", save_dir,
                "--anno_path", anno_path,
                "--debug", str(args.debug),
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--solver", str(args.solver),
            ]
            p = multiprocessing.Process(
                target=exec,
                args=(
                    cmd,
                    True,
                    f'{4*idx},{4*idx+1},{4*idx+2},{4*idx+3}' if args.model == 'llava_ov_72b' else str(idx)
                )
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'qwen2_5_vl_3b', 'qwen2_5_vl_7b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['qaego4d', 'mlvu', 'videomme'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--solver", type=str, default='vanilla', choices=['vanilla', 'rekv'])
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
        'qaego4d': eval_qaego4d,
        'mlvu': eval_mlvu,
        'videomme': eval_videomme,
    }
    
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
