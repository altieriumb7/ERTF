# pbt_orchestrator.py
import argparse, os, subprocess, time, random, json, shutil
def read_summary(outdir):
    p = os.path.join(outdir, 'latest_ckpt_summary.json')
    if os.path.exists(p):
        try:
            return json.load(open(p,'r'))
        except:
            return None
    return None

def mutate_role_embeddings(src_dir, dst_dir, std=0.02):
    import torch, numpy as np, os, shutil
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(os.path.join(src_dir,'role_embeddings.pt')):
        data = torch.load(os.path.join(src_dir,'role_embeddings.pt'), map_location='cpu')
        out = {}
        for k,v in data.items():
            arr = v.numpy() + np.random.normal(scale=std, size=v.numpy().shape)
            out[k] = torch.tensor(arr)
        torch.save(out, os.path.join(dst_dir,'role_embeddings.pt'))
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir,name); d = os.path.join(dst_dir,name)
        if not os.path.exists(d):
            try:
                shutil.copy(s,d)
            except:
                pass

def launch(cmd):
    print('Launching:', cmd)
    return subprocess.Popen(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_cmd_template', required=True)
    parser.add_argument('--population', type=int, default=4)
    parser.add_argument('--base_out', default='pbt_runs')
    parser.add_argument('--pbt_every', type=int, default=300)
    args = parser.parse_args()
    os.makedirs(args.base_out, exist_ok=True)
    procs = {}
    meta = {}
    for i in range(args.population):
        outdir = os.path.join(args.base_out, f'worker_{i}'); os.makedirs(outdir, exist_ok=True)
        seed = random.randint(1,999999)
        cmd = args.worker_cmd_template.format(seed=seed, output_dir=outdir)
        p = launch(cmd); procs[i]=p; meta[i]={'out':outdir,'seed':seed}
    try:
        while True:
            time.sleep(args.pbt_every)
            summaries=[]
            for i in range(args.population):
                s = read_summary(meta[i]['out']) or {'val_summary':{'val_avg_reward':-9e9}}
                summaries.append((i, s.get('val_summary', {}).get('val_avg_reward', -9e9), meta[i]))
            summaries.sort(key=lambda x:x[1], reverse=True)
            half = args.population//2; elites=summaries[:half]; losers=summaries[half:]
            for loser in losers:
                idx=loser[0]; elite=random.choice(elites); elite_out=elite[2]['out']; loser_out=meta[idx]['out']
                p=procs[idx]; 
                if p.poll() is None: p.kill(); p.wait()
                mutate_role_embeddings(elite_out, loser_out, std=0.02)
                seed=random.randint(1,999999); cmd=args.worker_cmd_template.format(seed=seed, output_dir=loser_out)
                pnew=launch(cmd); procs[idx]=pnew; meta[idx]['seed']=seed
    except KeyboardInterrupt:
        print('Stopping workers...')
        for i,p in procs.items():
            if p.poll() is None:
                p.kill(); p.wait()

if __name__ == '__main__':
    main()
