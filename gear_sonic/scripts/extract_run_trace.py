import sys, glob, os, json
import numpy as np, torch
import gear_sonic.trl.trainer.ppo_trainer  # installs the class-move compat shim

runs = {
 "scratch": "logs_rl/TRL_R1_Track/manager/universal_token/all_modes/sonic_r1_scratch_full-20260903_145050",
 "any2any": "logs_rl/TRL_R1_Any2Any/manager/universal_token/all_modes/sonic_r1_any2any_poseoffset_s7-20260902_162350",
}
res = {}
for tag, d in runs.items():
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "model_step_*.pt"))):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        st, ar, psd = ck["state"], ck["args"], ck["policy_state_dict"]
        rew = np.array(st.rewbuffer); ln = np.array(st.lenbuffer)
        row = dict(step=int(st.global_step), lr=float(ar.learning_rate),
                   reward=float(np.mean(rew.sum(axis=-1))) if len(rew) else None,
                   length=float(np.mean(ln)) if len(ln) else None, n_eps=int(len(ln)))
        if "std" in psd:
            row["std_mean"] = float(psd["std"].mean()); row["std_max"] = float(psd["std"].max())
        lora = {}
        for k in list(psd):
            if k.endswith("lora_A"):
                b = k[:-6]
                dW = (psd[b+"lora_B"].float() @ psd[k].float()) * (32.0/16.0)
                lora[b] = round(float(dW.norm()/psd[b+"weight"].float().norm()), 6)
        if lora: row["lora_rel"] = lora
        rows.append(row)
        print(f"{tag} step={row['step']:>5} lr={row['lr']:.4g} rew={row['reward']:.3f} "
              f"len={row['length']:.1f} std={row.get('std_mean',float('nan')):.4f} "
              f"lora_max={max(lora.values()) if lora else 0}", flush=True)
        del ck
    res[tag] = rows
json.dump(res, open("/tmp/run_trace.json", "w"), indent=1)
print("OK")
