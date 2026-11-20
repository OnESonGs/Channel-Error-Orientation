"""Recompute matched-model checks from scrambled Sobol draws. / 重放理论数值核验。"""
import argparse
import json
from pathlib import Path
import numpy as np
import kernel
import scope_summary
HERE = Path(__file__).resolve().parent


def evaluate(dataset, mode):
    record = json.loads((HERE / (dataset + ".json")).read_text())
    plan, archived = record["settings"], record["replicates"]
    model = plan["model"]
    powers = sorted({row["power"] for row in archived})
    gammas = np.atleast_1d(model["gamma"]).tolist()
    reference = {}
    for gamma in gammas:
        g = kernel.mode_weights(model["r"], model["r"]**2/gamma, model["qm"], 256)
        reference[gamma] = dict(d=g["radial"]+g["phase"]-2*g["orthogonal"])
    if mode != "summary":
        index = {(r["seed"],r["gamma"],r["epsilon_over_r"],r["x"],r["power"]):r for r in archived}
        seeds=plan["integration"]["scramble_seeds"]
        ps=powers if mode=="full" else powers[:1]
        for seed in seeds if mode=="full" else seeds[:1]:
            base=kernel.sobol_standard_inputs(max(ps),seed,model["qm"])
            for gamma in gammas if mode=="full" else gammas[:1]:
                for eps in model["epsilon_over_r"] if mode=="full" else model["epsilon_over_r"][:1]:
                    for x in model["x"] if mode=="full" else [0,.5,1]:
                        rows=kernel.evaluate_nested(base,x,eps,gamma,ps,model["r"],model["qm"],model["kappa"])
                        for row in rows:
                            old=index[(seed,gamma,eps,x,row["power"])]
                            for k in ("scale","risk_nats_per_bit"):
                                np.testing.assert_allclose(row["fit"][k],old["fit"][k],rtol=2e-8,atol=3e-14)
                            np.testing.assert_allclose(row["derivative_wrt_a"],old["derivative_wrt_a"],
                                                       rtol=3e-7,atol=4e-14)
                            old.update(row)
    if dataset == "scope":
        result=scope_summary.summarize(archived,plan,reference[gammas[0]],powers)
        tolerances={str(t):scope_summary.sampled_prefix(result["amplitudes"],t) for t in plan["approximation_tolerances"]}
        return {"amplitude_grid":result["amplitudes"],"quadratic_accuracy_prefix":tolerances}
    rows,adjacent=kernel.summarize(archived,reference,model,powers)
    points=[{k:row[k] for k in ("epsilon_over_r","x","normalized_gap","normalized_gap_se")}
            for row in rows if row["power"]==powers[-1] and row["gamma"]==10]
    return {"gamma":10,"kappa":model["kappa"],"rows":points,
            "positive_adjacent_comparisons":sum(x["gain_sign"]>0 for x in adjacent if x["power"]==powers[-1])}


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=["summary","quick","full"],default="quick")
    ap.add_argument("--output",type=Path,default=HERE.parent/"outputs"/"theory.json")
    args=ap.parse_args()
    result={name:evaluate(name,args.mode) for name in ("orientation","scope")}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(args.mode, result["orientation"]["positive_adjacent_comparisons"],result["scope"]["quadratic_accuracy_prefix"])
