"""Refit fixed predictors on full held-out physical slots. / 按冻结分折重拟合。"""
import argparse
import json
from pathlib import Path
import numpy as np
from stats import cluster_ratio, compare, tensor_features, ridge_predict

HERE=Path(__file__).resolve().parent
BASES=("joint_lmmse","glsnet","dncnn")
GROUPS=("deep","low","moderate","strong")
SETTINGS=dict(folds=5,ridge=1e-4,bootstrap_seed=2026111802,bootstrap_repetitions=2000)


def predict_errors(X,y,folds):
    """Train-only column standardization; no intercept penalty. / 仅在训练折上归一化。"""
    prediction=np.empty(y.size)
    fold=np.repeat(folds,y.shape[1])
    for i in range(5):
        test=fold==i
        prediction[test]=ridge_predict(X[~test],y.ravel()[~test],X[test],1e-4)
    return np.square(prediction.reshape(y.shape)-y)


def check(actual,expected):
    if isinstance(expected,dict):
        for k,v in expected.items():check(actual[k],v)
    elif expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=3e-11)


def replay(p, reference):
    result={}
    with np.load(HERE/(p+"_slots.npz"),allow_pickle=False) as shared:
        gamma=shared["gamma"]; folds=shared["folds"];count=shared["count"]
        np.testing.assert_array_equal(count.sum(axis=1),np.full(len(folds),972))
        for base in BASES:
            with np.load(HERE/(p+"_"+base+".npz"),allow_pickle=False) as a:
                group={}
                for j,k in enumerate(GROUPS):
                    group[k]=dict(mean_total_error_energy=cluster_ratio(a["energy"][:,j],count[:,j]),
                                  excess_over_true_diagonal_reference=cluster_ratio(a["excess"][:,j],count[:,j]))
                y=a["target"]; e=a["error_over_noise"]
                X=tensor_features(gamma.ravel(),e.ravel(),3)
                feats=np.column_stack((X,X*a["radial"].reshape(-1,1),X*a["phase"].reshape(-1,1)))
                err=predict_errors(feats,y,folds)
                matched=compare(y,gamma,e,err,folds,SETTINGS)
                absolute_ce = a["excess"] + shared["reference_ce_sum"]
                receiver = dict(mean_rate_bits_per_data_re=6*(1-absolute_ce.sum()/count.sum()/np.log(2)),
                                nmse_data_re_db=10*np.log10(a["energy"].sum()/shared["power"].sum()),
                                deep_share_of_all_ce=absolute_ce[:,0].sum()/absolute_ce.sum(),
                                deep_fraction=count[:,0].sum()/count.sum(),
                                deep_gamma_le_one_fraction=shared["gamma_le_one_count"][:,0].sum()/count[:,0].sum())
                if base != "dncnn":
                    relative = shared[base+"_relative_error_sum"].sum(axis=0)/count.sum(axis=0)
                    receiver["weak_to_strong_relative_error_ratio"]=relative[0]/relative[-1]
                row=dict(groups=group,matched=matched,receiver=receiver)
                if base!="dncnn":
                    scalar=predict_errors(tensor_features(gamma.ravel(),e.ravel(),5),y,folds)
                    row["reduction_36_percent"]=100*(scalar.mean()-err.mean())/scalar.mean()
                check(row,reference[base]);result[base]=row
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=HERE.parent/"outputs"/"waveform.json")
    args=ap.parse_args(); expected=json.loads((HERE/"expected.json").read_text())
    output={}
    for p in "abcd":
        output[p.upper()]=replay(p,expected[p.upper()])
        print(p.upper(),"passed",flush=True)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
