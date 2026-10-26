"""Recalculate all external development comparisons from shared-site score tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate import ranking

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results/external')
    args=parser.parse_args()
    protocol=json.loads((args.results/'protocol.json').read_text(encoding='utf-8'))
    records=[]
    for panel in protocol['panels']:
        frame=pd.read_csv(args.results/panel['file'],dtype={'chromosome':str},float_precision='round_trip')
        key=protocol['genomic_keys']+(['condition'] if 'condition' in frame else [])
        if frame.duplicated(key).any() or not frame.target.isin([0,1]).all() or frame.target.nunique()!=2:
            raise ValueError('Each panel requires unique condition/locus keys and both binary classes.')
        if len(frame)!=panel['observations'] or frame.target.sum()!=panel['positives']:
            raise ValueError('Panel counts disagree with the recorded evaluation universe.')
        for seed in protocol['seeds']:
            scores={model:frame[f'{model}_seed{seed}'] for model in
                    ['context_aware_trees_v1','local_xgboost','full_catboost','original_42']}
            scores.update(m6anet=frame.m6anet_ranking,singlemod=frame.singlemod_ranking)
            for model,prediction in scores.items():
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite prediction.')
                records.append(dict(panel=panel['panel'],species=panel['species'],seed=seed,model=model,
                    observations=len(frame),positives=int(frame.target.sum()),**ranking(frame.target,prediction)))
    result=pd.DataFrame(records)
    result.to_csv(args.results/'metrics_by_seed.csv',index=False)
    means=result.groupby(['panel','species','model'],sort=False)[['observations','positives','roc_auc','pr_auc','ap']].mean().reset_index()
    means.to_csv(args.results/'metrics_mean.csv',index=False)
    result.loc[result.seed.eq(protocol['deployment_seed'])].to_csv(args.results/'metrics_deployed_fit.csv',index=False)
    print(means.loc[means.model.isin(['context_aware_trees_v1','m6anet','original_42'])].to_string(index=False))


if __name__=='__main__':
    main()
