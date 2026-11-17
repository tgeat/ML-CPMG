from benchmark import *
with threadpool_limits(1):
    psutil.Process().cpu_affinity([19])
    case=next(cases('static'));target=DATA/'calibrate.json'
    rows=json.loads(target.read_text())
    for cap in (256,512):
        if not any('error' not in r and r['cap']==cap and r['stall']==.2 for r in rows):
            rows.append(run_one(case,'ML-CPMG',0,cap,.2))
            save_json(target,rows)
    # A cost/size compromise is chosen only on the named calibration system.
    good=[r for r in rows if r.get('converged') and r['method']=='ML-CPMG']
    feasible=[r for r in good if r['operator_complexity']<=3.0]
    fastest=min(r['total_seconds'] for r in feasible)
    acceptable=[r for r in feasible if r['total_seconds']<=1.05*fastest]
    chosen=min(acceptable,key=lambda r:r['operator_complexity'])
    selected=dict(max_basins=chosen['cap'],stall_ratio=chosen['stall'],prolong_omega=.45,
      selection='Exploratory cost/size choice: impose Cop <= 3.0, then select smallest complexity within 5% of the fastest feasible ML cold run. The faster fine-level algebraic fallback is separately reported, including its higher Cop and loss of basin-labelled aggregation.',
      calibration_case='SPE10-static',calibration_fine_smoother='automatic rule, not triggered on this coefficient field',
      fixed_configuration_evaluation='SPE10 three mobility states and Brugge three states',
      internal_SPE10_smoother='same forced vertical line on all four internal methods, including the algebraic-only control',
      primary_SA_theta=.04,SA_sensitivity=[0,.04,.1])
    save_json(DATA/'selected_config.json',selected)
    save_csv(DATA/'calibrate_clean.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows if 'error' not in r])
    print(json.dumps(selected,indent=2),flush=True)
