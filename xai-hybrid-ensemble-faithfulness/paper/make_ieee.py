"""Every number, table body and macro in the IEEE paper is generated here from the result files."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
E = json.load(open(R / "study2_3/summary_study2_3.json"))
S1 = json.load(open(R / "confirmatory/summary.json"))
S4 = json.load(open(R / "study4/summary_study4.json"))
S67 = json.load(open(R / "study6_7/summary_study6_7.json")); S6, S7 = S67["study6"], S67["study7"]
V = json.load(open(R / "revision/summary_revision.json")); C = V.get("components")
V2 = json.load(open(R / "revision/summary_revision2.json"))
SV = json.load(open(R / "revision/summary_supervisor.json"))
DS = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
LAB = {"wdbc": "WDBC", "dr_debrecen": "DR Debrecen", "cleveland": "Cleveland", "ilpd": "ILPD", "parkinsons": "Parkinsons",
       "mammographic": "Mammographic", "ctg": "CTG", "cdc_diabetes": "CDC Diabetes"}
INFO = {"wdbc": ("Breast mass (17)", "569", "30", "37", "1.00"), "dr_debrecen": ("Retinopathy (329)", "1146", "19", "53", "1.00"),
        "cleveland": ("Coronary disease (45)", "303", "13", "46", "0.59"), "ilpd": ("Liver disease (225)", "570", "10", "71", "0.96"),
        "parkinsons": ("Parkinson's, voice (174)", "195$^a$", "22", "75", "1.00"), "mammographic": ("Mammographic mass (161)", "961", "4", "46", "0.72"),
        "ctg": ("Fetal state (193)", "2114", "21", "22", "0.96"), "cdc_diabetes": ("Diabetes, survey$^b$", "5000", "21", "14", "0.45")}
c4h = lambda lo, hi: f"[{lo:.3f}, {hi:.3f}]".replace("-", "$-$")
c3 = lambda lo, hi: f"[{lo:.3f}, {hi:.3f}]".replace("-", "$-$")
mn = lambda x, n=3: f"{x:+.{n}f}".replace("-", "$-$")
c2 = lambda lo, hi: f"[{lo:.2f}, {hi:.2f}]".replace("-", "$-$")
fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
fq = lambda p: "{<}\\,0.001" if p < 0.001 else f"{{=}}\\,{p:.3f}"
star = lambda r: "$^*$" if r["p"] < 0.05 else ""
allds = {**E["study1_recomputed"], **E["study2"]}


def primary(d):
    v = allds[d]
    if d in S1["primary"]["H2"]:
        return S1["primary"]["H1"][d], S1["primary"]["H2"][d], S1["primary"]["holm"][f"H2_{d}"]["p_holm"], v["H2"]["p"]
    return v["H1"], v["H2"], v["H2"].get("p_holm", v["H2"]["p"]), v["H2"]["p"]


# Table: datasets and audit (full width)
AUD = {  # positive class, n (after processing), d, pos %, duplicates, missing, repeated subjects, |rho|max
 "wdbc": ("WDBC (17)", "Malignant", "569", "30", "37", "0", "0", "No", "1.00"),
 "dr_debrecen": ("DR Debrecen (329)", "Signs of retinopathy", "1146", "19", "53", "5 removed", "0", "No", "1.00"),
 "cleveland": ("Cleveland (45)", "Angiographic disease (num$>$0)", "303", "13", "46", "0", "6 cells / 6 rows", "No", "0.59"),
 "ilpd": ("ILPD (225)", "Liver patient", "570", "10", "71", "13 removed", "4 cells / 4 rows", "No", "0.96"),
 "parkinsons": ("Parkinsons (174)", "Parkinson's disease", "195", "22", "75", "0", "0", "32 subjects$^a$", "1.00"),
 "mammographic": ("Mammographic (161)", "Malignant (BI-RADS excluded)", "961", "4", "46", "353 kept$^b$", "160 cells / 130 rows", "No", "0.72"),
 "ctg": ("Cardiotocography (193)", "Suspect or pathologic (NSP 2--3)", "2114", "21", "22", "12 removed", "0", "No", "0.96"),
 "cdc_diabetes": ("CDC Diabetes (891)$^c$", "Prediabetes or diabetes", "5000", "21", "14", "42 kept$^b$", "0", "No", "0.45")}
EX = SV["exclusions"]
rows = []
for d in DS:
    a = AUD[d]; e = EX[d]
    rows.append(" & ".join(a) + f" & {e['included']} ({e['pct_included']:.0f}\\%) \\\\")
Path("tab_data.tex").write_text("\n".join(rows))

# Table: main results
rows = []
for d in DS:
    r1, r2, ph, praw = primary(d); s6 = S6["per_dataset"][d]["d_feat"]; tu = S7["tuned"][d]["H2"]; kb = S4["per_dataset"][d]["b_ndf"]
    rk = allds[d]["H2_knn"]
    pcell = f"{fp(ph)}" + ("" if d != "cdc_diabetes" else "$^\\dagger$")
    eqm = "$^{e}$" if SV["auc_tost"][d]["equivalent"] else ""
    rows.append(f"{LAB[d]} & {mn(r1['mean_diff'])}{eqm} & {mn(r2['mean_diff'])}{star(r2)} {c2(r2['ci_low'], r2['ci_high'])} & {pcell} & {mn(rk['mean_diff'])}{star(rk)} & {mn(s6['mean_diff'])}{star(s6)} & {mn(kb['mean_diff'])}{star(kb)} & {mn(tu['mean_diff'])}{star(tu)} \\\\")
M = V["meta"]; p6 = S6["pooled"]["d_feat"]; pb = S4["pooled"]["b_ndf"]; pt = S7["pooled"]["tuned_H2"]
rows.append("\\midrule")
rows.append(f"Pooled (DL) & {mn(M['H1']['pooled'], 4)} & {mn(M['H2']['pooled'])} {c2(M['H2']['ci_low'], M['H2']['ci_high'])} & & {mn(M['H2_knn']['pooled'])} & {mn(p6['pooled'])} & {mn(pb['pooled'])} & {mn(pt['pooled'])} \\\\")
rows.append(f"\\quad $p$ & {fp(M['H1']['p'])} & {fp(M['H2']['p'])} & & {fp(M['H2_knn']['p'])} & {fp(p6['p'])} & {fp(pb['p'])} & {fp(pt['p'])} \\\\")
rows.append(f"\\quad HKSJ 95\\% CI & -- & {c2(*M['H2']['hksj_ci'])} & -- & {c2(*M['H2_knn']['hksj_ci'])} & -- & -- & -- \\\\")
Path("tab_main.tex").write_text("\n".join(rows))

# Table: robustness (components, tuned by S* type)
rows = []
if C:
    PP = C["pooled_pairs"]
    for key, lab in [("e3-rf", "HE $-$ RF"), ("e3-xgb", "HE $-$ XGB"), ("e1-s_star", "SV $-$ S*"), ("e2-s_star", "ST $-$ S*"),
                     ("rf-lr", "RF $-$ LR"), ("xgb-lr", "XGB $-$ LR")]:
        a, b = PP[key], PP[key + "_knn"]
        rows.append(f"{lab} & {mn(a['pooled'])} {c2(a['ci_low'], a['ci_high'])} & {fp(a['p'])} & {mn(b['pooled'])} & {fp(b['p'])} \\\\")
T = V["tuned_by_sstar_type"]; TM = V2["tuned_mixed"]
rows.append("\\midrule")
TA = V2["tuned_mixed_all"]
rows.append(f"Tuned, all ({TA['d']['n_folds']} folds) & {mn(TA['d']['est'])} {c2(*TA['d']['ci'])} & {fp(TA['d']['p'])} & {mn(TA['dk']['est'])} & {fp(TA['dk']['p'])} \\\\")
for k, lab in [("linear", "Tuned, S* = LR"), ("nonlinear", "Tuned, S* = SVM/RF/XGB")]:
    a, b = TM[k + "_d"], TM[k + "_dk"]
    rows.append(f"{lab} ({a['n_folds']} folds) & {mn(a['est'])} {c2(*a['ci'])} & {fp(a['p'])} & {mn(b['est'])} & {fp(b['p'])} \\\\")
Path("tab_robust.tex").write_text("\n".join(rows))

# Table: remedy
rows = []
for d in DS:
    v = S6["per_dataset"][d]; s4 = S4["contrast"][d]["c_minus_a"]
    rows.append(f"{LAB[d]} & {mn(s4['mean_diff'])}{star(s4)} & {mn(v['d_feat']['mean_diff'])}{star(v['d_feat'])} & {mn(v['d_grp']['mean_diff'])}{star(v['d_grp'])} & {mn(v['R']['mean_diff'])}{star(v['R'])} & {mn(v['PL']['mean_diff'])}{star(v['PL'])} \\\\")
q = S6["pooled"]; s4p = S4["pooled"]["contrast_c_minus_a"]
rows.append("\\midrule")
rows.append(f"Pooled & {mn(s4p['pooled'])} & {mn(q['d_feat']['pooled'])} & {mn(q['d_grp']['pooled'])} & {mn(q['R']['pooled'])} & {mn(q['PL']['pooled'])} \\\\")
rows.append(f"\\quad $p$ & {fp(s4p['p'])} & {fp(q['d_feat']['p'])} & {fp(q['d_grp']['p'])} & {fp(q['R']['p'])} & {fp(q['PL']['p'])} \\\\")
Path("tab_remedy.tex").write_text("\n".join(rows))

Mx = {}
def put(k, v): Mx[k] = v
put("auc", mn(M["H1"]["pooled"], 4)); put("aucci", c2(M["H1"]["ci_low"], M["H1"]["ci_high"]).replace("0.00", "0.00"))
put("aucciL", f"{M['H1']['ci_low']:.4f}".replace("-", "$-$")); put("aucciH", f"{M['H1']['ci_high']:.4f}")
put("ndf", mn(M["H2"]["pooled"])); put("ndfci", c2(M["H2"]["ci_low"], M["H2"]["ci_high"])); put("ndfp", fq(M["H2"]["p"])); put("ndfI", f"{100*M['H2']['I2']:.0f}")
put("ndfhk", c2(*M["H2"]["hksj_ci"])); put("ndfhkp", fq(M["H2"]["hksj_p"])); put("ndfpi", c2(*M["H2"]["pred_int"]))
loo = M["H2"]["loo"]; put("looMin", mn(max(v["pooled"] for v in loo.values()))); put("looMax", mn(min(v["pooled"] for v in loo.values())))
put("looDR", mn(loo["dr_debrecen"]["pooled"])); put("looDRp", fq(loo["dr_debrecen"]["p"])); put("looPmax", fq(max(v["p"] for v in loo.values())))
put("knn", mn(M["H2_knn"]["pooled"])); put("knnp", fq(M["H2_knn"]["p"]))
put("heldout", mn(p6["pooled"])); put("heldoutci", c2(p6["ci_low"], p6["ci_high"])); put("heldoutp", fq(p6["p"]))
put("kernel", mn(pb["pooled"])); put("kernelp", fq(pb["p"])); put("perm", mn(S4["pooled"]["a_ndf"]["pooled"]))
put("bhatt", mn(S4["pooled"]["a_bhatt"]["pooled"])); put("bhattp", fq(S4["pooled"]["a_bhatt"]["p"])); put("bhattI", f"{100*S4['pooled']['a_bhatt']['I2']:.0f}")
put("tunauc", mn(S7["pooled"]["tuned_H1"]["pooled"], 4)); put("tunndf", mn(pt["pooled"])); put("tunndfci", c2(pt["ci_low"], pt["ci_high"])); put("tunndfp", fq(pt["p"]))
put("tunknn", mn(S7["pooled"]["tuned_H2_knn"]["pooled"])); put("tunknnp", fq(S7["pooled"]["tuned_H2_knn"]["p"]))
nonlr = sum(n for d in S7["tuned"].values() for k, n in d["s_star_counts"].items() if k != "lr"); put("tunnonlr", str(nonlr))
put("tunfolds", str(sum(d["n_folds"] for d in S7["tuned"].values())))
put("nlmean", mn(T["nonlinear_S*_d"]["mean"])); put("nlci", c2(*T["nonlinear_S*_d"]["ci"])); put("nlp", fq(T["nonlinear_S*_d"]["p"]))
put("lmean", mn(T["linear_S*_d"]["mean"])); put("nlcoefp", fq(T["model_nonlin_coef"]["p"]))
for d, k in [("dr_debrecen", "DR"), ("ilpd", "ILPD"), ("cdc_diabetes", "CDC")]:
    r1, r2, ph, praw = primary(d); put("p" + k, mn(r2["mean_diff"])); put("p" + k + "holm", fq(ph)); put("p" + k + "knn", mn(allds[d]["H2_knn"]["mean_diff"]))
X = V["exclusions_total"]; put("nsamp", f"{X['sampled']:,}".replace(",", "{,}")); put("ninc", f"{X['included']:,}".replace(",", "{,}"))
put("exclhe", str(X["excl_e3"])); put("excls", str(X["excl_s"])); put("Ghe", f"{X['median_G_e3']:.2f}"); put("Gs", f"{X['median_G_s']:.2f}")
put("medhe", f"{X['median_ndf_e3']:.2f}"); put("meds", f"{X['median_ndf_s']:.2f}"); put("sharelower", f"{100*X['share_e3_lower']:.0f}")
put("shareDR", f"{100*V['exclusions']['dr_debrecen']['share_e3_lower']:.0f}")
sh = V["shrinkage_similarG"]; put("simG", mn(sh["pooled"]["pooled"])); put("simGp", fq(sh["pooled"]["p"])); put("simGn", f"{sh['n_patients']:,}".replace(",", "{,}"))
sim = {(r["truth"], r["rho"], r["op"]): r for r in E["study3_summary"]}
for t in ["L", "N"]:
    for rho, rk in [(0.0, "zero"), (0.5, "half"), (0.9, "nine")]:
        put(f"sim{t}{rk}", mn(sim[(t, rho, 'marg')]["mean_dndf"])); put(f"sim{t}{rk}knn", mn(sim[(t, rho, 'knn')]["mean_dndf"]))
rg = E["study3_regression"]["rho:marg"]; put("simint", f"{rg['coef']:+.2f}".replace("-", "$-$")); put("simintse", f"{rg['se']:.3f}"); put("simintp", fq(rg["p"]))
for k in ["R", "PL", "d_feat", "d_grp"]:
    put("pool" + k.replace("_", ""), mn(q[k]["pooled"])); put("pool" + k.replace("_", "") + "p", fq(q[k]["p"]))
put("Rmerged", mn(q["R_merged"]["pooled"])); put("Rmergedp", fq(q["R_merged"]["p"]))
for d, k in [("dr_debrecen", "DR"), ("ilpd", "ILPD"), ("wdbc", "WDBC"), ("ctg", "CTG")]:
    v = S6["per_dataset"][d]
    for a in ["R", "PL", "d_feat", "d_grp"]:
        put(a.replace("_", "") + k, mn(v[a]["mean_diff"])); put(a.replace("_", "") + k + "p", fq(v[a]["p"]))
put("NAall", mn(q["d_na"]["pooled"])); put("NAallp", fq(q["d_na"]["p"]))
if C:
    PP = C["pooled_pairs"]
    for key, k in [("e3-rf", "HErf"), ("e3-xgb", "HExgb"), ("e1-s_star", "SVs"), ("e2-s_star", "STs"), ("rf-lr", "RFlr"), ("xgb-lr", "XGBlr"), ("e3-s_star", "HEsr")]:
        put(k, mn(PP[key]["pooled"])); put(k + "p", fq(PP[key]["p"])); put(k + "hk", c2(*PP[key]["hksj_ci"])); put(k + "knn", mn(PP[key + "_knn"]["pooled"]))
    DK = C["per_model_mean_DexplK"]; D1 = C["per_model_mean_Dexpl1"]; DR = C["per_model_mean_DrandK"]
    put("DoneHE", f"{100*D1['e3']:.0f}"); put("DoneS", f"{100*D1['s_star']:.0f}"); put("DKHE", f"{100*DK['e3']:.0f}"); put("DKS", f"{100*DK['s_star']:.0f}")
    put("DrandK", f"{100*DR['s_star']:.0f}")
    for m, k in [("lr", "LR"), ("svm", "SVM"), ("knn", "KNN"), ("rf", "RF"), ("xgb", "XGB"), ("e1", "SV"), ("e2", "ST"), ("e3", "HE")]:
        put("med" + k, f"{C['per_model_median_ndf'][m]:.2f}")
    for key, k in [("svm-lr", "SVMlr"), ("knn-lr", "KNNlr"), ("e1-lr", "SVlr"), ("e2-lr", "STlr"), ("e3-lr", "HElr")]:
        put(k, mn(PP[key]["pooled"])); put(k + "p", fq(PP[key]["p"]))
    put("HErfknnp", fq(PP["e3-rf_knn"]["p"])); put("STsknnp", fq(PP["e2-s_star_knn"]["p"]))
    if "treeshap" in C:
        ts = C["treeshap"]; put("treerho", f"{min(v['median_rho'] for v in ts.values()):.2f}"); put("treediff", f"{max(v['mean_abs_ndf_diff'] for v in ts.values()):.2f}")
    if "mc_error" in C:
        mc = C["mc_error"]; put("mcshap", f"{max(v['median_sd_shap_seed'] for v in mc.values()):.3f}"); put("mcorder", f"{max(v['median_sd_order_seed'] for v in mc.values()):.3f}")
TM = V2["tuned_mixed"]
for k, lab in [("nonlinear", "nl"), ("linear", "l")]:
    put(lab + "mix", mn(TM[k + "_d"]["est"])); put(lab + "mixci", c2(*TM[k + "_d"]["ci"])); put(lab + "mixp", fq(TM[k + "_d"]["p"]))
    put(lab + "mixk", mn(TM[k + "_dk"]["est"])); put(lab + "mixkp", fq(TM[k + "_dk"]["p"]))
TA = V2["tuned_mixed_all"]; put("tunallmix", mn(TA["d"]["est"])); put("tunallmixk", mn(TA["dk"]["est"])); put("tunallmixkp", fq(TA["dk"]["p"]))
put("nlfoldsall", str(TM["folds_nonlin_total"])); put("nlfoldsused", str(TM["nonlinear_d"]["n_folds"]))
for t, k in [("0.02", "two"), ("0.1", "ten")]:
    r = V2["threshold"][t]; put("thr" + k, mn(r["marg"]["pooled"]["pooled"])); put("thr" + k + "p", fq(r["marg"]["pooled"]["p"])); put("thr" + k + "hk", c2(*r["marg"]["hksj_ci"])); put("thr" + k + "n", str(r["included"]))
DSH = V2["deletion_shares"]
for m, k in [("e3", "HE"), ("s_star", "S"), ("lr", "LR")]:
    for q in ["top1", "rand1", "topK", "randK"]:
        put(q.replace("1", "one") + k, f"{100*DSH[m][q]:.0f}")
put("overLR", f"{100*DSH['lr']['over100_topK']:.0f}"); put("overHE", f"{100*DSH['e3']['over100_topK']:.0f}")
put("FCDR", mn(S4["per_dataset"]["dr_debrecen"]["a_bhatt"]["mean_diff"])); put("FCDRp", fq(S4["per_dataset"]["dr_debrecen"]["a_bhatt"]["p"]))
T2 = SV["auc_tost"]; TP = SV["auc_tost_pooled"]
put("eqn", str(len(SV["auc_equivalent_datasets"])))
put("eqlist", ", ".join({"wdbc": "WDBC", "dr_debrecen": "DR Debrecen", "cleveland": "Cleveland", "ilpd": "ILPD", "parkinsons": "Parkinsons", "mammographic": "Mammographic", "ctg": "CTG", "cdc_diabetes": "CDC"}[d] for d in SV["auc_equivalent_datasets"]))
put("aucnine", c4h(*TP["hksj_ci90"])); put("auctostp", fq(TP["hksj_p_tost"]))
for d, k in [("dr_debrecen", "DR"), ("ilpd", "ILPD"), ("parkinsons", "PARK")]:
    put("tost" + k, c4h(*T2[d]["ci90"]))
A = SV["adf_all_patients"]
put("adf", f"{A['marg']['pooled']['pooled']:+.3f}".replace("-", "$-$")); put("adfp", fq(A["marg"]["pooled"]["p"])); put("adfhk", c3(*A["marg"]["hksj_ci"]))
put("adfk", f"{A['knn']['pooled']['pooled']:+.3f}".replace("-", "$-$")); put("adfkhk", c3(*A["knn"]["hksj_ci"]))
put("adfn", f"{A['n_defined']:,}".replace(",", "{,}")); put("adfneg", str(sum(1 for d in DS if A["marg"]["per_dataset"][d]["mean_diff"] < 0 and A["marg"]["per_dataset"][d]["p"] < 0.05)))
put("adfCTG", f"{A['marg']['per_dataset']['ctg']['mean_diff']:+.3f}".replace("-", "$-$"))
put("adfmedhe", f"{A['median_adf_e3']:.2f}"); put("adfmeds", f"{A['median_adf_s']:.2f}")
put("inclmin", f"{min(e['pct_included'] for e in EX.values()):.0f}"); put("inclmax", f"{max(e['pct_included'] for e in EX.values()):.0f}")
Path("numbers.tex").write_text("\n".join(f"\\newcommand{{\\N{k}}}{{{v}}}" for k, v in Mx.items()))
print(len(Mx), "macros")
