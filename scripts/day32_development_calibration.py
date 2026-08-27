#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,itertools,json,math,re,subprocess,time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/day32_development_calibration.json"
PROMPT=ROOT/"data/protocol/day32_scoring_prompt_contract.json"
CONTRACT=ROOT/"data/protocol/day32_calibration_operational_contract.json"
D31CFG=ROOT/"configs/day31_root_cause_baseline.json"
D31PROMPT=ROOT/"data/protocol/day31_baseline_prompt_contract.json"
D31SCRIPT=ROOT/"scripts/day31_root_cause_baseline.py"
D31PRED=ROOT/"data/eval/day31_development_baseline_predictions.jsonl"
D31MET=ROOT/"data/eval/day31_development_baseline_metrics.json"
D31REC=ROOT/"data/protocol/day31_baseline_freeze_receipt.json"
D30REC=ROOT/"data/protocol/day30_split_freeze_receipt.json"
SPLIT=ROOT/"data/splits/day30_episode_split.jsonl"
PAIR=ROOT/"data/splits/day30_pair_group_split.json"
GT=ROOT/"data/annotations/day29_ground_truth_records.jsonl"
SRC=ROOT/"data/protocol/day28_registered_source_manifest.csv"
RAWCFG=ROOT/"configs/day28_raw_audit.yaml"

SCORES=ROOT/"data/eval/day32_development_scoring_predictions.jsonl"
SEARCH=ROOT/"data/eval/day32_calibration_search.json"
CALPRED=ROOT/"data/eval/day32_development_calibrated_predictions.jsonl"
METRICS=ROOT/"data/eval/day32_development_calibrated_metrics.json"
FROZEN=ROOT/"data/protocol/day32_frozen_diagnostic_config.json"
RECEIPT=ROOT/"data/protocol/day32_calibration_freeze_receipt.json"
WORK=ROOT/"reports/day32_calibration_work"
PARTIAL=WORK/"partial_scoring_predictions.jsonl"

D31_FINAL="eb423b152555533e577315667e85067dd47069b7"
D31_SCRIPT_BLOB="afd55294495b6cb552a3484662072fb8d46c3ef9"
HASHES={
 D31PRED:"6c323f1432723e897306f20a8a0804c713b7f7b8c8d93a48b99492f9c394d768",
 D31MET:"2e57c70b91eda0cc385be63a218d0a4802ca3a3800953749919330ec098437cd",
 D31REC:"8f5a2e5f714a175019835ae5aedcb8d0ca6b615f5c715e1a99f3ec97e98870ac",
 D31CFG:"eef3ed506ce434c9df2aafd236d4c848cb640bbbce7646fa6d143ac4798eb63f",
 D31PROMPT:"faee60d40b710005a265ef7c657a2b19921c8b40c41ddda8c3d69d4916dbd79f",
 D30REC:"1523fd3fdfea33d2c5818ddee92c5fc161d73baa6acab5b56bf0c9c385f1465d",
 SPLIT:"0b37a499904dcf8568ac39a9641097f7d73c952a01a79f00cbcda2b3b7793312",
 PAIR:"d43937c60279bbddc71ff078334dda40c900ff3dabe53cc06164773f3f77f5d2",
 GT:"e03ec1ab443e4fb4dab606e16fbae8439411d7c3acbcf5f078ed5a0660d389bf",
}
CAUSE=("target_offset_or_perception","gripper_close_timing","trajectory_execution_deviation")
SUB=CAUSE+("clean_success",)
ALL=CAUSE+("insufficient_evidence","clean_success")
TOOLING=(
 "configs/day32_development_calibration.json",
 "data/protocol/day32_scoring_prompt_contract.json",
 "data/protocol/day32_calibration_operational_contract.json",
 "docs/day32_development_only_calibration.md",
 "scripts/day32_development_calibration.py",
 "tests/test_day32_development_calibration.py",
)

def sh(*a): return subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""): h.update(c)
 return h.hexdigest()
def rj(p): return json.loads(p.read_text(encoding="utf-8"))
def rjl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def wj(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def wjl(p,rows): p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in rows),encoding="utf-8")
def app(p,x): p.parent.mkdir(parents=True,exist_ok=True);f=p.open("a",encoding="utf-8");f.write(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n");f.close()

def tooling_commit():
 for rel in TOOLING: sh("rev-parse",f"HEAD:{rel}")
 dirty=subprocess.check_output(["git","status","--porcelain","--",*TOOLING],cwd=ROOT,text=True).strip()
 if dirty: raise RuntimeError("Day32 tooling dirty:\n"+dirty)
 return sh("rev-parse","HEAD")

def d31mod():
 spec=importlib.util.spec_from_file_location("d31",D31SCRIPT)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def env(require_raw=False):
 if sh("branch","--show-current")!="master": raise RuntimeError("must run on master")
 if subprocess.run(["git","merge-base","--is-ancestor",D31_FINAL,"HEAD"],cwd=ROOT).returncode: raise RuntimeError("Day31 final not ancestor")
 if sh("rev-parse","HEAD:scripts/day31_root_cause_baseline.py")!=D31_SCRIPT_BLOB: raise RuntimeError("Day31 script changed")
 for p,e in HASHES.items():
  if sha(p)!=e: raise RuntimeError(f"frozen SHA mismatch {p}")
 cfg,prompt,contract=rj(CFG),rj(PROMPT),rj(CONTRACT)
 if contract["scoring_prompt_sha256"]!=sha(PROMPT): raise RuntimeError("Day32 prompt SHA mismatch")
 rows=rjl(SPLIT);dev=[r for r in rows if r["split"]=="development"];held=[r for r in rows if r["split"]=="held_out"]
 if len(dev)!=60 or len(held)!=30: raise RuntimeError("split count mismatch")
 did=[r["episode_id"] for r in dev];hid={r["episode_id"] for r in held}
 dg=sorted({r["pair_group_id"] for r in dev});hg={r["pair_group_id"] for r in held}
 if len(dg)!=10 or set(dg)&hg or set(did)&hid: raise RuntimeError("split leakage")
 d31={r["episode_id"]:r for r in rjl(D31PRED)}
 if set(d31)!=set(did): raise RuntimeError("Day31 population drift")
 raw=None
 if require_raw:
  import yaml
  rc=yaml.safe_load(RAWCFG.read_text(encoding="utf-8"));raw=Path(rc["raw_source"]["compatibility_wsl_root"])
  if not raw.is_dir(): raise RuntimeError(f"raw root unavailable {raw}")
 return dict(cfg=cfg,prompt=prompt,dev=dev,held=held,did=did,hid=hid,dg=dg,e2g={r["episode_id"]:r["pair_group_id"] for r in dev},d31=d31,raw=raw)

def internal(groups,seed,fitn):
 ranked=sorted(((g,hashlib.sha256(f"{seed}|{g}".encode()).hexdigest()) for g in groups),key=lambda x:x[1])
 return [x[0] for x in ranked[:fitn]],[x[0] for x in ranked[fitn:]],ranked

def manifest():
 with SRC.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
 return {r["episode_id"]:r for r in rows}

def devgt(did,hid):
 out={};pat=re.compile(r'"episode_id":"([^"]+)"')
 for line in GT.read_text(encoding="utf-8").splitlines():
  if not line.strip(): continue
  m=pat.search(line)
  if not m: raise RuntimeError("GT episode_id not found")
  eid=m.group(1)
  if eid in hid: continue
  if eid not in set(did): raise RuntimeError("GT outside frozen population")
  r=json.loads(line)
  out[eid]={k:r[k] for k in ("physical_cause_gt","diagnostic_decision_gt","evidence_answerability_gt","task_success")}
 if len(out)!=60: raise RuntimeError("development GT count mismatch")
 return out

def stripf(s):
 s=s.strip()
 if s.startswith("```"): s=re.sub(r"^```(?:json)?\s*","",s,flags=re.I);s=re.sub(r"\s*```$","",s)
 return s.strip()

def parse(resp,selected):
 try:
  s=stripf(resp)
  try: p=json.loads(s)
  except json.JSONDecodeError:
   m=re.search(r"\{.*\}",s,re.S)
   if not m: raise
   p=json.loads(m.group(0))
  sc=p["scores"]
  if set(sc)!=set(SUB): raise ValueError("scores keys")
  raw={k:float(sc[k]) for k in SUB}
  if any((not math.isfinite(v) or v<0 or v>100) for v in raw.values()) or sum(raw.values())<=0: raise ValueError("scores range")
  norm={k:v/sum(raw.values()) for k,v in raw.items()}
  top=max(SUB,key=lambda k:(norm[k],-SUB.index(k)))
  decl=p["top_substantive_decision"]
  if decl not in SUB: raise ValueError("top label")
  conf=float(p["confidence"])
  if not 0<=conf<=1: raise ValueError("confidence")
  ev=[]
  for x in p["evidence_frame_indices"]:
   if not isinstance(x,int) or isinstance(x,bool) or x not in set(selected): raise ValueError("evidence frame")
   if x not in ev: ev.append(x)
  rat=p["rationale"]
  if not isinstance(rat,str) or not rat.strip(): raise ValueError("rationale")
  return dict(parse_ok=True,parse_error=None,raw_scores=raw,normalized_scores=norm,declared_top_substantive_decision=decl,derived_top_substantive_decision=top,declared_top_matches_scores=(decl==top),model_confidence=conf,evidence_frame_indices=ev,rationale=rat.strip())
 except Exception as ex:
  return dict(parse_ok=False,parse_error=f"{type(ex).__name__}: {ex}",raw_scores={k:0.0 for k in SUB},normalized_scores={k:.25 for k in SUB},declared_top_substantive_decision=None,derived_top_substantive_decision=None,declared_top_matches_scores=False,model_confidence=0.0,evidence_frame_indices=[],rationale="parse_failure")

def messages(sheets,state,selected,prompt,cfg):
 content=[{"type":"image","image":p.resolve().as_uri(),"min_pixels":cfg["model"]["min_pixels"],"max_pixels":cfg["model"]["max_pixels"]} for p in sheets]
 outcomes="\n".join(f"- {k}: {v}" for k,v in prompt["substantive_outcomes"].items())
 content.append({"type":"text","text":prompt["user_instruction_template"]+"\n\nFROZEN OUTCOMES\n"+outcomes+"\n\nSUPPLIED FRAMES\n"+json.dumps(selected)+"\n\n"+state+"\n\nOUTPUT JSON SCHEMA\n"+json.dumps(prompt["required_output_schema"],ensure_ascii=False,indent=2)})
 return [{"role":"system","content":prompt["system_prompt"]},{"role":"user","content":content}]

def validate_scores(rows,did,hid,complete=True):
 err=[];ids=[r.get("episode_id") for r in rows]
 if len(ids)!=len(set(ids)): err.append("duplicate IDs")
 if set(ids)-set(did): err.append("non-development score rows")
 if set(ids)&hid: err.append("held-out score rows")
 if complete and (len(rows)!=60 or set(ids)!=set(did)): err.append("incomplete score population")
 forbidden={"pair_group_id","physical_cause_gt","diagnostic_decision_gt","evidence_answerability_gt","task_success","intervention_type","human_review_notes"}
 for r in rows:
  if forbidden&set(r): err.append(f"{r.get('episode_id')}: forbidden fields")
  if r.get("schema_version")!="evidencemm_day32_scoring_prediction_v1": err.append("bad score schema")
  if r.get("split")!="development": err.append("bad split")
  if r.get("score_prompt_sha256")!=sha(PROMPT): err.append("score prompt SHA drift")
  ns=r.get("normalized_scores",{})
  if set(ns)!=set(SUB) or abs(sum(float(v) for v in ns.values())-1)>1e-8: err.append("bad normalized scores")
 return err

def softmax(x):
 m=max(x.values());e={k:math.exp(v-m) for k,v in x.items()};s=sum(e.values());return {k:v/s for k,v in e.items()}

def cprobs(ns,bias):
 return softmax({k:math.log(max(float(ns[k]),1e-9))+float(bias[k]) for k in SUB})

def decision(row,cand):
 if not row["parse_ok"]: return dict(diagnostic_decision="insufficient_evidence",top_substantive_decision=None,top_probability=0.0,second_probability=0.0,margin=0.0,calibrated_probabilities={k:.25 for k in SUB},abstention_reason="score_parse_failure")
 p=cprobs(row["normalized_scores"],cand["biases"]);rank=sorted(SUB,key=lambda k:(-p[k],SUB.index(k)));a,b=rank[:2];margin=p[a]-p[b];reasons=[]
 if p[a]<cand["confidence_threshold"]: reasons.append("top_probability_below_threshold")
 if margin<cand["margin_threshold"]: reasons.append("margin_below_threshold")
 return dict(diagnostic_decision=("insufficient_evidence" if reasons else a),top_substantive_decision=a,top_probability=p[a],second_probability=p[b],margin=margin,calibrated_probabilities=p,abstention_reason=("+".join(reasons) if reasons else None))

def div(a,b): return None if b==0 else a/b
def f1s(y,p,labels):
 out={}
 for lab in labels:
  tp=sum(a==lab and b==lab for a,b in zip(y,p));fp=sum(a!=lab and b==lab for a,b in zip(y,p));fn=sum(a==lab and b!=lab for a,b in zip(y,p));sup=sum(a==lab for a in y)
  pr=div(tp,tp+fp);rc=div(tp,tp+fn);f=0.0 if pr is None or rc is None or pr+rc==0 else 2*pr*rc/(pr+rc)
  out[lab]=dict(support=sup,tp=tp,fp=fp,fn=fn,precision=pr,recall=rc,f1=f)
 return out
def macro(x): return sum(v["f1"] for v in x.values())/len(x) if x else 0.0

def metrics(ids,gt,dec):
 ans=[i for i in ids if gt[i]["task_success"] is False and gt[i]["evidence_answerability_gt"]=="answerable"]
 fail=[i for i in ids if gt[i]["task_success"] is False];clean=[i for i in ids if gt[i]["task_success"] is True]
 insuff=[i for i in fail if gt[i]["evidence_answerability_gt"]=="insufficient_evidence"]
 subids=[i for i in ids if gt[i]["diagnostic_decision_gt"] in SUB]
 sf=f1s([gt[i]["diagnostic_decision_gt"] for i in subids],[dec[i] for i in subids],SUB)
 tf=f1s([gt[i]["physical_cause_gt"] for i in ans],[dec[i] for i in ans],CAUSE)
 ff=f1s([gt[i]["diagnostic_decision_gt"] for i in fail],[dec[i] for i in fail],CAUSE+("insufficient_evidence",))
 abst=sum((gt[i]["evidence_answerability_gt"]=="insufficient_evidence" and dec[i]=="insufficient_evidence") or (gt[i]["evidence_answerability_gt"]=="answerable" and dec[i]!="insufficient_evidence") for i in fail)
 return {
  "episode_count":len(ids),"substantive_four_class_macro_f1":macro(sf),"answerable_three_class_macro_f1":macro(tf),"failed_case_four_way_diagnostic_macro_f1":macro(ff),
  "abstention_accuracy":div(abst,len(fail)),"false_answer_rate":div(sum(dec[i]!="insufficient_evidence" for i in insuff),len(insuff)),
  "false_abstention_rate":div(sum(dec[i]=="insufficient_evidence" for i in ans),len(ans)),
  "clean_control_false_positive_cause_rate":div(sum(dec[i] in CAUSE for i in clean),len(clean)),
  "clean_control_accuracy":div(sum(dec[i]=="clean_success" for i in clean),len(clean)),
  "development_decision_accuracy":div(sum(dec[i]==gt[i]["diagnostic_decision_gt"] for i in ids),len(ids)),
  "gt_support":{"answerable_failure":len(ans),"failed":len(fail),"clean":len(clean),"insufficient_evidence_failure":len(insuff),"substantive":len(subids)}
 }

def grid(cfg):
 vals=cfg["calibration_grid"]["cause_log_bias_values"];cs=cfg["calibration_grid"]["confidence_threshold_values"];ms=cfg["calibration_grid"]["margin_threshold_values"];out=[]
 for bt,bg,br in itertools.product(vals,repeat=3):
  b={"target_offset_or_perception":float(bt),"gripper_close_timing":float(bg),"trajectory_execution_deviation":float(br),"clean_success":0.0}
  for c in cs:
   for m in ms: out.append({"candidate_id":f"bT{bt:+.1f}_bG{bg:+.1f}_bR{br:+.1f}_c{c:.2f}_m{m:.2f}","biases":b,"confidence_threshold":float(c),"margin_threshold":float(m)})
 if len(out)!=cfg["calibration_grid"]["candidate_count"]: raise RuntimeError("grid size mismatch")
 return out

def key(c,m):
 l1=sum(abs(c["biases"][x]) for x in CAUSE);fa=m["false_abstention_rate"] if m["false_abstention_rate"] is not None else 1.0
 return (m["substantive_four_class_macro_f1"],m["answerable_three_class_macro_f1"],m["clean_control_accuracy"] or -1,m["development_decision_accuracy"] or -1,-fa,-l1,-c["confidence_threshold"],-c["margin_threshold"],c["candidate_id"])

def fit(rows,gt,fitids,validids,allids,cfg):
 by={r["episode_id"]:r for r in rows};best=None;ranked=[]
 for c in grid(cfg):
  dec={i:decision(by[i],c)["diagnostic_decision"] for i in fitids};m=metrics(fitids,gt,dec);ranked.append((key(c,m),c,m))
 ranked.sort(key=lambda x:x[0],reverse=True);_,c,fm=ranked[0]
 vm=metrics(validids,gt,{i:decision(by[i],c)["diagnostic_decision"] for i in validids})
 am=metrics(allids,gt,{i:decision(by[i],c)["diagnostic_decision"] for i in allids})
 return c,fm,vm,am,[{"candidate":x[1],"fit_summary":{k:x[2][k] for k in ("substantive_four_class_macro_f1","answerable_three_class_macro_f1","clean_control_accuracy","development_decision_accuracy","false_abstention_rate")}} for x in ranked[:20]]

def calibrated_rows(rows,c):
 out=[]
 for r in rows:
  d=decision(r,c);out.append({"schema_version":"evidencemm_day32_calibrated_prediction_v1","episode_id":r["episode_id"],"split":"development","model_name":r["model_name"],"score_prompt_sha256":r["score_prompt_sha256"],"selected_frame_indices":r["selected_frame_indices"],"day31_evidence_input_sha256":r["day31_evidence_input_sha256"],"score_parse_ok":r["parse_ok"],"raw_scores":r["raw_scores"],"normalized_scores":r["normalized_scores"],**d,"evidence_frame_indices":r["evidence_frame_indices"],"rationale":r["rationale"]})
 return out

def preflight():
 tooling_commit();e=env(True);s=e["cfg"]["internal_calibration_split"];fitg,valg,_=internal(e["dg"],s["seed"],s["fit_pair_group_count"])
 if any(p.exists() for p in (SCORES,SEARCH,CALPRED,METRICS,FROZEN,RECEIPT)): raise RuntimeError("Day32 output already exists")
 print("===== DAY32 CALIBRATION PREFLIGHT =====");print("head =",sh("rev-parse","HEAD"));print("development_episode_count = 60");print("held_out_episode_count = 30");print("internal_fit_pair_group_count =",len(fitg));print("internal_validation_pair_group_count =",len(valg));print("candidate_count =",e["cfg"]["calibration_grid"]["candidate_count"]);print("held_out_inference_allowed = false");print("held_out_gt_json_parsing = false");print("day30_receipt_parsed = false");print("DAY32 CALIBRATION PREFLIGHT: PASS")

def run_scores():
 tc=tooling_commit();e=env(True)
 if SCORES.exists():
  er=validate_scores(rjl(SCORES),e["did"],e["hid"],True)
  if er: raise RuntimeError(er)
  print("DAY32 score predictions already complete: PASS");return
 partial=rjl(PARTIAL) if PARTIAL.exists() else [];er=validate_scores(partial,e["did"],e["hid"],False)
 if er: raise RuntimeError(er)
 done={r["episode_id"] for r in partial};remain=[i for i in e["did"] if i not in done]
 print("===== DAY32 DEVELOPMENT SCORING RUN =====");print("tooling_commit =",tc);print("completed_before_run =",len(done));print("remaining =",len(remain));print("held_out_inference_count = 0")
 if remain:
  import torch
  from qwen_vl_utils import process_vision_info
  from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
  from evidencemm.state_action_selection import load_state_action_samples
  d31=d31mod();src=manifest();d31cfg=rj(D31CFG);mc=e["cfg"]["model"];name=mc["model_name"]
  print("loading model:",name);model=Qwen3VLForConditionalGeneration.from_pretrained(name,dtype="auto",device_map="auto",attn_implementation=mc["attn_implementation"],local_files_only=True);model.eval();processor=AutoProcessor.from_pretrained(name,local_files_only=True)
  for n,eid in enumerate(remain,1):
   print(f"[{len(done)+n:02d}/60] episode={eid}",flush=True);r31=e["d31"][eid];sel=[int(x) for x in r31["selected_frame_indices"]];ep=e["raw"]/src[eid]["raw_episode_relpath"];sp=ep/"samples.csv"
   if sha(sp)!=src[eid]["samples_sha256"] or sha(sp)!=r31["samples_sha256"]: raise RuntimeError(f"{eid}: samples SHA")
   samples=load_state_action_samples(sp);state=d31.state_action_text(samples,sel);sheets,imsha=d31.build_contact_sheets(episode_dir=ep,selected=sel,config=d31cfg,output_dir=WORK/"inputs"/eid)
   if imsha!=r31["raw_selected_image_hashes_sha256"]: raise RuntimeError(f"{eid}: image evidence drift")
   fp=d31.evidence_fingerprint(episode_id=eid,selected=sel,state_text=state,raw_image_hash_sha256=imsha,samples_sha256=sha(sp),prompt_sha256=HASHES[D31PROMPT])
   if fp!=r31["evidence_input_sha256"]: raise RuntimeError(f"{eid}: Day31 evidence fingerprint drift")
   msg=messages(sheets,state,sel,e["prompt"],e["cfg"]);txt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);images,videos,vkwargs=process_vision_info(msg,image_patch_size=mc["image_patch_size"],return_video_kwargs=True,return_video_metadata=True)
   if videos is not None: videos,vmeta=zip(*videos);videos=list(videos);vmeta=list(vmeta)
   else: vmeta=None
   inputs=processor(text=txt,images=images,videos=videos,video_metadata=vmeta,return_tensors="pt",do_resize=False,**vkwargs).to(model.device)
   if torch.cuda.is_available(): torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
   t=time.perf_counter()
   with torch.inference_mode(): ids=model.generate(**inputs,max_new_tokens=mc["max_new_tokens"],do_sample=False)
   if torch.cuda.is_available(): torch.cuda.synchronize()
   latency=time.perf_counter()-t;trim=[o[len(i):] for i,o in zip(inputs.input_ids,ids)];resp=processor.batch_decode(trim,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();peak=(torch.cuda.max_memory_allocated()/1024/1024 if torch.cuda.is_available() else None);parsed=parse(resp,sel)
   app(PARTIAL,{"schema_version":"evidencemm_day32_scoring_prediction_v1","episode_id":eid,"split":"development","model_name":name,"scoring_mode":"four_substantive_outcome_support_scores","score_prompt_sha256":sha(PROMPT),"selected_frame_indices":sel,"day31_evidence_input_sha256":fp,"raw_selected_image_hashes_sha256":imsha,"samples_sha256":sha(sp),"response_raw":resp,**parsed,"latency_sec":latency,"peak_gpu_memory_mb":peak})
   del inputs,ids,trim
   if torch.cuda.is_available(): torch.cuda.empty_cache()
 rows=rjl(PARTIAL);er=validate_scores(rows,e["did"],e["hid"],True)
 if er: raise RuntimeError(er)
 by={r["episode_id"]:r for r in rows};wjl(SCORES,[by[i] for i in e["did"]]);print("score_prediction_count = 60");print("parse_ok_count =",sum(r["parse_ok"] for r in rows));print("score_predictions_sha256 =",sha(SCORES));print("DAY32 DEVELOPMENT SCORING: PASS")

def validate():
 e=env();rows=rjl(SCORES);er=validate_scores(rows,e["did"],e["hid"],True);tops=Counter(r["derived_top_substantive_decision"] for r in rows if r["parse_ok"])
 print("===== DAY32 SCORE VALIDATION =====");print("score_prediction_count =",len(rows));print("parse_ok_count =",sum(r["parse_ok"] for r in rows));print("raw_top_counts =",dict(sorted(tops.items())));print("held_out_prediction_count =",sum(r["episode_id"] in e["hid"] for r in rows));print("errors =",er)
 if er: raise SystemExit(1)
 print("DAY32 SCORE VALIDATION: PASS")

def calibrate():
 e=env();rows=rjl(SCORES);er=validate_scores(rows,e["did"],e["hid"],True)
 if er: raise RuntimeError(er)
 gt=devgt(e["did"],e["hid"]);s=e["cfg"]["internal_calibration_split"];fitg,valg,ranking=internal(e["dg"],s["seed"],s["fit_pair_group_count"]);fitids=[i for i in e["did"] if e["e2g"][i] in set(fitg)];validids=[i for i in e["did"] if e["e2g"][i] in set(valg)]
 c,fm,vm,am,top=fit(rows,gt,fitids,validids,e["did"],e["cfg"]);cp=calibrated_rows(rows,c);dec={r["episode_id"]:r["diagnostic_decision"] for r in cp};assert metrics(e["did"],gt,dec)==am
 search={"schema_version":"evidencemm_day32_calibration_search_v1","status":"development_calibration_search_complete","internal_split":{"seed":s["seed"],"fit_pair_groups":fitg,"validation_pair_groups":valg,"ranked_pair_groups":[{"pair_group_id":g,"sha256":h} for g,h in ranking],"fit_episode_count":len(fitids),"validation_episode_count":len(validids),"pair_group_cross_internal_split_count":0},"grid":e["cfg"]["calibration_grid"],"selected_candidate":c,"selection_population":"internal_fit_only","internal_validation_used_for_selection":False,"refit_after_internal_validation":False,"selected_fit_metrics":fm,"selected_internal_validation_metrics":vm,"selected_full_development_metrics":am,"top_20_fit_candidates":top,"held_out_gt_rows_used":0,"held_out_inference_count":0}
 d31m=rj(D31MET)["primary_metrics"];mp={"schema_version":"evidencemm_day32_calibrated_metrics_v1","status":"development_only_calibration_complete","selected_candidate":c,"internal_fit_metrics":fm,"internal_validation_metrics":vm,"full_development_metrics":am,"day31_baseline_primary_metrics":d31m,"day31_to_day32_delta":{"answerable_three_class_macro_f1":am["answerable_three_class_macro_f1"]-d31m["answerable_three_class_macro_f1"],"failed_case_four_way_diagnostic_macro_f1":am["failed_case_four_way_diagnostic_macro_f1"]-d31m["failed_case_four_way_diagnostic_macro_f1"],"false_abstention_rate":None if am["false_abstention_rate"] is None else am["false_abstention_rate"]-d31m["false_abstention_rate"]},"score_parse_ok_count":sum(r["parse_ok"] for r in rows),"final_decision_counts":dict(sorted(Counter(r["diagnostic_decision"] for r in cp).items())),"held_out_prediction_count":0,"held_out_gt_rows_used":0}
 frozen={"schema_version":"evidencemm_day32_frozen_diagnostic_config_v1","status":"frozen_for_single_day33_heldout_evaluation","source_day32_tooling_commit":tooling_commit(),"model_name":e["cfg"]["model"]["model_name"],"model_loading":{"do_sample":False,"local_files_only":True,"attn_implementation":e["cfg"]["model"]["attn_implementation"],"max_new_tokens":e["cfg"]["model"]["max_new_tokens"]},"evidence_contract":{"day31_selected_frame_indices_reused":True,"day31_contact_sheet_convention_reused":True,"day31_state_action_text_reused":True,"frame_count_per_episode":12,"retrieval_used":False,"manual_corpus_used":False},"scoring_prompt_sha256":sha(PROMPT),"score_labels":list(SUB),"score_normalization":"divide_by_sum_then_log","calibration_transform":"additive_class_log_bias_then_softmax","selected_log_biases":c["biases"],"confidence_threshold":c["confidence_threshold"],"margin_threshold":c["margin_threshold"],"parse_failure_decision":"insufficient_evidence","fit_population":{"pair_groups":fitg,"episode_count":len(fitids)},"internal_validation_population":{"pair_groups":valg,"episode_count":len(validids),"used_for_selection":False},"held_out_boundary":{"held_out_inference_count_on_day32":0,"held_out_gt_rows_used_on_day32":0,"held_out_prompt_tuning_allowed":False,"held_out_calibration_refit_allowed":False,"day33_final_evaluation_count_consumed":0}}
 for p,x in ((SEARCH,search),(CALPRED,cp),(METRICS,mp),(FROZEN,frozen)):
  if p.exists():
   old=rjl(p) if p==CALPRED else rj(p)
   if old!=x: raise RuntimeError(f"artifact differs {p}")
  else: wjl(p,x) if p==CALPRED else wj(p,x)
 print("===== DAY32 DEVELOPMENT-ONLY CALIBRATION =====");print("internal_fit_pair_groups =",fitg);print("internal_validation_pair_groups =",valg);print("selected_candidate =",c);print("fit_substantive_four_class_macro_f1 =",fm["substantive_four_class_macro_f1"]);print("validation_substantive_four_class_macro_f1 =",vm["substantive_four_class_macro_f1"]);print("full_dev_answerable_three_class_macro_f1 =",am["answerable_three_class_macro_f1"]);print("full_dev_clean_control_accuracy =",am["clean_control_accuracy"]);print("full_dev_false_abstention_rate =",am["false_abstention_rate"]);print("held_out_prediction_count = 0");print("held_out_gt_rows_used = 0");print("DAY32 DEVELOPMENT-ONLY CALIBRATION: PASS")

def freeze():
 tc=tooling_commit();e=env();validate()
 for p in (SEARCH,CALPRED,METRICS,FROZEN):
  if not p.exists(): raise RuntimeError(f"missing {p}")
 search=rj(SEARCH);mp=rj(METRICS);c=search["selected_candidate"];rows=rjl(SCORES)
 if rjl(CALPRED)!=calibrated_rows(rows,c): raise RuntimeError("calibrated rows drift")
 if RECEIPT.exists(): raise RuntimeError("receipt exists")
 blobs={rel:sh("rev-parse",f"{tc}:{rel}") for rel in TOOLING}
 rec={"schema_version":"evidencemm_day32_calibration_freeze_receipt_v1","status":"development_only_calibration_frozen_day32_complete","tooling_commit":tc,"day31_final_commit":D31_FINAL,"day31_predictions_sha256":HASHES[D31PRED],"day31_metrics_sha256":HASHES[D31MET],"day31_freeze_receipt_sha256":HASHES[D31REC],"day30_episode_split_sha256":HASHES[SPLIT],"day30_pair_group_split_sha256":HASHES[PAIR],"day29_ground_truth_records_sha256":HASHES[GT],"config_sha256":sha(CFG),"scoring_prompt_sha256":sha(PROMPT),"operational_contract_sha256":sha(CONTRACT),"tooling_git_blobs":blobs,"score_predictions_sha256":sha(SCORES),"calibration_search_sha256":sha(SEARCH),"calibrated_predictions_sha256":sha(CALPRED),"calibrated_metrics_sha256":sha(METRICS),"frozen_inference_config_sha256":sha(FROZEN),"development_score_prediction_count":60,"held_out_score_prediction_count":0,"held_out_gt_rows_used":0,"internal_validation_used_for_selection":False,"refit_after_internal_validation":False,"model_weights_modified":False,"retrieval_used":False,"manual_corpus_used":False,"held_out_evaluation_started":False,"held_out_final_evaluation_count_consumed":0,"day33_config_frozen":True,"selected_candidate":c,"internal_fit_metrics":mp["internal_fit_metrics"],"internal_validation_metrics":mp["internal_validation_metrics"],"full_development_metrics":mp["full_development_metrics"]}
 wj(RECEIPT,rec);print("score_predictions_sha256 =",rec["score_predictions_sha256"]);print("calibration_search_sha256 =",rec["calibration_search_sha256"]);print("calibrated_predictions_sha256 =",rec["calibrated_predictions_sha256"]);print("calibrated_metrics_sha256 =",rec["calibrated_metrics_sha256"]);print("frozen_inference_config_sha256 =",rec["frozen_inference_config_sha256"]);print("freeze_receipt_sha256 =",sha(RECEIPT));print("DAY32 CALIBRATION FREEZE RECEIPT: PASS")

def audit():
 e=env();required=(SCORES,SEARCH,CALPRED,METRICS,FROZEN,RECEIPT)
 for p in required:
  if not p.exists(): raise RuntimeError(f"missing {p}")
 errors=[];rows=rjl(SCORES);errors+=validate_scores(rows,e["did"],e["hid"],True);gt=devgt(e["did"],e["hid"]);s=e["cfg"]["internal_calibration_split"];fitg,valg,_=internal(e["dg"],s["seed"],s["fit_pair_group_count"]);fitids=[i for i in e["did"] if e["e2g"][i] in set(fitg)];validids=[i for i in e["did"] if e["e2g"][i] in set(valg)];c,fm,vm,am,_=fit(rows,gt,fitids,validids,e["did"],e["cfg"]);search=rj(SEARCH);mp=rj(METRICS);fr=rj(FROZEN);rec=rj(RECEIPT)
 if search["selected_candidate"]!=c: errors.append("selected candidate drift")
 if search["selected_fit_metrics"]!=fm or search["selected_internal_validation_metrics"]!=vm: errors.append("calibration metrics drift")
 if rjl(CALPRED)!=calibrated_rows(rows,c): errors.append("calibrated predictions drift")
 if mp["full_development_metrics"]!=am: errors.append("full metrics drift")
 if fr["selected_log_biases"]!=c["biases"] or fr["confidence_threshold"]!=c["confidence_threshold"] or fr["margin_threshold"]!=c["margin_threshold"]: errors.append("frozen config drift")
 checks={"status":"development_only_calibration_frozen_day32_complete","score_predictions_sha256":sha(SCORES),"calibration_search_sha256":sha(SEARCH),"calibrated_predictions_sha256":sha(CALPRED),"calibrated_metrics_sha256":sha(METRICS),"frozen_inference_config_sha256":sha(FROZEN),"held_out_score_prediction_count":0,"held_out_gt_rows_used":0,"held_out_final_evaluation_count_consumed":0}
 for k,v in checks.items():
  if rec.get(k)!=v: errors.append(f"receipt {k} mismatch")
 tc=rec.get("tooling_commit")
 try:
  for rel in TOOLING:
   if rec["tooling_git_blobs"][rel]!=sh("rev-parse",f"{tc}:{rel}") or sh("rev-parse",f"HEAD:{rel}")!=sh("rev-parse",f"{tc}:{rel}"): errors.append(f"tooling drift {rel}")
 except Exception as ex: errors.append(str(ex))
 print("===== DAY32 CALIBRATION FREEZE AUDIT =====");print("development_score_prediction_count =",len(rows));print("held_out_score_prediction_count =",sum(r["episode_id"] in e["hid"] for r in rows));print("held_out_gt_rows_used = 0");print("selected_candidate =",c);print("internal_fit_metrics =",fm);print("internal_validation_metrics =",vm);print("full_development_metrics =",am);print("score_predictions_sha256 =",sha(SCORES));print("calibration_search_sha256 =",sha(SEARCH));print("calibrated_predictions_sha256 =",sha(CALPRED));print("calibrated_metrics_sha256 =",sha(METRICS));print("frozen_inference_config_sha256 =",sha(FROZEN));print("freeze_receipt_sha256 =",sha(RECEIPT));print("errors =",errors)
 if errors: raise SystemExit(1)
 print("DAY32 CALIBRATION AUDIT: PASS");print("DAY32: CLOSED / FROZEN")

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
 for x in ("preflight","run-scores","validate-scores","calibrate","freeze","audit"): sp.add_parser(x)
 a=p.parse_args();{"preflight":preflight,"run-scores":run_scores,"validate-scores":validate,"calibrate":calibrate,"freeze":freeze,"audit":audit}[a.cmd]()
if __name__=="__main__": main()
