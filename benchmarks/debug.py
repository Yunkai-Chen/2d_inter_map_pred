# ======= Drop-in: ColabFoldPairedMSA + get_paired_msa =======
import os, io, re, json, time, tarfile, shutil, hashlib, warnings, tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import requests

class ColabFoldPairedMSA:
    """Minimal client for paired MSA via https://api.colabfold.com with filtering."""
    def __init__(self, host_url: str = "https://api.colabfold.com", cache_dir: Optional[str] = None):
        self.host_url = host_url
        self.job_id = None
        self.parsed_entries = None
        self.cache_dir = Path(cache_dir or (Path.home() / ".colabfold_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # tables for uniprot→number enc (used for genomic distance)
        from string import ascii_uppercase
        self.pa = {a: 0 for a in ascii_uppercase}
        for a in ["O", "P", "Q"]: self.pa[a] = 1
        self.ma = [[{} for _ in range(6)],[{} for _ in range(6)]]
        for n,t in enumerate(range(10)):
            for i in [0,1]:
                for j in [0,4]: self.ma[i][j][str(t)] = n
        for n,t in enumerate(list(ascii_uppercase)+list(range(10))):
            for i in [0,1]:
                for j in [1,2]: self.ma[i][j][str(t)] = n
            self.ma[1][3][str(t)] = n
        for n,t in enumerate(ascii_uppercase):
            self.ma[0][3][str(t)] = n
            for i in [0,1]: self.ma[i][5][str(t)] = n
        self.upi_encoding = {str(c):i for i,c in enumerate(list(range(10))+['A','B','C','D','E','F'])}

    def _extract_uniprot_id(self, header: str) -> str:
        pos = header.find("UniRef")
        if pos == -1: return ""
        start = header.find('_', pos)
        if start == -1: return ""
        start += 1
        end = start
        while end < len(header) and header[end] not in ' _\t': end += 1
        uid = header[start:end]
        if len(uid) >= 3 and uid[:3] == "UPI": return uid
        if len(uid) not in [6,10]: return ""
        if not uid[0].isalpha(): return ""
        return uid

    def _uniprot_to_number(self, uniprot_ids: List[str]) -> List[int]:
        numbers = []
        for uni in uniprot_ids:
            if not uni or not uni[0].isalpha(): numbers.append(0); continue
            if uni.startswith("UPI") and len(uni)==13:
                hex_part = uni[3:]; num=0; tot=1
                for u in reversed(hex_part):
                    if str(u) in self.upi_encoding: num += self.upi_encoding[str(u)]*tot; tot*=16
                    else: num=0; break
                numbers.append(num + 10**15); continue
            p = self.pa.get(uni[0],0); tot, num = 1, 0
            if len(uni)==10:
                for n,u in enumerate(reversed(uni[-4:])):
                    if str(u) in self.ma[p][n]: num += self.ma[p][n][str(u)]*tot; tot *= len(self.ma[p][n])
            for n,u in enumerate(reversed(uni[:6])):
                if n < len(self.ma[p]) and str(u) in self.ma[p][n]:
                    num += self.ma[p][n][str(u)]*tot; tot *= len(self.ma[p][n])
            numbers.append(num)
        return numbers

    def _calculate_genomic_distances(self, entry: Dict) -> List[int]:
        distances=[]; nums=entry['uniprot_nums']
        for i in range(1,len(nums)):
            if nums[i-1] and nums[i]: distances.append(abs(nums[i]-nums[i-1]))
            else: distances.append(-1)
        return distances

    def _parse_msa_lines(self, lines: List[str]) -> List[Dict]:
        entries=[]; i=0; is_first=True
        while i < len(lines):
            line = lines[i].rstrip()
            if line.startswith('>'):
                header=line; seq_lines=[]; i+=1
                while i < len(lines) and not lines[i].startswith('>'):
                    if lines[i].strip(): seq_lines.append(lines[i].rstrip())
                    i+=1
                sequence=''.join(seq_lines)
                header_parts=header.split('\t')
                header_clean=header_parts[0].lstrip('>').replace('UniRef100_','')
                uid = self._extract_uniprot_id(header)
                has_uniref = "UniRef" in header
                uniprot_num = 0
                if uid:
                    n = self._uniprot_to_number([uid]); uniprot_num = n[0] if n else 0
                if is_first:
                    coverage=1.0; identity=1.0; evalue=0.0; alnscore=float('inf'); is_first=False
                else:
                    coverage=identity=evalue=alnscore=None
                    if len(header_parts) >= 10:
                        try:
                            alnscore=float(header_parts[1]); identity=float(header_parts[2]); evalue=float(header_parts[3])
                            q_start=int(header_parts[4]); q_end=int(header_parts[5]); q_len=int(header_parts[6])
                            coverage=(q_end - q_start + 1)/q_len
                        except: pass
                    coverage = coverage if coverage is not None else 0.0
                    identity = identity if identity is not None else 0.0
                    evalue = evalue if evalue is not None else float('inf')
                    alnscore = alnscore if alnscore is not None else 0.0
                entries.append({
                    'header': header_clean, 'sequence': sequence,
                    'coverage': coverage, 'identity': identity,
                    'evalue': evalue, 'alnscore': alnscore,
                    'uid': uid, 'uniprot_num': uniprot_num, 'has_uniref': has_uniref
                })
            else:
                i+=1
        return entries

    def _parse_paired_a3m(self, a3m_path: str) -> List[Dict]:
        raw_msas={}; update_M=True; M=None
        with open(a3m_path,'r',errors='ignore') as f:
            for line in f:
                if "\x00" in line: line=line.replace("\x00",""); update_M=True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip().split('_')[-1]); update_M=False
                    if M not in raw_msas: raw_msas[M]=[]
                if M is not None: raw_msas[M].append(line.rstrip())
        parsed_msas={sid:self._parse_msa_lines(lines) for sid,lines in raw_msas.items()}
        seq_ids = sorted(parsed_msas.keys())
        min_entries = min(len(parsed_msas[s]) for s in seq_ids)
        stitched=[]
        for i in range(min_entries):
            headers=[]; sequences=[]; coverages=[]; identities=[]; evalues=[]; alnscores=[]; uids=[]; uniprot_nums=[]; has_uniref=True
            for sid in seq_ids:
                e = parsed_msas[sid][i]
                headers.append(e['header']); sequences.append(e['sequence'])
                coverages.append(e['coverage']); identities.append(e['identity'])
                evalues.append(e['evalue']); alnscores.append(e['alnscore'])
                uids.append(e['uid']); uniprot_nums.append(e['uniprot_num'])
                has_uniref = has_uniref and e['has_uniref']
            stitched.append({
                'headers': headers, 'sequences': sequences,
                'coverages': coverages, 'identities': identities,
                'evalues': evalues, 'alnscores': alnscores,
                'uids': uids, 'uniprot_nums': uniprot_nums,
                'has_uniref': has_uniref, 'is_query': (i==0)
            })
        return stitched

    def _get_cache_key(self, sequences: List[str], genomic_distance: Optional[int], prefix: Optional[str]) -> str:
        cache_data={'sequences':sequences,'genomic_distance':genomic_distance,'prefix':prefix,'host_url':self.host_url}
        cache_hash = hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()[:16]
        seq_info=f"{len(sequences)}seq"; 
        if prefix: seq_info += f"_{prefix}"
        return f"{seq_info}_{cache_hash}"

    def submit(self, sequences: List[str], genomic_distance: Optional[int]=20, prefix: Optional[str]=None) -> str:
        query=""
        for i,seq in enumerate(sequences, start=101):
            query += f">{prefix+'_' if prefix else ''}{i}\n{seq}\n"
        if len(sequences)==1:
            endpoint="ticket/msa"; mode="env"
        else:
            endpoint="ticket/pair"
            mode = "paircomplete" if genomic_distance is None else f"paircomplete-pairfilterprox_{genomic_distance}"
        r = requests.post(f"{self.host_url}/{endpoint}", data={'q':query,'mode':mode}, timeout=60)
        if r.status_code!=200: raise RuntimeError(f"Submit failed: {r.text}")
        self.job_id = r.json()['id']; return self.job_id

    def wait(self, poll=5):
        while True:
            s = requests.get(f"{self.host_url}/ticket/{self.job_id}", timeout=30).json().get('status','UNKNOWN')
            print(f"[STATUS] {s}")
            if s=="COMPLETE": break
            if s=="ERROR": raise RuntimeError("Job failed")
            time.sleep(poll)

    def download_and_parse(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tar_path = Path(output_dir) / f"{self.job_id}.tar.gz"
        r = requests.get(f"{self.host_url}/result/download/{self.job_id}", timeout=120)
        tar_path.write_bytes(r.content)
        with tarfile.open(tar_path, "r:gz") as tar: tar.extractall(output_dir)
        pair_a3m = Path(output_dir) / "pair.a3m"
        if pair_a3m.exists(): self.parsed_entries = self._parse_paired_a3m(str(pair_a3m))
        else:
            # 单链情况（不常用）：拼接 uniref / bfd 等
            a3ms = [Path(output_dir)/"uniref.a3m", Path(output_dir)/"bfd.mgnify30.metaeuk30.smag30.a3m"]
            lines=[]
            for p in a3ms:
                if p.exists(): lines += p.read_text(errors='ignore').splitlines()
            # 简单解析为单链 entries
            self.parsed_entries = []
            # 第一条为 query
            self.parsed_entries.append({'headers':['query'],'sequences':[''], 'coverages':[1.0],'identities':[1.0],
                                        'evalues':[0.0],'alnscores':[float('inf')],'uids':[''],'uniprot_nums':[0],'has_uniref':False,'is_query':True})

    def save_msa(self, output_file: str,
                 min_coverage: Optional[float]=None,
                 min_identity: Optional[float]=None,
                 max_evalue: Optional[float]=None,
                 min_alnscore: Optional[float]=None,
                 max_genomic_distance: Optional[int]=None) -> Tuple[int, List[Dict]]:
        assert self.parsed_entries, "No MSA parsed yet."
        if min_coverage and min_coverage>1: min_coverage/=100.0
        if min_identity and min_identity>1: min_identity/=100.0
        num_written=0; kept=[]
        num_chains = len(self.parsed_entries[0]['sequences']) if self.parsed_entries else 0
        with open(output_file,'w') as f:
            for entry in self.parsed_entries:
                if not entry['is_query']:
                    reasons=[]
                    if min_coverage and any(c < min_coverage for c in entry['coverages']): reasons.append("cov")
                    if min_identity and any(i < min_identity for i in entry['identities']): reasons.append("id")
                    if max_evalue is not None and any(e is not None and e > max_evalue for e in entry['evalues']): reasons.append("e")
                    if min_alnscore is not None and any(a is not None and a < min_alnscore for a in entry['alnscores']): reasons.append("aln")
                    if max_genomic_distance is not None and entry['has_uniref']:
                        dists = self._calculate_genomic_distances(entry)
                        if num_chains==2:
                            if dists[0] != -1 and dists[0] > max_genomic_distance: reasons.append("gdist")
                        else:
                            vd=[d for d in dists if d!=-1]
                            if vd and all(d > max_genomic_distance for d in vd): reasons.append("gdist")
                    if reasons: continue
                # header
                if entry['is_query']:
                    header="query" + "".join([f"_len{len(s)}" for s in entry['sequences']])
                else:
                    ids = [u if u else h for u,h in zip(entry['uids'], entry['headers'])]
                    header="_".join(ids)
                    if entry['has_uniref'] and all(entry['uids']):
                        for d in self._calculate_genomic_distances(entry):
                            if d != -1: header += f"_{d}"
                seq = ''.join(entry['sequences']).replace('\x00','')
                f.write(f">{header}\n{seq}\n")
                num_written+=1; kept.append(entry)
        return num_written, kept

def get_paired_msa(sequences: Union[str, List[str]],
                   output_file: str,
                   genomic_distance: Optional[int]=20,
                   min_coverage: Optional[float]=None,
                   min_identity: Optional[float]=None,
                   max_evalue: Optional[float]=None,
                   min_alnscore: Optional[float]=None,
                   prefix: Optional[str]=None,
                   host_url: str="https://api.colabfold.com",
                   cache_dir: Optional[str]=None) -> str:
    # normalize input
    if isinstance(sequences,str):
        sequences=[s.strip().upper() for s in sequences.split(':') if s.strip()]
    else:
        sequences=[s.strip().upper() for s in sequences if s.strip()]
    assert sequences, "Empty sequences"
    # run
    msa = ColabFoldPairedMSA(host_url, cache_dir)
    msa.submit(sequences, genomic_distance, prefix)
    msa.wait()
    tmp_dir = tempfile.mkdtemp(prefix="colabfold_api_")
    try:
        msa.download_and_parse(tmp_dir)
        _, _ = msa.save_msa(output_file,
                            min_coverage=min_coverage,
                            min_identity=min_identity,
                            max_evalue=max_evalue,
                            min_alnscore=min_alnscore,
                            max_genomic_distance=genomic_distance)
    finally:
        # clean null bytes
        p = Path(output_file)
        if p.exists():
            p.write_bytes(p.read_bytes().replace(b"\x00", b""))
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_file
# ======= End drop-in =======
# ===== 输入两条链（Peptide + Protein） =====
peptide_seq = "MFLKVRAEKRLGNFRLNVDFEMGRDYCVLLGPTGAGKSVFLELIAGIVKPDRGEVRLNGADITPLPPERRGIGFVPQDYALFPHLSVYRNIAYGLRNVERVERDRRVREMAEKLGIAHLLDRKPARLSGGERQRVALARALVIQPRLLLLDEPLSAVDLKTKGVLMEELRFVQREFDVPILHVTHDLIEAAMLADEVAVMLNGRIVEKGKLKELFSAKNGEVAEFLSARNLLLKVSKILD"
protein_seq = "MGHHHHHHHHHHSSGENLYFQGHMRLLFSALLALLSSIILLFVLLPVAATVTLQLFNFDEFLKAASDPAVWKVVLTTYYAALISTLIAVIFGTPLAYILARKSFPGKSVVEGIVDLPVVIPHTVAGIALLVVFGSSGLIGSFSPLKFVDALPGIVVAMLFVSVPIYINQAKEGFASVDVRLEHVARTLGSSPLRVFFTVSLPLSVRHIVAGAIMSWARGISEFGAVVVIAYYPMIAPTLIYERYLSEGLSAAMPVAAILILLSLAVFVALRIIVGREDVSEGQG"
out_dir = Path("test_pair_api_out_complex")
out_dir.mkdir(exist_ok=True)
a3m_path = out_dir / "pair_filtered.a3m"
'''
# 这里的 cov、qid 就是你在 Pairformer 笔记里用的设置（75 / 15）
get_paired_msa(
    sequences=[peptide_seq, protein_seq],
    output_file=str(a3m_path),
    genomic_distance=20,   # Δgene
    min_coverage=75,       # 可用 0.75 或 75
    min_identity=15,       # 可用 0.15 或 15
    # max_evalue=1e-3,     # 需要再收紧可以打开
    # min_alnscore=50,     # 需要再收紧可以打开
)

print(f"[OK] Saved filtered MSA to: {a3m_path}")
'''
# 验证预览
with open(a3m_path) as f:
    lines = f.readlines()
print(f"[INFO] seqs: {sum(1 for l in lines if l.startswith('>'))}")
print("".join(lines[:20]))


import torch, os
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import MSA, prepare_msa_masks, aa2tok_d

# 1) 设备自适应
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(f"[INFO] use_cuda={use_cuda}, device={device}")

# 如果你之前硬编码了某个 GPU，比如 os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# 在无 GPU 场景请去掉或注释掉；否则 PyTorch 依旧会判断为不可用。

# 2) 从 Hugging Face / 本地加载权重 —— 强制 map_location
#    如果你是本地路径：
#    model = MSAPairformer.from_pretrained(weights_dir="../weights/model.bin", device=device)
#    如果是从 Hub：
model = MSAPairformer.from_pretrained(device=device)  # 内部会用到 torch.load
# 若你的 from_pretrained 版本不传 map_location，你可以把 device 设为 cpu，
# 或者在库里 torch.load 的位置加 map_location=torch.device('cpu')（见下方进阶修复）

model = model.to(device)

# 3) MSA 预处理
msa_obj = MSA(
    msa_file_path=str(a3m_path),
    max_seqs=512,
    max_length=2000,
    max_tokens=1e12,
    diverse_select_method="hhfilter",
    hhfilter_kwargs={"binary": "hhfilter"},
)

msa_tokenized_t = msa_obj.diverse_tokenized_msa
msa_onehot_t = torch.nn.functional.one_hot(
    msa_tokenized_t, num_classes=len(aa2tok_d)
).unsqueeze(0).float().to(device)

mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_obj.diverse_tokenized_msa.unsqueeze(0))
mask, msa_mask, full_mask, pairwise_mask = [x.to(device) for x in [mask, msa_mask, full_mask, pairwise_mask]]

# 4) Query bias
use_query_biasing = True


# 5) 前向推理：无 GPU 时不要用 cuda autocast / bfloat16
breaks = [len(peptide_seq)]
import time

print("[INFO] Starting model inference ...")
start_time = time.time()
print(f"[INFO] Input tensor shape: {msa_onehot_t.shape}")
print(f"[INFO] MSA tokens: {msa_tokenized_t.shape}, breaks: {breaks}")
print(f"[INFO] Using device: {device}")
print(f"[INFO] Return seq weights: True, pairwise repr: None, msa repr: None")

with torch.no_grad():
    if use_cuda:
        print("[INFO] Running on GPU with autocast (bfloat16)...")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            results = model(
                msa=msa_onehot_t.to(torch.bfloat16),
                mask=mask, msa_mask=msa_mask, full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                complex_chain_break_indices=[breaks],
                return_seq_weights=True,
                return_pairwise_repr_layer_idx=None,
                return_msa_repr_layer_idx=None
            )
        torch.cuda.synchronize()
        print(f"[INFO] GPU inference done in {time.time()-t0:.2f}s.")
    else:
        print("[INFO] Running on CPU (this may take several minutes)...")
        t0 = time.time()
        results = model(
            msa=msa_onehot_t,
            mask=mask, msa_mask=msa_mask, full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            complex_chain_break_indices=[breaks],
            return_seq_weights=True,
            return_pairwise_repr_layer_idx=None,
            return_msa_repr_layer_idx=None
        )
        print(f"[INFO] CPU inference done in {time.time()-t0:.2f}s.")

elapsed = time.time() - start_time
print(list(results.keys()))

print(f"[INFO] Inference completed in {elapsed:.2f} seconds total.")
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import json
import psutil, time

# ====== Settings ======
# 自动生成 key —— 用输入序列哈希或前缀名

key = "3abm_X_Q_nomutation"


# 输出根目录
out_root = Path("outputs/exp_axial_withpadandseq")
out_npz = out_root / "pred_npz"
out_txt = out_root / "texts"
out_fig = out_root / "figs"
for p in [out_npz, out_txt, out_fig]:
    p.mkdir(parents=True, exist_ok=True)

# ====== Extract contacts ======
t0 = time.time()
contacts = results["contacts"].detach().cpu().float().numpy()
if contacts.ndim == 3:
    contacts = contacts[0]

L_pep = len(peptide_seq)
L_total = contacts.shape[0]
L_prot = L_total - L_pep

print(f"[INFO] Contact map shape: {contacts.shape}  (L_pep={L_pep}, L_prot={L_prot})")

# sigmoid normalization
prob = 1 / (1 + np.exp(-contacts))
binary = (prob >= 0.8).astype(np.uint8)
pair_mask = np.ones_like(prob, dtype=bool)

# ====== Optional: crop cross-section ======
SAVE_FULL_MAP = False  # <- 改为 True 可同时保存完整 map
cross_prob = prob[:L_pep, L_pep:]
cross_scores = contacts[:L_pep, L_pep:]
cross_binary = binary[:L_pep, L_pep:]
cross_mask = pair_mask[:L_pep, L_pep:]

# ====== Save npz ======
if SAVE_FULL_MAP:
    np.savez_compressed(
        out_npz / f"{key}_full.npz",
        scores=contacts, prob=prob, binary=binary, pair_mask=pair_mask, key=key,
    )
    print(f"[SAVED] full map → {out_npz/f'{key}_full.npz'}")

np.savez_compressed(
    out_npz / f"{key}.npz",
    scores=cross_scores,
    prob=cross_prob,
    binary=cross_binary,
    pair_mask=cross_mask,
    key=key,
)
print(f"[SAVED] cross map → {out_npz/f'{key}.npz'}")

# ====== Save text summary ======
mem_used = psutil.Process().memory_info().rss / 1024**3  # GB
elapsed_post = time.time() - t0
lines = [
    f"key: {key}",
    f"shape_full: {contacts.shape}, shape_cross: {cross_prob.shape}",
    f"prob: min={prob.min():.3f}, mean={prob.mean():.3f}, max={prob.max():.3f}",
    f"mem_used_GB: {mem_used:.3f}",
    f"time_postprocess_s: {elapsed_post:.2f}",
]
(out_txt / f"{key}.txt").write_text("\n".join(lines))
print(f"[SAVED] → {out_txt/f'{key}.txt'}")

# ====== Plot probability heatmap (cross section only) ======
plt.figure(figsize=(6, 3))
plt.imshow(cross_prob, origin="upper", aspect="auto", vmin=0, vmax=1, cmap="viridis")
plt.title(f"{key} peptide-protein prob")
plt.xlabel("Protein residue index")
plt.ylabel("Peptide residue index")
plt.colorbar(fraction=0.03, pad=0.02, label="Contact probability")
plt.tight_layout()
plt.savefig(out_fig / f"{key}_prob.png", dpi=200)
plt.close()
print(f"[SAVED] → {out_fig/f'{key}_prob.png'}")

# ====== ---------- GT: load, crop, metrics, plots ---------- ======
GT_INDEX      = "../../generate_phase1_db/full_distance_maps_v2/full_map_cb.json"  # 改成你的路径；None 表示不启用
GT_THRESHOLD  = 8.0   # Å
BIN_TH        = 0.8   # 与保存binary一致
SAVE_FULL_MAP = False # 你上面已有（保持不变）

def _maybe_strip_suffix(k: str) -> str:
    # 和 run_eval 里的思路一致：某些 key 有像 `_nomutation` 之类后缀
    for suf in ["_nomutation", "_mutation", "_mut", "_nomut"]:
        if k.endswith(suf):
            return k[: -len(suf)]
    return k

def _load_gt_npz(gt_npz_path: str) -> tuple[np.ndarray, str]:
    """
    返回 (gt_array, kind)
    kind ∈ {"contact","distance"}
    兼容字段：
      - contact: 'labels_contact', 'gt_map' (0/1)
      - distance: 'distance', 'scores' (Å)
    """
    d = np.load(gt_npz_path)
    # contact (preferred)
    for k in ["labels_contact", "gt_map", "contact", "contacts"]:
        if k in d:
            arr = d[k].astype(np.float32)
            # 若不是 0/1，做 clip
            arr = (arr > 0.5).astype(np.float32) if arr.max() > 1.0 else arr
            return arr, "contact"
    # distance (Å)
    for k in ["distance", "dist", "scores"]:
        if k in d:
            arr = d[k].astype(np.float32)
            return arr, "distance"
    raise RuntimeError(f"Unrecognized GT fields in {gt_npz_path}. Keys={list(d.keys())}")

def _binarize_from_distance(dist: np.ndarray, thr: float) -> np.ndarray:
    # 小于等于阈值为接触
    return (dist <= float(thr)).astype(np.float32)

def _metrics_from_binary(pred_prob: np.ndarray, gt_bin: np.ndarray, mask: np.ndarray | None = None) -> dict:
    p = pred_prob.ravel()
    g = gt_bin.ravel().astype(np.int32)
    if mask is not None:
        m = mask.ravel().astype(bool)
        p, g = p[m], g[m]
    eps = 1e-12
    pred_bin = (p >= float(BIN_TH)).astype(np.int32)
    tp = int(((pred_bin == 1) & (g == 1)).sum())
    fp = int(((pred_bin == 1) & (g == 0)).sum())
    fn = int(((pred_bin == 0) & (g == 1)).sum())

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)

    out = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

    # try AUROC / AUPRC via sklearn
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(g)) > 1:
            out["auroc"] = float(roc_auc_score(g, p))
            out["auprc"] = float(average_precision_score(g, p))
        else:
            out["auroc"] = None
            out["auprc"] = None
    except Exception as e:
        print(f"[WARN] sklearn not available or AUROC/AUPRC failed: {e}")
        out["auroc"] = None
        out["auprc"] = None

    return out

def _save_heatmap(mat: np.ndarray, path: Path, title: str, vmin=None, vmax=None):
    plt.figure(figsize=(6, 3))
    plt.imshow(mat, origin="upper", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    plt.title(title)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

if GT_INDEX and Path(GT_INDEX).exists():
    try:
        # 1) 读取 key->npz 的索引
        with open(GT_INDEX, "r") as f:
            gt_map_index = json.load(f)
        gt_npz_path = gt_map_index.get(key, None)
        if gt_npz_path is None:
            key2 = _maybe_strip_suffix(key)
            gt_npz_path = gt_map_index.get(key2, None)
        if gt_npz_path is None:
            raise FileNotFoundError(f"GT npz not found for key '{key}' in {GT_INDEX}")

        # 2) 加载 GT 原图
        gt_full, kind = _load_gt_npz(gt_npz_path)  # shape ~ [Tl,Tp] 或全图
        # 3) 裁剪到 peptide×protein
        #    你的 pred是 full-map 524x524；我们要 cross = [:L_pep, L_pep:]
        gt_shape = gt_full.shape
        # 如果 GT 已经是 cross 形状（近似 L_pep x L_prot），就直接 min 裁剪
        # 否则若 GT 也是 full-map（L_total x L_total），也按 cross 切
        if gt_shape == contacts.shape:
            gt_cross_full = gt_full[:L_pep, L_pep:]
        else:
            # 假设 GT 就是 cross 区（Tl≈L_pep, Tp≈L_prot）
            gt_cross_full = gt_full

        # 4) 若 GT 为距离，阈值化
        if kind == "distance":
            gt_bin = _binarize_from_distance(gt_cross_full, GT_THRESHOLD)
        else:
            # contact -> 保证是 0/1
            gt_bin = (gt_cross_full > 0.5).astype(np.float32)

        # 5) 同步裁剪 pred（你上面已得到 cross_prob / cross_binary）
        # 确保大小一致（取最小公共区域）
        rr = min(cross_prob.shape[0], gt_bin.shape[0])
        cc = min(cross_prob.shape[1], gt_bin.shape[1])
        pred_prob_use = cross_prob[:rr, :cc]
        gt_bin_use    = gt_bin[:rr, :cc]
        mask_use      = None  # 若你有更严格的掩码可以加在这里

        # 6) 计算指标
        m = _metrics_from_binary(pred_prob_use, gt_bin_use, mask_use)

        # 7) 保存 GT 图与 diff
        _save_heatmap(gt_bin_use, out_fig / f"{key}_gt.png",
                      title=f"{key} GT contact (thr={GT_THRESHOLD}Å)", vmin=0.0, vmax=1.0)
        _save_heatmap(np.abs(pred_prob_use - gt_bin_use), out_fig / f"{key}_diff.png",
                      title=f"{key} |pred-gt|", vmin=0.0, vmax=1.0)

        # 8) 追加写入文本与 summary
        with open(out_txt / f"{key}.txt", "a") as f:
            f.write("\n[GT]\n")
            f.write(f"gt_shape: {gt_bin.shape}, used: {gt_bin_use.shape}\n")
            f.write(f"metrics@{BIN_TH}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}\n")
            f.write(f"AUROC={m['auroc'] if m['auroc'] is not None else 'NA'}  "
                    f"AUPRC={m['auprc'] if m['auprc'] is not None else 'NA'}\n")

        # summary.json（追加/合并）
        summary_file = out_root / "summary.json"
        summary_obj = {}
        if summary_file.exists():
            try:
                summary_obj = json.load(open(summary_file, "r"))
            except Exception:
                summary_obj = {}
        summary_obj.setdefault(key, {})
        summary_obj[key].update({
            "shape_cross_pred": [int(x) for x in cross_prob.shape],
            "shape_cross_gt":   [int(x) for x in gt_bin.shape],
            "metrics": m,
            "bin_th": BIN_TH,
            "gt_threshold_A": GT_THRESHOLD,
        })
        json.dump(summary_obj, open(summary_file, "w"), indent=2)
        print(f"[GT] metrics saved -> {summary_file}")

    except Exception as e:
        print(f"[WARN] GT evaluation skipped: {e}")
else:
    print("[INFO] GT_INDEX not set or not found. Skipping GT evaluation.")

# ====== 原有 skip_summary.json （保留）======
summary_path = out_root / "skip_summary.json"
if not summary_path.exists():
    json.dump(
        {"total_samples": 1, "ok_aligned": 1, "skipped_misaligned": 0},
        open(summary_path, "w"), indent=2
    )

print(f"[DONE] All results saved under {out_root.resolve()}")

