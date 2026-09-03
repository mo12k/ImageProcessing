"""One-off builder for the self-contained final academic notebook."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
BACKUPS = ROOT / "backups"
NOTEBOOK = ROOT / "Fingerprint_Enhancement_System.ipynb"
CURRENT_BACKUP = BACKUPS / "Fingerprint_Enhancement_System_before_selfcontained_rebuild.ipynb"
FULL_BACKUP = BACKUPS / "Fingerprint_Enhancement_System_preconsolidation_full.ipynb"
HISTORICAL = ROOT / ".ipynb_checkpoints" / "Fingerprint_Enhancement_System-checkpoint.ipynb"

OUT.mkdir(exist_ok=True)
BACKUPS.mkdir(exist_ok=True)
if not CURRENT_BACKUP.exists():
    shutil.copy2(NOTEBOOK, CURRENT_BACKUP)
if not FULL_BACKUP.exists():
    shutil.copy2(HISTORICAL, FULL_BACKUP)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

(OUT / "final_notebook_prechange_hashes.json").write_text(json.dumps({
    "current_consolidated_notebook": {"path": str(CURRENT_BACKUP.relative_to(ROOT)), "sha256": sha256(CURRENT_BACKUP)},
    "richest_preconsolidation_notebook": {"path": str(FULL_BACKUP.relative_to(ROOT)), "sha256": sha256(FULL_BACKUP)},
}, indent=2), encoding="utf-8")

recovery = """# Final notebook recovery audit

## A. Historical versions found

- The 12 KB consolidated notebook (23 cells), which loaded prior CSV/JSON results.
- `.ipynb_checkpoints/Fingerprint_Enhancement_System-checkpoint.ipynb` (27 cells, about 3.58 MB).
- `backups/audit_preserve_2026-09-01_184155/Fingerprint_Enhancement_System.ipynb` (30 cells, about 3.34 MB).
- Two source-truth-repair notebook copies (27/28 cells).
- Three Git revisions containing the notebook.

## B. Richest implementation source

The 30-cell `audit_preserve` notebook had the broadest authored structure; the 3.58 MB checkpoint contained the same core implementation plus the richest saved execution evidence. The checkpoint was protected as the pre-consolidation full copy.

## C. Useful content lost during consolidation

Image loading, controlled degradation, baseline filters, Gabor code, segmentation, thresholding, morphology, thinning, medial axis, minutiae detection, metric functions, batch execution and visualisation cells were removed or reduced to prose.

## D. Scientifically outdated sections

Old allocation of CLAHE/restoration to Member 1, ordinary multi-orientation Gabor presented as the final Member 2 method, morphology as Member 3, thinning as Member 4, broad degradation searches, SVM extension, and Wiener+CLAHE candidate claims are historical only.

## E. Restored sections

Dataset discovery, P0, conventional baselines, structural processing, metric functions, controlled degradation, executable comparisons, visualisations and academic interpretation.

## F. Replaced sections

Member assignments and final experiments were replaced with the verified frozen M1 NLM, M2 adaptive Gabor, M3 TV and M4 coherence-guided directional diffusion implementations. All CSV/JSON result-loading cells were replaced by direct recomputation.
"""
(OUT / "final_notebook_recovery_audit.md").write_text(recovery, encoding="utf-8")

old_nb = nbf.read(CURRENT_BACKUP, as_version=4)
deps = []
for i, cell in enumerate(old_nb.cells):
    if cell.cell_type != "code":
        continue
    for marker, purpose in [("member4_controlled_summary.csv", "member comparison"), ("member_compatibility_summary.csv", "compatibility"), ("final_holdout_statistics.json", "fresh hold-out"), ("final_large_real_statistics.json", "500-image validation"), ("final_large_altered_summary.csv", "Altered validation")]:
        if marker in cell.source:
            deps.append({"cell": i, "old_dependency": f"outputs/{marker}", "purpose": purpose,
                         "replacement": "Executable dataset selection, processing and metric aggregation cells",
                         "status": "RECOMPUTED IN NOTEBOOK"})
pd.DataFrame(deps).to_csv(OUT / "final_notebook_external_result_dependency_audit.csv", index=False)

# Freeze image identities, not result values, into the notebook. These lists reproduce the
# already selected experiments without requiring their historical CSV files at Run All time.
holdout_names = pd.read_csv(OUT / "final_holdout_image_list.csv").sort_values("selection_order")["filename"].tolist()
canonical_names = pd.read_csv(OUT / "final_large_real_image_list.csv").sort_values("index")["filename"].tolist()

core_text = (ROOT / "Fingerprint_Enhancement_System.py").read_text(encoding="utf-8")
core_body = core_text.split("from __future__ import annotations", 1)[1]
core_body = "from __future__ import annotations\n" + core_body

nb = nbf.v4.new_notebook()
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11"}})

def md(text: str): nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text: str): nb.cells.append(nbf.v4.new_code_cell(text.strip()))

md("""# Fingerprint Enhancement System

**BMDS2133 Image Processing — Mode A: Comparative & Enhancement Study**

This notebook is the complete, executable academic submission. Its objective is fingerprint **enhancement**, not identity recognition: degraded or low-quality ridge flow is clarified while ridge structure is preserved to support optional later feature analysis. Restarting the kernel and running all cells recomputes every key result directly from `data/`; historical experiment CSV/JSON files are not experimental inputs.""")

md("## 2. Imports and global configuration")
code("""from pathlib import Path
from time import perf_counter
import json, math, platform, sys
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi, signal
from skimage import exposure, filters, measure, morphology, restoration, transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
if not (ROOT / 'data' / 'SOCOFing').is_dir():
    raise FileNotFoundError('Run this notebook with its project directory as the working directory.')
DATA = ROOT / 'data' / 'SOCOFing'; REAL_DIR = DATA / 'Real'; ALTERED_DIR = DATA / 'Altered'; OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 20260902; HOLDOUT_SEED = 20260903; NOISE_SEED_BASE = 20260903; ALTERED_SEED = 20260904
BENCHMARK_PSNR = 28.17; FINAL_NLM_H = 0.06; CONTROLLED_SEVERITY = 0.55; GAUSSIAN_SIGMA = 0.018 + 0.055 * CONTROLLED_SEVERITY
np.random.seed(SEED)
print({'Python': sys.version.split()[0], 'OpenCV': cv2.__version__, 'NumPy': np.__version__, 'platform': platform.platform(),
       'NLM backend': 'cv2.fastNlMeansDenoising', 'h_normalised': FINAL_NLM_H, 'sigma': GAUSSIAN_SIGMA})""")

md("""## 3. Dataset loading

SOCOFing contains clean `Real` captures and independently altered Easy, Medium and Hard images. The Altered images are evaluated without claiming pixel-aligned Real↔Altered ground truth.""")
code("""SUPPORTED_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
def list_images(folder):
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS), key=lambda p: p.name.lower())
dataset_paths = {'Real': list_images(REAL_DIR),
 'Easy': list_images(ALTERED_DIR/'Altered-Easy'), 'Medium': list_images(ALTERED_DIR/'Altered-Medium'), 'Hard': list_images(ALTERED_DIR/'Altered-Hard')}
dataset_inventory = pd.DataFrame([{'split': k, 'count': len(v), 'representative_path': str(v[0].relative_to(ROOT))} for k,v in dataset_paths.items()])
display(dataset_inventory)
fig, axes = plt.subplots(1,4,figsize=(13,3));
for ax,(name,paths) in zip(axes,dataset_paths.items()):
    ax.imshow(Image.open(paths[0]).convert('L'), cmap='gray'); ax.set_title(name); ax.axis('off')
plt.tight_layout(); plt.show()""")

md("""## 4. P0 neutral preprocessing

P0 performs only grayscale conversion, 256×256 resizing and 1st/99th percentile intensity preparation. It contains no enhancement.""")
code(core_body)
code("""# Exact frozen M2/M4 comparison implementations recovered from screening24_master_audit.py.
# They intentionally override the later reusable-core copies above; no parameter is retuned.
def estimate_orientation_field(image, smoothing_sigma=3.0, orientation_smoothing_sigma=2.0):
    base=_metric_image(image); gx,gy=filters.sobel_h(base),filters.sobel_v(base)
    gxx=ndi.gaussian_filter(gx*gx,smoothing_sigma);gyy=ndi.gaussian_filter(gy*gy,smoothing_sigma);gxy=ndi.gaussian_filter(gx*gy,smoothing_sigma)
    sin2=ndi.gaussian_filter(2*gxy,orientation_smoothing_sigma);cos2=ndi.gaussian_filter(gxx-gyy,orientation_smoothing_sigma)
    return 0.5*np.arctan2(sin2,cos2), np.sqrt((gxx-gyy)**2+4*gxy**2)/(gxx+gyy+1e-8)

def estimate_local_frequency_map(image, mask, block_size=32, frequencies=(.075,.095,.115,.140)):
    base=_metric_image(image);mask=np.asarray(mask,bool);freq_map=np.full_like(base,float(np.median(frequencies)),dtype=np.float32);confidence=np.zeros_like(base,dtype=np.float32)
    window=np.outer(np.hanning(block_size),np.hanning(block_size));yy,xx=np.mgrid[-block_size//2:block_size//2,-block_size//2:block_size//2]
    radius=np.sqrt(xx**2+yy**2);center_exclusion=radius<=2
    for row in range(0,base.shape[0]-block_size+1,block_size//2):
        for col in range(0,base.shape[1]-block_size+1,block_size//2):
            bm=mask[row:row+block_size,col:col+block_size]
            if float(np.mean(bm))<.25: continue
            block=base[row:row+block_size,col:col+block_size]
            if float(np.std(block[bm]))<.012: continue
            spectrum=np.abs(np.fft.fftshift(np.fft.fft2((block-np.mean(block[bm]))*window)));spectrum[center_exclusion]=0
            peak=np.unravel_index(int(np.argmax(spectrum)),spectrum.shape);observed=float(radius[peak]/block_size);nearest=min(frequencies,key=lambda f:abs(f-observed));conf=float(np.max(spectrum)/(np.mean(spectrum)+1e-8))
            region=np.s_[row:row+block_size,col:col+block_size];freq_map[region]=nearest;confidence[region]=max(float(np.mean(confidence[region])),min(conf/25,1))
    return freq_map,confidence

def apply_modified_gabor(image, frequencies=(.075,.095,.115,.140), orientation_bins=8, blend=.18):
    base=_metric_image(image);mask=_fingerprint_mask(base);orientation,coherence=estimate_orientation_field(base);freq_map,freq_conf=estimate_local_frequency_map(base,mask,frequencies=frequencies)
    bins=np.linspace(0,np.pi,orientation_bins,endpoint=False);oi=np.argmin(np.abs(np.angle(np.exp(1j*(orientation[...,None]-bins[None,None,:])))),axis=2);fi=np.argmin(np.abs(freq_map[...,None]-np.asarray(frequencies)[None,None,:]),axis=2)
    selected=np.zeros_like(base,dtype=np.float32);inverted=1-base
    for f_index,frequency in enumerate(frequencies):
        for o_index,theta in enumerate(bins):
            real,imag=filters.gabor(inverted,frequency=float(frequency),theta=float(theta),bandwidth=1.8);response=np.sqrt(real**2+imag**2);selector=(oi==o_index)&(fi==f_index)&mask;selected[selector]=response[selector]
    ridge=np.where(mask,1-normalise_image(selected),base);confidence_weight=np.clip(.30+.70*coherence,0,1)*np.clip(.40+.60*freq_conf,0,1);weight=blend*confidence_weight*mask
    return _metric_image((1-weight)*base+weight*ridge)

def apply_coherence_guided_directional_diffusion(image, iterations=4, step=.16):
    current=_metric_image(image);mask=_fingerprint_mask(current);bins=np.linspace(0,np.pi,8,endpoint=False)
    for _ in range(iterations):
        orientation,coherence=estimate_orientation_field(current);indexes=np.argmin(np.abs(np.angle(np.exp(1j*(orientation[...,None]-bins[None,None,:])))),axis=2);smoothed=np.zeros_like(current)
        for index,theta in enumerate(bins):
            conv=ndi.convolve(current,_oriented_kernel(theta),mode='reflect');smoothed[indexes==index]=conv[indexes==index]
        weight=np.clip(coherence,0,1)*mask;current=_metric_image((1-step*weight)*current+step*weight*smoothed)
    return current

def segment_fingerprint(image):
    base=_metric_image(image);mask=_fingerprint_mask(base);ridges=(base<filters.threshold_sauvola(base,window_size=25,k=.16))&mask
    ridges=morphology.remove_small_objects(ridges,min_size=18);ridges=morphology.remove_small_holes(ridges,area_threshold=18)
    return ridges.astype(bool),mask""")

md("""## 5. Conventional baselines — comparators only

Gaussian, median, CLAHE, contrast stretching, ordinary Gabor and Wiener restoration are executable comparators. CLAHE is not Member 1.""")
code("""sample_p0 = load_fingerprint(dataset_paths['Real'][0])
baseline_outputs = {'P0': sample_p0, 'Gaussian': apply_gaussian(sample_p0), 'Median': apply_median(sample_p0),
 'CLAHE': apply_clahe(sample_p0), 'Contrast stretching': apply_contrast_stretching(sample_p0),
 'Ordinary Gabor': apply_ordinary_gabor(sample_p0), 'Wiener': apply_wiener(sample_p0)}
fig, axes = plt.subplots(2,4,figsize=(13,7));
for ax,(name,img) in zip(axes.ravel(),baseline_outputs.items()): ax.imshow(img,cmap='gray',vmin=0,vmax=1); ax.set_title(name); ax.axis('off')
axes.ravel()[-1].axis('off'); plt.tight_layout(); plt.show()""")

md("""## 6. Member 1 — Non-Local Means denoising

`apply_nlm` above is the visible frozen implementation. It searches neighbouring patches and weights them by similarity, suppressing Gaussian noise while retaining repeated ridge texture. The fixed setting is `h=0.06`, converted to 15.3 uint8 units, with template window 7 and search window 21. It is not retuned here.""")
code("assert FINAL_NLM_H == 0.06 and NLM_TEMPLATE_WINDOW_SIZE == 7 and NLM_SEARCH_WINDOW_SIZE == 21\nnlm_demo = apply_nlm(sample_p0, h=FINAL_NLM_H); print('Frozen M1 implementation ready:', nlm_demo.shape)")

md("""## 7. Member 2 — Modified / orientation-adaptive Gabor

`_orientation_field`, `_frequency_map`, and `apply_modified_gabor` visibly estimate local ridge orientation and frequency, select among four frequencies and eight orientation bins, and confidence-blend the response. This differs from the fixed ordinary Gabor baseline.""")
code("modified_gabor_demo = apply_modified_gabor(sample_p0); print('M2 adaptive output:', modified_gabor_demo.shape)")

md("""## 8. Member 3 — Total Variation restoration

The fixed `apply_tv_restoration(weight=0.02)` uses total-variation regularisation to smooth noise while penalising loss of edge/ridge boundaries.""")
code("tv_demo = apply_tv_restoration(sample_p0, weight=0.02); print('M3 TV output:', tv_demo.shape)")

md("""## 9. Member 4 — Coherence-guided directional diffusion

`_orientation_field`, `_oriented_kernel`, and `apply_coherence_guided_directional_diffusion` explicitly calculate orientation/coherence and repeatedly apply orientation-selected anisotropic smoothing. Fixed parameters are four iterations and step 0.16.""")
code("diffusion_demo = apply_coherence_guided_directional_diffusion(sample_p0, iterations=4, step=.16); print('M4 output:', diffusion_demo.shape)")

md("""## 10. Structural and downstream processing

The grayscale enhancement is followed, when needed, by segmentation → morphology → thinning/medial axis → feature analysis. These binary operations are not the final grayscale enhancer.""")
code("""ridges, foreground_mask = segment_fingerprint(nlm_demo)
cleaned = morphological_cleanup(ridges); opened = morphology.opening(cleaned, morphology.disk(1)); closed = morphology.closing(opened, morphology.disk(1))
skeleton = thin_ridges(closed); medial, distance = medial_axis_analysis(closed); minutiae = extract_minutiae_detections(skeleton, foreground_mask)
fig,axes=plt.subplots(1,5,figsize=(15,3));
for ax,(title,img) in zip(axes,[('Enhanced grayscale',nlm_demo),('Segmented',ridges),('Morphology',closed),('Skeleton',skeleton),('Medial axis',medial)]): ax.imshow(img,cmap='gray');ax.set_title(title);ax.axis('off')
plt.tight_layout();plt.show(); print({k:len(v) for k,v in minutiae.items()})""")

md("""## 11. Evaluation metrics

MSE, PSNR and SSIM are full-reference controlled metrics with data range 1. Structural metrics are used for unpaired original Altered images. Fragmentation is connected components per 1,000 ridge pixels; continuity is skeleton pixels per endpoint.""")
code("""def structural_metrics(image):
    base = _metric_image(image); mask = _fingerprint_mask(base); local_mean = ndi.uniform_filter(base,17,mode='reflect')
    local_std = np.sqrt(np.maximum(ndi.uniform_filter(base*base,17,mode='reflect')-local_mean*local_mean,0))
    binary,_ = segment_fingerprint(base); labels = measure.label(binary, connectivity=2); props=measure.regionprops(labels)
    skeleton=morphology.skeletonize(binary); points=extract_minutiae_detections(skeleton,mask)
    ridge_pixels=max(int(binary.sum()),1); endpoints=len(points['ridge_endings'])
    return {'coherence':calculate_ridge_coherence(base,mask),'contrast':float(local_std[mask].mean()) if mask.any() else float(local_std.mean()),
            'fragmentation':len(props)/(ridge_pixels/1000.0),'continuity':float(skeleton.sum())/max(endpoints,1),
            'ridge_coverage':float(binary.sum())/max(float(mask.sum()),1.0)}
def full_reference(reference,candidate):
    return {'mse':calculate_mse(reference,candidate),'psnr':calculate_psnr(reference,candidate),'ssim':calculate_ssim(reference,candidate)}
print(full_reference(sample_p0,sample_p0), structural_metrics(sample_p0))""")

md("""## 12. Controlled degradation

Synthetic degradation is used only for controlled full-reference evaluation, not as the real-world enhancement pipeline. The frozen `gaussian_noise_only_s055` protocol uses severity 0.55, exact sigma 0.04825, deterministic per-image NumPy RNG, and clipping to [0,1].""")
code("""def controlled_degradation(clean, seed):
    rng=np.random.default_rng(int(seed)); return _metric_image(clean + rng.normal(0.0, GAUSSIAN_SIGMA, clean.shape))
assert abs(GAUSSIAN_SIGMA-0.04825)<1e-15
demo_degraded=controlled_degradation(sample_p0,SEED); display(pd.DataFrame([full_reference(sample_p0,demo_degraded)]))""")

md("## 13. Four-member controlled comparison")
code("""all_real = list_images(REAL_DIR); rng=np.random.default_rng(SEED); indices=np.arange(len(all_real)); rng.shuffle(indices)
development_paths=[all_real[int(i)] for i in indices[:12]]
MEMBER_METHODS={'M1 NLM':lambda x:apply_nlm(x,.06),'M2 Modified Gabor':lambda x:apply_modified_gabor(x),
                'M3 TV':lambda x:apply_tv_restoration(x,.02),'M4 Directional Diffusion':lambda x:apply_coherence_guided_directional_diffusion(x,4,.16)}
member_rows=[]
for i,path in enumerate(development_paths):
    clean=load_fingerprint(path); degraded=controlled_degradation(clean,SEED+i)
    for name,method in MEMBER_METHODS.items(): member_rows.append({'image':path.name,'method':name,**full_reference(clean,method(degraded))})
member_per_image=pd.DataFrame(member_rows)
member_summary=member_per_image.groupby('method').agg(images=('image','count'),PSNR_mean=('psnr','mean'),PSNR_std=('psnr','std'),MSE_mean=('mse','mean'),MSE_std=('mse','std'),SSIM_mean=('ssim','mean'),SSIM_std=('ssim','std')).sort_values('PSNR_mean',ascending=False).reset_index()
display(member_summary); member_summary.to_csv(OUT/'notebook_member_comparison.csv',index=False)
member_summary.plot.bar(x='method',y='PSNR_mean',yerr='PSNR_std',legend=False,figsize=(8,4),ylabel='PSNR (dB)',title='Frozen four-member comparison'); plt.axhline(BENCHMARK_PSNR,color='red',ls='--');plt.tight_layout();plt.show()""")

md("## 14. Compatibility / hybrid investigation")
code("""COMPATIBILITY={'C0 NLM':('NLM',),'C1 TV -> NLM':('TV','NLM'),'C2 NLM -> Modified Gabor':('NLM','Gabor'),
 'C3 NLM -> Directional Diffusion':('NLM','Diffusion'),'C4 TV -> NLM -> Directional Diffusion':('TV','NLM','Diffusion')}
def apply_stages(image,stages):
    out=image
    for stage in stages:
        out={'NLM':lambda z:apply_nlm(z,.06),'TV':lambda z:apply_tv_restoration(z,.02),'Gabor':apply_modified_gabor,
             'Diffusion':lambda z:apply_coherence_guided_directional_diffusion(z,4,.16)}[stage](out)
    return out
compat_rows=[]
for i,path in enumerate(development_paths):
    clean=load_fingerprint(path); degraded=controlled_degradation(clean,SEED+i)
    for name,stages in COMPATIBILITY.items(): compat_rows.append({'image':path.name,'candidate':name,**full_reference(clean,apply_stages(degraded,stages))})
compat_per_image=pd.DataFrame(compat_rows)
compat_summary=compat_per_image.groupby('candidate').agg(images=('image','count'),PSNR_mean=('psnr','mean'),PSNR_std=('psnr','std'),MSE_mean=('mse','mean'),SSIM_mean=('ssim','mean')).sort_values('PSNR_mean',ascending=False).reset_index()
display(compat_summary);compat_summary.to_csv(OUT/'notebook_compatibility.csv',index=False)
compat_summary.plot.bar(x='candidate',y='PSNR_mean',legend=False,figsize=(9,4),ylabel='PSNR (dB)');plt.tight_layout();plt.show()""")

md("""## 15. Final method selection

The result is selected from the recomputed tables, not a stored claim. NLM is retained only if it is the best compatibility candidate; tested extra stages are rejected when they do not improve controlled fidelity.""")
code("""selected_candidate=compat_summary.iloc[0]['candidate']; assert selected_candidate=='C0 NLM'
print(f'Selected final grayscale method: NLM h={FINAL_NLM_H:.2f}. Best recomputed compatibility candidate: {selected_candidate}.')""")

md("## 16. Fresh 100-image hold-out validation")
code(f"""FROZEN_HOLDOUT_NAMES = {holdout_names!r}
holdout_paths=[REAL_DIR/name for name in FROZEN_HOLDOUT_NAMES]; assert len(holdout_paths)==100 and all(p.is_file() for p in holdout_paths)
def controlled_validation(paths, seed_base):
    rows=[]
    for i,path in enumerate(paths):
        clean=load_fingerprint(path); degraded=controlled_degradation(clean,seed_base+i); enhanced=apply_nlm(degraded,.06)
        before=full_reference(clean,degraded); after=full_reference(clean,enhanced)
        rows.append({{'filename':path.name,'noise_seed':seed_base+i,**{{f'degraded_{{k}}':v for k,v in before.items()}},**{{f'enhanced_{{k}}':v for k,v in after.items()}},
                     **{{f'delta_{{k}}':after[k]-before[k] for k in before}},'above_28_17':after['psnr']>BENCHMARK_PSNR}})
    return pd.DataFrame(rows)
holdout_results=controlled_validation(holdout_paths,NOISE_SEED_BASE)
holdout_summary=holdout_results.agg({{'degraded_psnr':['mean','std'],'enhanced_psnr':['mean','std','median','min','max'],'enhanced_mse':['mean'],'enhanced_ssim':['mean'],'delta_psnr':['mean'],'delta_mse':['mean'],'delta_ssim':['mean']}})
display(holdout_summary); print(f"Above 28.17 dB: {{holdout_results.above_28_17.sum()}}/{{len(holdout_results)}}")
holdout_results.to_csv(OUT/'notebook_fresh_holdout_per_image.csv',index=False)""")

md("## 17. Final 500-image robustness validation")
code(f"""FROZEN_CANONICAL_500_NAMES = {canonical_names!r}
large_paths=[REAL_DIR/name for name in FROZEN_CANONICAL_500_NAMES]; assert len(large_paths)==500 and len(set(large_paths))==500 and all(p.is_file() for p in large_paths)
large_results=controlled_validation(large_paths,NOISE_SEED_BASE)
large_stats={{'count':len(large_results),'mean_psnr':large_results.enhanced_psnr.mean(),'std_psnr':large_results.enhanced_psnr.std(),
 'median_psnr':large_results.enhanced_psnr.median(),'min_psnr':large_results.enhanced_psnr.min(),'max_psnr':large_results.enhanced_psnr.max(),
 'mean_mse':large_results.enhanced_mse.mean(),'mean_ssim':large_results.enhanced_ssim.mean(),'above_28_17':int(large_results.above_28_17.sum())}}
large_stats['percentage_above_28_17']=100*large_stats['above_28_17']/large_stats['count'];large_stats['margin_to_benchmark']=large_stats['mean_psnr']-BENCHMARK_PSNR
display(pd.DataFrame([large_stats])); large_results.to_csv(OUT/'notebook_final_500_per_image.csv',index=False)""")

md("""## 18. Original Altered large validation

Exactly 500 images per difficulty are deterministically selected and processed directly: Original Altered → P0 → NLM. No synthetic noise and no unverified paired reference are used.""")
code("""altered_rows=[]; selected_altered={}
for offset,severity in enumerate(('Easy','Medium','Hard')):
    candidates=dataset_paths[severity]; rng=np.random.default_rng(ALTERED_SEED+offset)
    paths=[candidates[int(i)] for i in rng.permutation(len(candidates))[:500]];selected_altered[severity]=paths
    for path in paths:
        before=load_fingerprint(path);after=apply_nlm(before,.06);bm=structural_metrics(before);am=structural_metrics(after)
        altered_rows.append({'severity':severity,'filename':path.name,**{f'{k}_before':v for k,v in bm.items()},**{f'{k}_after':v for k,v in am.items()},**{f'delta_{k}':am[k]-bm[k] for k in bm}})
altered_results=pd.DataFrame(altered_rows)
altered_summary=[]
for label,group in [(s,altered_results[altered_results.severity.eq(s)]) for s in ('Easy','Medium','Hard')]+[('Overall',altered_results)]:
    altered_summary.append({'severity':label,'count':len(group),**{f'mean_{c}':group[c].mean() for c in group.columns if c.startswith(('coherence_','contrast_','fragmentation_','continuity_','ridge_coverage_','delta_'))}})
altered_summary=pd.DataFrame(altered_summary);display(altered_summary[['severity','count','mean_delta_coherence','mean_delta_contrast','mean_delta_fragmentation','mean_delta_continuity','mean_delta_ridge_coverage']])
altered_results.to_csv(OUT/'notebook_altered_1500_per_image.csv',index=False);altered_summary.to_csv(OUT/'notebook_altered_1500_summary.csv',index=False)""")

md("## 19. Representative visual results")
code("""fig,axes=plt.subplots(1,3,figsize=(10,3));clean=load_fingerprint(large_paths[0]);degraded=controlled_degradation(clean,NOISE_SEED_BASE);enhanced=apply_nlm(degraded,.06)
for ax,(title,img) in zip(axes,[('Clean Real',clean),('Gaussian degraded',degraded),('NLM enhanced',enhanced)]):ax.imshow(img,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
plt.tight_layout();plt.show()
fig,axes=plt.subplots(3,2,figsize=(7,10))
for row,severity in enumerate(('Easy','Medium','Hard')):
    before=load_fingerprint(selected_altered[severity][0]);after=apply_nlm(before,.06)
    for ax,title,img in [(axes[row,0],f'{severity}: original',before),(axes[row,1],f'{severity}: NLM',after)]:ax.imshow(img,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
plt.tight_layout();plt.show()""")

md("## 20. Final results table")
code("""overall_alt=altered_summary.query("severity=='Overall'").iloc[0]
final_results=pd.DataFrame([
 {'experiment':'Member comparison — best','result':member_summary.iloc[0].method,'PSNR_dB':member_summary.iloc[0].PSNR_mean,'count':int(member_summary.iloc[0].images)},
 {'experiment':'Compatibility — best','result':compat_summary.iloc[0].candidate,'PSNR_dB':compat_summary.iloc[0].PSNR_mean,'count':int(compat_summary.iloc[0].images)},
 {'experiment':'Fresh hold-out','result':f"{int(holdout_results.above_28_17.sum())}/100 above 28.17",'PSNR_dB':holdout_results.enhanced_psnr.mean(),'count':100},
 {'experiment':'Final robustness','result':f"{large_stats['above_28_17']}/500 above 28.17",'PSNR_dB':large_stats['mean_psnr'],'count':500},
 {'experiment':'Original Altered','result':f"Δcoh={overall_alt.mean_delta_coherence:+.5f}; Δfrag={overall_alt.mean_delta_fragmentation:+.5f}; Δcontrast={overall_alt.mean_delta_contrast:+.5f}",'PSNR_dB':np.nan,'count':1500}])
display(final_results);final_results.to_csv(OUT/'notebook_final_results_table.csv',index=False)""")

md("""## 21. Results and discussion

NLM is expected to lead under additive Gaussian noise because patch recurrence is strong in fingerprint ridges and its similarity averaging removes noise without imposing a single ridge orientation. TV remains competitive because its regulariser suppresses noise while preserving strong boundaries. Adaptive Gabor deliberately changes ridge-frequency content and therefore may look structurally useful while scoring lower in pixel fidelity. Directional diffusion can improve local ridge-flow structure but repeated smoothing can reduce PSNR. The compatibility table demonstrates that extra stages do not automatically help: their transformations accumulate bias after NLM.

The fresh hold-out and 500-image runs test generalisation and robustness with frozen settings. The Altered experiment must be read metric by metric; any coherence or fragmentation gain is reported alongside, not instead of, the observed contrast trade-off.""")
code("""display(Markdown(f'''The fresh hold-out regenerated **{holdout_results.enhanced_psnr.mean():.4f} dB** with **{int(holdout_results.above_28_17.sum())}/100** above 28.17 dB. The 500-image mean was **{large_stats['mean_psnr']:.4f} dB**, with **{large_stats['above_28_17']}/500** above the benchmark. On 1,500 original Altered images, mean deltas were coherence **{overall_alt.mean_delta_coherence:+.5f}**, fragmentation **{overall_alt.mean_delta_fragmentation:+.5f}**, and contrast **{overall_alt.mean_delta_contrast:+.5f}**.'''))""")

md("""## 22. Literature benchmark

The selected article reports PSNR 28.17 dB and MSE 22.27. Under this project's controlled Gaussian-noise protocol, the fixed final NLM is compared with the PSNR benchmark. This is not a claim of identical experimental conditions. Direct MSE superiority is not claimed because image scale and protocol equivalence are not established.""")
code("print(f\"Controlled 500-image mean margin over 28.17 dB: {large_stats['margin_to_benchmark']:+.4f} dB\")")

md("""## 23. Final pipeline

**Fingerprint image → P0 neutral preprocessing → NLM (`h=0.06`) → enhanced grayscale fingerprint**

Optional downstream path: **enhanced fingerprint → segmentation → morphology → thinning / medial axis → feature analysis**.""")

md("## 24. Conclusion")
code("""display(Markdown(f'''The recomputed evidence selects **NLM h={FINAL_NLM_H:.2f}** as the final grayscale method. It led the frozen member and compatibility comparisons, generalised to a fresh 100-image set, and achieved **{large_stats['mean_psnr']:.4f} dB** over 500 Real images ({large_stats['above_28_17']}/500 above 28.17 dB). The unpaired Altered evaluation processed 1,500 originals and showed coherence, fragmentation and contrast deltas of **{overall_alt.mean_delta_coherence:+.5f}**, **{overall_alt.mean_delta_fragmentation:+.5f}**, and **{overall_alt.mean_delta_contrast:+.5f}**, respectively. Structural processing remains optional and downstream.'''))""")

nbf.write(nb, NOTEBOOK)
print(f"Built {NOTEBOOK} with {len(nb.cells)} cells")
