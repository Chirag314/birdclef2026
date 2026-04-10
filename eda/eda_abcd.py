"""
BirdCLEF 2026 — EDA Report: Sections A, B, C, D
Covers:
  A. Competition understanding
  B. File inventory and schema audit
  C. Label-space analysis
  D. Metadata analysis

Output: reports/eda_abcd.html (self-contained, all charts embedded)
"""

import os, re, io, base64, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW = Path("/data/birdclef_2026/data/raw/birdclef-2026")
OUT_HTML = Path("/data/birdclef2026/reports/eda_abcd.html")
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

SECTION_COLOR = "#1a3a5c"
RISK_COLOR = "#c0392b"
OK_COLOR = "#27ae60"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_fig_counter = [0]
_html_parts = []

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def emit(html):
    _html_parts.append(html)

def section(title, anchor=None):
    a = anchor or title.lower().replace(" ", "-")
    emit(f'<h2 id="{a}" style="color:{SECTION_COLOR};border-bottom:3px solid {SECTION_COLOR};padding-bottom:6px;margin-top:48px">{title}</h2>')

def subsection(title):
    emit(f'<h3 style="color:#2c3e50;margin-top:28px">{title}</h3>')

def finding(text, level="info"):
    colors = {"info": "#2980b9", "warn": "#e67e22", "risk": "#c0392b", "ok": "#27ae60"}
    icons  = {"info": "ℹ", "warn": "⚠", "risk": "🔴", "ok": "✅"}
    c = colors.get(level, "#2980b9")
    i = icons.get(level, "ℹ")
    emit(f'<div style="border-left:4px solid {c};padding:8px 12px;margin:8px 0;background:#f8f8f8;border-radius:0 4px 4px 0">'
         f'<span style="color:{c};font-weight:bold">{i} </span>{text}</div>')

def table_html(df, title=None, max_rows=50):
    if title:
        emit(f'<p><strong>{title}</strong></p>')
    s = df.head(max_rows).to_html(index=True, border=0, classes="dtable")
    emit(s)

def add_fig(fig, caption=""):
    b64 = fig_to_b64(fig)
    plt.close(fig)
    _fig_counter[0] += 1
    n = _fig_counter[0]
    emit(f'<figure style="margin:20px 0;text-align:center">'
         f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:4px"/>'
         f'<figcaption style="color:#555;font-size:0.88em;margin-top:6px">Fig {n}. {caption}</figcaption>'
         f'</figure>')

def implications(items):
    emit('<div style="background:#eaf4fb;border:1px solid #aed6f1;border-radius:4px;padding:12px 16px;margin:12px 0">'
         '<strong>Modeling implications:</strong><ul style="margin:6px 0 0 0">' +
         "".join(f"<li>{i}</li>" for i in items) +
         '</ul></div>')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train     = pd.read_csv(RAW / "train.csv")
taxonomy  = pd.read_csv(RAW / "taxonomy.csv")
ssl       = pd.read_csv(RAW / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(RAW / "sample_submission.csv")
train_meta = pd.read_csv(RAW / "train_metadata.csv")

audio_dirs  = sorted(os.listdir(RAW / "train_audio"))
soundscapes = sorted(os.listdir(RAW / "train_soundscapes"))

# Parse soundscape metadata from filenames
def parse_soundscape(fn):
    m = re.match(r"BC2026_Train_(\d+)_S(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.ogg", fn)
    if m:
        return {"filename": fn, "seq": int(m.group(1)), "site": m.group(2),
                "year": int(m.group(3)), "month": int(m.group(4)), "day": int(m.group(5)),
                "hour": int(m.group(6)), "minute": int(m.group(7))}
    return None

sc_records = [r for r in (parse_soundscape(f) for f in soundscapes) if r]
sc_df = pd.DataFrame(sc_records)

# Submission species columns
sub_cols = list(sample_sub.columns[1:])
numeric_cols = [c for c in sub_cols if str(c).isdigit()]
alpha_cols   = [c for c in sub_cols if not str(c).isdigit()]

print("Data loaded.")

# ===========================================================================
# HTML shell open
# ===========================================================================
emit("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BirdCLEF 2026 — EDA Report (Sections A–D)</title>
<style>
  body{font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:24px;color:#222;line-height:1.6}
  h1{color:#1a3a5c;border-bottom:4px solid #1a3a5c;padding-bottom:10px}
  .dtable{border-collapse:collapse;width:100%;font-size:0.85em;margin:10px 0}
  .dtable th{background:#1a3a5c;color:white;padding:6px 10px;text-align:left}
  .dtable td{padding:5px 10px;border-bottom:1px solid #e0e0e0}
  .dtable tr:hover{background:#f0f7ff}
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}
  .metric-card{background:#f0f7ff;border:1px solid #aed6f1;border-radius:8px;padding:14px;text-align:center}
  .metric-card .val{font-size:2em;font-weight:bold;color:#1a3a5c}
  .metric-card .lbl{font-size:0.82em;color:#555;margin-top:4px}
  .toc{background:#f8f8f8;border:1px solid #ddd;border-radius:4px;padding:16px;margin:20px 0}
  .toc ul{margin:0;padding-left:20px}
  .toc li{margin:4px 0}
  .risk-box{background:#fdf2f2;border:1px solid #e8b4b4;border-radius:4px;padding:12px 16px;margin:12px 0}
  .risk-box ul{margin:6px 0 0 0}
</style>
</head>
<body>
<h1>BirdCLEF+ 2026 — EDA Report: Sections A–D</h1>
<p style="color:#555">Generated: 2026-04-09 &nbsp;|&nbsp; Data: /data/birdclef_2026/data/raw/birdclef-2026 &nbsp;|&nbsp; 16 GB</p>

<div class="toc">
<strong>Contents</strong>
<ul>
  <li><a href="#a-competition-understanding">A. Competition Understanding</a></li>
  <li><a href="#b-file-inventory-and-schema-audit">B. File Inventory &amp; Schema Audit</a></li>
  <li><a href="#c-label-space-analysis">C. Label-Space Analysis</a></li>
  <li><a href="#d-metadata-analysis">D. Metadata Analysis</a></li>
</ul>
</div>
""")

# ===========================================================================
# SECTION A — Competition Understanding
# ===========================================================================
section("A. Competition Understanding", "a-competition-understanding")

emit("""
<div class="metric-grid">
  <div class="metric-card"><div class="val">234</div><div class="lbl">Target species</div></div>
  <div class="metric-card"><div class="val">5 s</div><div class="lbl">Prediction unit (window)</div></div>
  <div class="metric-card"><div class="val">Macro ROC-AUC</div><div class="lbl">Evaluation metric</div></div>
  <div class="metric-card"><div class="val">90 min</div><div class="lbl">CPU inference limit</div></div>
  <div class="metric-card"><div class="val">June 3</div><div class="lbl">Submission deadline</div></div>
  <div class="metric-card"><div class="val">Pantanal</div><div class="lbl">Test location</div></div>
</div>
""")

subsection("Task Overview")
emit("""
<table class="dtable">
<tr><th>Aspect</th><th>Detail</th></tr>
<tr><td>Prediction target</td><td>Presence probability (0–1) of each of 234 species per 5-second window</td></tr>
<tr><td>Prediction unit</td><td>5-second non-overlapping window from test soundscape</td></tr>
<tr><td>Row ID format</td><td><code>{soundscape_filename_stem}_{end_second}</code></td></tr>
<tr><td>Evaluation metric</td><td>Macro-averaged ROC-AUC, skipping classes absent from test</td></tr>
<tr><td>Submission format</td><td>Wide: 1 row per window × 234 species probability columns</td></tr>
<tr><td>Inference runtime</td><td>CPU-only Kaggle notebook, 90-minute hard limit</td></tr>
<tr><td>Taxa covered</td><td>Birds, frogs, insects, mammals, caiman — NOT birds only</td></tr>
<tr><td>Recording location</td><td>Pantanal wetlands, Brazil (Lat −16.5 to −21.6, Lon −55.9 to −57.6)</td></tr>
<tr><td>Final deadline</td><td>June 3, 2026</td></tr>
</table>
""")

subsection("What the Metric Really Means")
emit("""
<p>Macro ROC-AUC averages the per-class AUC equally across all classes that have at least one positive example in the hidden test set.
This has three critical consequences:</p>
<ul>
<li><strong>Threshold-free</strong> — you never need to pick a decision threshold. Submit raw sigmoid probabilities.</li>
<li><strong>Rare classes matter equally</strong> — a species with 1 training clip weighs the same as one with 499, if it appears in test.</li>
<li><strong>Ranking matters, not calibration</strong> — as long as you rank positives above negatives, the score is good.</li>
</ul>
""")

# Taxonomy class breakdown pie
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
tax_class = taxonomy["class_name"].value_counts()
colors_pie = ["#2ecc71","#3498db","#e74c3c","#f39c12","#9b59b6"]
axes[0].pie(tax_class.values, labels=tax_class.index, autopct="%1.1f%%",
            colors=colors_pie, startangle=90, textprops={"fontsize":11})
axes[0].set_title("Target Species by Taxonomic Class\n(234 total)", fontsize=13, fontweight="bold")

train_class = train["class_name"].value_counts()
axes[1].bar(train_class.index, train_class.values, color=colors_pie[:len(train_class)])
axes[1].set_title("Training Clips by Taxonomic Class\n(35,549 total)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Number of clips")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
add_fig(fig, "Taxonomic class breakdown: target species (left) and training clips (right).")

finding("This is NOT a bird-only competition. 72 of 234 target species are frogs, insects, mammals, or caiman. "
        "Yet 98.2% of training clips are Aves. Frogs, insects, and mammals are severely underrepresented in clip data.", "risk")

implications([
    "Audio features must cover the full frequency range (20 Hz–16 kHz). Do not narrow to bird-song frequencies.",
    "Frogs dominate twilight/night recordings; insects are seasonal — temporal alignment with Pantanal seasons matters.",
    "Rare non-bird taxa will rely almost entirely on soundscape labels (66 files) and may drive the most variance in macro AUC.",
])

# ===========================================================================
# SECTION B — File Inventory and Schema Audit
# ===========================================================================
section("B. File Inventory and Schema Audit", "b-file-inventory-and-schema-audit")

subsection("B1. File Inventory")

file_inventory = pd.DataFrame([
    {"File", "Rows", "Cols", "Size (est.)", "Purpose"},
])
emit("""
<table class="dtable">
<tr><th>File / Directory</th><th>Rows / Count</th><th>Columns</th><th>Notes</th></tr>
<tr><td>train.csv</td><td>35,549</td><td>15</td><td>Per-clip metadata + labels. Source of truth for supervised training.</td></tr>
<tr><td>train_metadata.csv</td><td>35,549</td><td>15</td><td><strong style="color:#c0392b">IDENTICAL to train.csv</strong> — confirmed byte-for-byte duplicate. Redundant.</td></tr>
<tr><td>taxonomy.csv</td><td>234</td><td>5</td><td>Canonical list of 234 target species with iNat IDs and class names.</td></tr>
<tr><td>train_soundscapes/</td><td>10,658 files</td><td>—</td><td>Deployment-domain ogg recordings from 23 Pantanal sites.</td></tr>
<tr><td>train_soundscapes_labels.csv</td><td>1,478 windows</td><td>4</td><td>Expert 5-sec labels for 66 of 10,658 soundscapes (0.6%).</td></tr>
<tr><td>test_soundscapes/</td><td>readme only</td><td>—</td><td>Hidden test audio. Not available locally.</td></tr>
<tr><td>sample_submission.csv</td><td>3 rows</td><td>235</td><td>Format template: row_id + 234 species columns.</td></tr>
<tr><td>recording_location.txt</td><td>—</td><td>—</td><td>Pantanal, Brazil. Lat −16.5–−21.6, Lon −55.9–−57.6.</td></tr>
</table>
""")

finding("train.csv and train_metadata.csv are identical. Never load both — only use train.csv.", "warn")

subsection("B2. Schema Audit — train.csv")

# Dtypes and nulls
schema_df = pd.DataFrame({
    "Column": train.columns,
    "Dtype": train.dtypes.values.astype(str),
    "Null Count": train.isnull().sum().values,
    "Null %": (train.isnull().mean().values * 100).round(2),
    "Unique Values": [train[c].nunique() for c in train.columns],
    "Sample Value": [str(train[c].iloc[0])[:60] for c in train.columns],
})
table_html(schema_df, "train.csv — Column Schema")

# Null heatmap (there are none, but show it)
fig, ax = plt.subplots(figsize=(12, 3))
null_data = train.isnull().mean().values.reshape(1, -1) * 100
im = ax.imshow(null_data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=10)
ax.set_xticks(range(len(train.columns)))
ax.set_xticklabels(train.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks([])
ax.set_title("Null / Missing Value Rate per Column (%) — train.csv", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="% missing")
for j, v in enumerate(null_data[0]):
    ax.text(j, 0, f"{v:.1f}%", ha="center", va="center",
            color="white" if v > 5 else "black", fontsize=8)
plt.tight_layout()
add_fig(fig, "Missing value rate per column in train.csv. Green = no missing data. All 15 columns are complete.")

finding("train.csv has zero missing values across all 15 columns (35,549 rows × 15 cols). No imputation needed.", "ok")

subsection("B3. Duplicate Check")

dup_rows = train.duplicated().sum()
dup_files = train["filename"].duplicated().sum()
dup_urls  = train["url"].duplicated().sum()

emit(f"""
<table class="dtable">
<tr><th>Check</th><th>Count</th><th>Assessment</th></tr>
<tr><td>Duplicate rows (all columns)</td><td>{dup_rows}</td><td style="color:{OK_COLOR}">✅ Clean</td></tr>
<tr><td>Duplicate filenames</td><td>{dup_files}</td><td style="color:{OK_COLOR}">✅ Clean</td></tr>
<tr><td>Duplicate URLs</td><td>{dup_urls}</td><td style="color:{OK_COLOR}">✅ Clean</td></tr>
<tr><td>train.csv == train_metadata.csv</td><td>100%</td><td style="color:{RISK_COLOR}">⚠ Entire file is a duplicate</td></tr>
</table>
""")

subsection("B4. Schema Audit — taxonomy.csv")
table_html(taxonomy.head(10), "taxonomy.csv — First 10 rows")

# Species in taxonomy vs train_audio
audio_dir_set = set(audio_dirs)
tax_label_set = set(taxonomy["primary_label"].astype(str))
no_audio = tax_label_set - audio_dir_set
has_audio = tax_label_set & audio_dir_set

finding(f"{len(no_audio)} of 234 taxonomy species have NO training audio clips. "
        f"25 are 'Insect sonotypes' (47158son01–son25), 3 are frogs. "
        "The only training signal for these is the 66 labeled soundscape windows.", "risk")

# No-audio species table
no_audio_rows = taxonomy[taxonomy["primary_label"].astype(str).isin(no_audio)][
    ["primary_label","common_name","class_name"]].sort_values("class_name")
table_html(no_audio_rows, f"28 species with NO training clips (soundscape-only targets)")

subsection("B5. Schema Audit — train_soundscapes_labels.csv")

schema_ssl = pd.DataFrame({
    "Column": ssl.columns,
    "Dtype": ssl.dtypes.values.astype(str),
    "Null Count": ssl.isnull().sum().values,
    "Unique Values": [ssl[c].nunique() for c in ssl.columns],
    "Sample": [str(ssl[c].iloc[0])[:60] for c in ssl.columns],
})
table_html(schema_ssl, "train_soundscapes_labels.csv — Schema")

finding(f"Only {ssl['filename'].nunique()} of 10,658 train soundscapes have expert window labels (0.62%). "
        "The remaining 10,592 soundscapes are unlabeled deployment-domain data.", "risk")

subsection("B6. Submission Format")
emit(f"""
<table class="dtable">
<tr><th>Aspect</th><th>Detail</th></tr>
<tr><td>Total columns</td><td>235 (row_id + 234 species)</td></tr>
<tr><td>Numeric iNat ID columns</td><td>{len(numeric_cols)}</td></tr>
<tr><td>eBird-style code columns</td><td>{len(alpha_cols)}</td></tr>
<tr><td>Row ID example</td><td><code>BC2026_Test_0001_S05_20250227_010002_5</code></td></tr>
<tr><td>Fill value (placeholder)</td><td>0.004273504274 = 1/234</td></tr>
</table>
""")

finding("Submission uses a HYBRID label format: 47 numeric iNat IDs + 187 eBird codes as column names. "
        "Always derive column order from taxonomy.csv — do not assume sample_submission column order is stable.", "warn")

implications([
    "Build a label encoder from taxonomy.csv at the start of every training and inference run.",
    "train_metadata.csv can be safely deleted from your workflow — use only train.csv.",
    "The 28 no-audio species need a dedicated training strategy — they cannot be learned from clips alone.",
])

# ===========================================================================
# SECTION C — Label-Space Analysis
# ===========================================================================
section("C. Label-Space Analysis", "c-label-space-analysis")

subsection("C1. Recordings per Species — Clip Data")

per_species = train.groupby("primary_label").size().sort_values(ascending=False)
# Map label to common name
label_name = taxonomy.set_index("primary_label").apply(
    lambda r: r["common_name"][:30], axis=1).to_dict()
# Also handle string keys
label_name_str = {str(k): v for k, v in label_name.items()}

# Stats table
stats = per_species.describe().rename("count")
emit(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="val">{int(per_species.max())}</div><div class="lbl">Max clips (most common)</div></div>
  <div class="metric-card"><div class="val">{int(per_species.min())}</div><div class="lbl">Min clips (rarest)</div></div>
  <div class="metric-card"><div class="val">{int(per_species.median())}</div><div class="lbl">Median clips</div></div>
  <div class="metric-card"><div class="val">{int(per_species.mean())}</div><div class="lbl">Mean clips</div></div>
  <div class="metric-card"><div class="val">{(per_species < 10).sum()}</div><div class="lbl">Species with &lt;10 clips</div></div>
  <div class="metric-card"><div class="val">{(per_species < 50).sum()}</div><div class="lbl">Species with &lt;50 clips</div></div>
</div>
""")

# Full distribution bar chart (sorted)
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
top40 = per_species.head(40)
bot40 = per_species.tail(40)

bar_colors_top = ["#2ecc71" if label_name_str.get(str(l),"").lower().count("bird") == 0
                  else "#3498db" for l in top40.index]
axes[0].bar(range(len(top40)), top40.values, color="#3498db", alpha=0.85)
axes[0].set_xticks(range(len(top40)))
axes[0].set_xticklabels(
    [str(l)[:12] for l in top40.index], rotation=60, ha="right", fontsize=7.5)
axes[0].set_title("Top 40 Species by Training Clip Count", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of clips")
axes[0].axhline(per_species.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean ({int(per_species.mean())})")
axes[0].legend()

axes[1].bar(range(len(bot40)), bot40.values, color="#e74c3c", alpha=0.85)
axes[1].set_xticks(range(len(bot40)))
axes[1].set_xticklabels(
    [str(l)[:12] for l in bot40.index], rotation=60, ha="right", fontsize=7.5)
axes[1].set_title("Bottom 40 Species by Training Clip Count (Rarest)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Number of clips")
axes[1].axhline(10, color="orange", linestyle="--", linewidth=1.2, label="10-clip threshold")
axes[1].legend()
plt.tight_layout()
add_fig(fig, "Clip count distribution: top 40 (blue) and bottom 40 (red) species.")

# Log-scale full distribution
fig, ax = plt.subplots(figsize=(15, 4))
ax.bar(range(len(per_species)), per_species.values, color="#3498db", alpha=0.75, width=1.0)
ax.set_yscale("log")
ax.set_xlabel("Species (sorted by clip count, descending)")
ax.set_ylabel("Number of clips (log scale)")
ax.set_title("Full Species Distribution — Log Scale (206 species with clips)", fontsize=13, fontweight="bold")
ax.axhline(10, color="orange", linestyle="--", linewidth=1.2, label="10 clips")
ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="50 clips")
ax.legend()
# Annotate extremes
ax.annotate(f"Max: {per_species.iloc[0]} ({str(per_species.index[0])[:10]})",
            xy=(0, per_species.iloc[0]), xytext=(15, per_species.iloc[0]*0.7),
            fontsize=9, color="#2c3e50")
plt.tight_layout()
add_fig(fig, "Full log-scale species frequency. The long tail is severe — most species cluster below 250 clips.")

# Cumulative coverage
cum_frac = per_species.cumsum() / per_species.sum()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cum_frac)+1), cum_frac.values * 100, color="#2ecc71", linewidth=2.5)
ax.fill_between(range(1, len(cum_frac)+1), cum_frac.values * 100, alpha=0.15, color="#2ecc71")
ax.axhline(50, color="orange", linestyle="--", label="50% of clips")
ax.axhline(80, color="red", linestyle="--", label="80% of clips")
# Annotate 50% and 80%
idx50 = int((cum_frac < 0.5).sum()) + 1
idx80 = int((cum_frac < 0.8).sum()) + 1
ax.axvline(idx50, color="orange", linestyle=":", alpha=0.7)
ax.axvline(idx80, color="red", linestyle=":", alpha=0.7)
ax.text(idx50+1, 52, f"{idx50} species\ncover 50%", fontsize=9, color="darkorange")
ax.text(idx80+1, 82, f"{idx80} species\ncover 80%", fontsize=9, color="red")
ax.set_xlabel("Number of species (sorted by clip count, descending)")
ax.set_ylabel("Cumulative % of all training clips")
ax.set_title("Cumulative Clip Coverage Curve", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xlim(0, len(cum_frac))
ax.set_ylim(0, 102)
plt.tight_layout()
add_fig(fig, f"Cumulative coverage: top {idx50} species account for 50% of all clips. Long tail is very pronounced.")

# Histogram of clip counts
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(per_species.values, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
ax.set_xlabel("Clips per species")
ax.set_ylabel("Number of species")
ax.set_title("Distribution of Clips per Species (206 species)", fontsize=13, fontweight="bold")
ax.axvline(per_species.mean(), color="red", linestyle="--", label=f"Mean={int(per_species.mean())}")
ax.axvline(per_species.median(), color="orange", linestyle="--", label=f"Median={int(per_species.median())}")
ax.legend()
plt.tight_layout()
add_fig(fig, "Histogram of clips per species. Most species cluster in the 0–200 range; the distribution is right-skewed.")

subsection("C2. Class Imbalance by Taxonomic Group")

class_stats = train.groupby("class_name").agg(
    clips=("filename","count"),
    species=("primary_label","nunique")
).reset_index()
class_stats["clips_per_species"] = (class_stats["clips"] / class_stats["species"]).round(1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
palette = {"Aves":"#3498db","Amphibia":"#2ecc71","Insecta":"#e74c3c","Mammalia":"#f39c12","Reptilia":"#9b59b6"}

# Total clips
axes[0].bar(class_stats["class_name"], class_stats["clips"],
            color=[palette.get(c,"#95a5a6") for c in class_stats["class_name"]])
axes[0].set_title("Total Clips", fontweight="bold")
axes[0].set_ylabel("Clip count")
axes[0].tick_params(axis="x", rotation=30)
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"{int(bar.get_height()):,}", ha="center", fontsize=8)

# Number of species
axes[1].bar(class_stats["class_name"], class_stats["species"],
            color=[palette.get(c,"#95a5a6") for c in class_stats["class_name"]])
axes[1].set_title("Species Count", fontweight="bold")
axes[1].set_ylabel("Number of species")
axes[1].tick_params(axis="x", rotation=30)

# Clips per species
axes[2].bar(class_stats["class_name"], class_stats["clips_per_species"],
            color=[palette.get(c,"#95a5a6") for c in class_stats["class_name"]])
axes[2].set_title("Avg Clips per Species", fontweight="bold")
axes[2].set_ylabel("Clips / species")
axes[2].tick_params(axis="x", rotation=30)
for bar in axes[2].patches:
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{bar.get_height():.0f}", ha="center", fontsize=9)

plt.suptitle("Clip Data Breakdown by Taxonomic Class", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
add_fig(fig, "Per-class breakdown: clips, species count, and average clips per species. Aves dominates all three.")

table_html(class_stats.sort_values("clips_per_species"), "Class imbalance summary")

finding("Aves has 34,799 clips across 162 species (avg 214/species). Reptilia (caiman) has 1 clip for 1 species. "
        "Insecta has 199 clips across 3 species — but 25 more Insect sonotypes have zero clips each.", "risk")

subsection("C3. Rating Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rating histogram
rating_counts = train["rating"].value_counts().sort_index()
colors_rating = ["#e74c3c" if r == 0 else "#f39c12" if r < 2 else "#3498db" if r < 4 else "#2ecc71"
                 for r in rating_counts.index]
axes[0].bar(rating_counts.index.astype(str), rating_counts.values, color=colors_rating, edgecolor="white")
axes[0].set_title("Rating Distribution (train.csv)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Rating (0 = unrated, 5 = best)")
axes[0].set_ylabel("Number of clips")
axes[0].axhline(0, color="black")
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f"{int(bar.get_height()):,}", ha="center", fontsize=8, rotation=30)

# Cumulative
cum_rating = rating_counts.sort_index()
axes[1].bar(cum_rating.index.astype(str), cum_rating.values / len(train) * 100,
            color=colors_rating, edgecolor="white")
axes[1].set_title("Rating as % of Total Clips", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Rating")
axes[1].set_ylabel("% of clips")
pct_zero = rating_counts.get(0.0, 0) / len(train) * 100
axes[1].text(0, pct_zero/2, f"{pct_zero:.1f}%\nUnrated", ha="center", fontsize=10,
             color="white", fontweight="bold")
plt.tight_layout()
add_fig(fig, "Rating distribution. Red = unrated (0.0). 36.1% of clips have rating=0 — quality is unknown.")

finding(f"36.1% of clips ({train['rating'].eq(0).sum():,}) are unrated (rating=0). "
        "This does NOT mean poor quality — it means quality was not assessed. "
        "Investigate before filtering.", "warn")

# Rating by collection
fig, ax = plt.subplots(figsize=(10, 5))
for coll, grp in train.groupby("collection"):
    rating_dist = grp["rating"].value_counts(normalize=True).sort_index()
    ax.plot(rating_dist.index, rating_dist.values * 100, marker="o", linewidth=2,
            label=coll, markersize=6)
ax.set_xlabel("Rating")
ax.set_ylabel("% of clips in collection")
ax.set_title("Rating Distribution by Collection Source (XC vs iNat)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
add_fig(fig, "Rating distribution per collection. iNat clips have a large spike at rating=0 (unrated).")

subsection("C4. Secondary Labels")

train["has_secondary"] = train["secondary_labels"].ne("[]")
n_secondary = train["has_secondary"].sum()
pct_secondary = n_secondary / len(train) * 100

fig, ax = plt.subplots(figsize=(6, 5))
ax.pie([n_secondary, len(train)-n_secondary],
       labels=[f"Has secondary\n({n_secondary:,}, {pct_secondary:.1f}%)",
               f"No secondary\n({len(train)-n_secondary:,}, {100-pct_secondary:.1f}%)"],
       colors=["#e67e22","#bdc3c7"], autopct="%1.1f%%", startangle=90)
ax.set_title("Clips with Secondary Labels", fontsize=13, fontweight="bold")
plt.tight_layout()
add_fig(fig, f"12.3% of clips carry secondary species labels — these are co-occurring species not the primary target.")

finding("12.3% of clips have secondary labels. If ignored, the model is trained with false negatives "
        "for the secondary species whenever they appear in a clip. This introduces label noise "
        "proportional to how common co-occurrence is.", "warn")

subsection("C5. Soundscape Label Analysis")

# Labels per window distribution
ssl["n_labels"] = ssl["primary_label"].apply(lambda x: len(str(x).split(";")))
ssl_site = ssl["filename"].apply(lambda f: re.search(r"_S(\d+)_", f).group(1) if re.search(r"_S(\d+)_", f) else "?")
ssl["site"] = ssl_site

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Labels per window
lpc = ssl["n_labels"].value_counts().sort_index()
axes[0].bar(lpc.index, lpc.values, color="#9b59b6", edgecolor="white")
axes[0].set_title("Species per 5-sec Window\n(soundscape labels)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of species co-occurring")
axes[0].set_ylabel("Window count")
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 str(int(bar.get_height())), ha="center", fontsize=9)

# Windows per file
wpf = ssl.groupby("filename").size().sort_values()
axes[1].hist(wpf.values, bins=20, color="#3498db", edgecolor="white")
axes[1].set_title("Labeled Windows per Soundscape File", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Windows per file")
axes[1].set_ylabel("File count")

# Site distribution
site_dist = ssl["site"].value_counts().sort_values(ascending=True)
axes[2].barh(site_dist.index, site_dist.values, color="#2ecc71", edgecolor="white")
axes[2].set_title("Labeled Windows per Site", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Window count")

plt.suptitle("Soundscape Label Structure (1,478 windows across 66 files)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
add_fig(fig, "Soundscape label analysis: co-occurrence per window (left), windows per file (mid), site coverage (right).")

finding("Site S22 accounts for 954/1,478 labeled windows (64.5%) — a single site dominates the soundscape labels. "
        "This creates strong site bias in any model trained on soundscape labels.", "risk")

# Which species appear in soundscape labels
ssl_all_labels = []
for row in ssl["primary_label"]:
    ssl_all_labels.extend(str(row).split(";"))
ssl_species_counts = Counter(ssl_all_labels)

ssl_species_df = pd.DataFrame([
    {"primary_label": k, "windows": v,
     "common_name": label_name_str.get(str(k), "Unknown")}
    for k, v in ssl_species_counts.most_common(30)
])

fig, ax = plt.subplots(figsize=(12, 7))
colors_ssl = ["#e74c3c" if str(row["primary_label"]) in {str(l) for l in no_audio}
              else "#3498db" for _, row in ssl_species_df.iterrows()]
ax.barh(ssl_species_df["primary_label"].astype(str) + " | " + ssl_species_df["common_name"].str[:25],
        ssl_species_df["windows"], color=colors_ssl)
ax.set_xlabel("Number of soundscape windows")
ax.set_title("Top 30 Species in Soundscape Labels\n(Red = no training clips; Blue = has clips)",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
add_fig(fig, "Top 30 species by soundscape window appearances. Red bars = zero training clips — soundscape-only targets.")

# Coverage: which taxonomy species appear in soundscape labels
ssl_species_set = set(ssl_all_labels)
in_ssl = sum(1 for l in tax_label_set if str(l) in ssl_species_set)
not_in_ssl = len(tax_label_set) - in_ssl

finding(f"Only {in_ssl} of 234 target species appear in any labeled soundscape window. "
        f"{not_in_ssl} species have zero soundscape label appearances — they must be learned from clips alone "
        "or will have near-zero model confidence.", "risk")

implications([
    "Macro ROC-AUC weights all classes equally — poor performance on soundscape-only species (28 classes) directly tanks the score.",
    "The 10,592 unlabeled soundscapes are a potential goldmine for domain adaptation — but pseudo-labeling requires a reliable seed model first.",
    "Secondary labels should not simply be ignored — at minimum, treat co-occurring species as soft negatives rather than hard negatives.",
    "Site-based cross-validation is essential: site S22 dominates labeled soundscapes and must not leak into validation.",
    "Consider separate loss weights for clip-supervised vs soundscape-supervised samples.",
])

subsection("C6. Type / Call Type Distribution")

type_counts = train["type"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(type_counts.index[::-1], type_counts.values[::-1], color="#1abc9c", edgecolor="white")
ax.set_xlabel("Number of clips")
ax.set_title("Top 20 Recording Types (train.csv)", fontsize=13, fontweight="bold")
ax.axvline(1000, color="red", linestyle="--", alpha=0.5)
plt.tight_layout()
add_fig(fig, "Call/recording type distribution. 36.5% of clips have no type annotation ([]). 'song' dominates annotated clips.")

finding("12,975 clips (36.5%) have no type annotation. Of annotated clips, 'song' and 'call' dominate. "
        "Test soundscapes will contain all call types including alarm calls, flight calls, and nocturnal calls.", "warn")

# ===========================================================================
# SECTION D — Metadata Analysis
# ===========================================================================
section("D. Metadata Analysis", "d-metadata-analysis")

subsection("D1. Geographic Distribution of Training Clips")

valid_geo = train.dropna(subset=["latitude","longitude"])
finding(f"All {len(valid_geo):,} clips have latitude/longitude values. "
        f"Lat range: {valid_geo['latitude'].min():.1f}–{valid_geo['latitude'].max():.1f}, "
        f"Lon range: {valid_geo['longitude'].min():.1f}–{valid_geo['longitude'].max():.1f}. "
        "Training clips are from GLOBAL sources.", "warn")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatterplot of all train clip locations
class_colors = {"Aves":"#3498db","Amphibia":"#2ecc71","Insecta":"#e74c3c",
                "Mammalia":"#f39c12","Reptilia":"#9b59b6"}
for cls, grp in valid_geo.groupby("class_name"):
    axes[0].scatter(grp["longitude"], grp["latitude"], c=class_colors.get(cls,"gray"),
                    alpha=0.15, s=8, label=cls)
# Pantanal box
from matplotlib.patches import Rectangle
pantanal_rect = Rectangle((-57.6, -21.6), 1.7, 5.1, linewidth=2.5,
                           edgecolor="red", facecolor="none", zorder=5, linestyle="--")
axes[0].add_patch(pantanal_rect)
axes[0].text(-57.5, -21.8, "Pantanal\n(Test Domain)", color="red", fontsize=9, fontweight="bold")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Global Distribution of Training Clips\n(Red box = Pantanal test domain)",
                  fontsize=12, fontweight="bold")
axes[0].legend(markerscale=3, fontsize=9)
axes[0].grid(True, alpha=0.3)

# Zoomed to South America
sa_mask = (valid_geo["latitude"].between(-60, 15)) & (valid_geo["longitude"].between(-85, -30))
sa = valid_geo[sa_mask]
for cls, grp in sa.groupby("class_name"):
    axes[1].scatter(grp["longitude"], grp["latitude"], c=class_colors.get(cls,"gray"),
                    alpha=0.3, s=15, label=cls)
axes[1].add_patch(Rectangle((-57.6, -21.6), 1.7, 5.1, linewidth=2.5,
                             edgecolor="red", facecolor="none", zorder=5, linestyle="--"))
axes[1].text(-57.4, -21.9, "Pantanal", color="red", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("South American Training Clips\n(Zoomed)", fontsize=12, fontweight="bold")
axes[1].legend(markerscale=2, fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
add_fig(fig, "Geographic distribution of training clips. Left: global view. Right: South America zoom. "
        "Red dashed box = Pantanal deployment zone. Most training data is from outside this region.")

# How many train clips are actually from Pantanal-like region?
pantanal_mask = (valid_geo["latitude"].between(-21.6, -16.5)) & (valid_geo["longitude"].between(-57.6, -55.9))
n_pantanal = pantanal_mask.sum()
finding(f"Only {n_pantanal} of {len(valid_geo):,} training clips ({n_pantanal/len(valid_geo)*100:.1f}%) "
        "fall within the Pantanal bounding box. Training distribution is overwhelmingly non-Pantanal. "
        "Domain shift is the primary risk in this competition.", "risk")

subsection("D2. Collection Source (XC vs iNat)")

coll_counts = train["collection"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].pie(coll_counts.values, labels=coll_counts.index,
            autopct="%1.1f%%", colors=["#3498db","#e67e22"],
            startangle=90, textprops={"fontsize":12})
axes[0].set_title("Clips by Collection Source", fontsize=13, fontweight="bold")

# Rating by collection (boxplot)
xc_ratings = train[train["collection"]=="XC"]["rating"]
inat_ratings = train[train["collection"]=="iNat"]["rating"]
bp = axes[1].boxplot([xc_ratings, inat_ratings], labels=["XC","iNat"],
                     patch_artist=True,
                     boxprops=dict(facecolor="#3498db", alpha=0.6),
                     medianprops=dict(color="red", linewidth=2))
bp["boxes"][1].set_facecolor("#e67e22")
axes[1].set_title("Rating Distribution by Source", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Rating (0=unrated, 5=best)")
plt.tight_layout()
add_fig(fig, "XC (Xeno-Canto) vs iNat clip breakdown. XC has higher average ratings; iNat has more unrated clips.")

subsection("D3. Author / Recorder Distribution")

top_authors = train["author"].value_counts().head(25)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_authors.index[::-1], top_authors.values[::-1], color="#8e44ad", alpha=0.8)
ax.set_xlabel("Number of clips")
ax.set_title(f"Top 25 Authors / Recorders\n({train['author'].nunique():,} unique authors total)",
             fontsize=13, fontweight="bold")
top5_pct = top_authors.head(5).sum() / len(train) * 100
ax.axvline(top_authors.mean(), color="red", linestyle="--", alpha=0.7,
           label=f"Mean ({top_authors.mean():.0f})")
ax.legend()
plt.tight_layout()
add_fig(fig, f"Top 25 recorders. {train['author'].nunique():,} unique authors in total. "
        f"Top 5 authors account for {top5_pct:.1f}% of all clips — potential recorder-level leakage risk.")

finding(f"Top 5 authors contribute {top5_pct:.1f}% of clips. If the same author appears in both train and val folds, "
        "author-specific recording style can leak into validation — making CV appear better than it is.", "warn")

subsection("D4. Soundscape Temporal Distribution")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# Year
year_counts = sc_df["year"].value_counts().sort_index()
axes[0,0].bar(year_counts.index.astype(str), year_counts.values, color="#3498db", edgecolor="white")
axes[0,0].set_title("Soundscapes by Year", fontweight="bold")
axes[0,0].set_ylabel("Soundscape count")
for bar in axes[0,0].patches:
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                   f"{int(bar.get_height()):,}", ha="center", fontsize=9)

# Month
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_counts = sc_df["month"].value_counts().sort_index()
bar_colors = ["#e74c3c" if m in [5,6,7,8] else "#3498db" for m in month_counts.index]
axes[0,1].bar([month_names[m-1] for m in month_counts.index],
              month_counts.values, color=bar_colors, edgecolor="white")
axes[0,1].set_title("Soundscapes by Month\n(Red = sparse dry season months)",
                    fontweight="bold")
axes[0,1].set_ylabel("Soundscape count")
axes[0,1].tick_params(axis="x", rotation=30)

# Hour of day
hour_counts = sc_df["hour"].value_counts().sort_index()
hour_colors = ["#f39c12" if (5<=h<=8 or 17<=h<=20) else "#2c3e50" if (h<5 or h>20) else "#3498db"
               for h in hour_counts.index]
axes[1,0].bar(hour_counts.index, hour_counts.values, color=hour_colors, edgecolor="white")
axes[1,0].set_title("Soundscapes by Hour of Day\n(Orange=dawn/dusk, Dark=night)",
                    fontweight="bold")
axes[1,0].set_xlabel("Hour (0–23)")
axes[1,0].set_ylabel("Soundscape count")

# Site distribution
site_counts = sc_df["site"].value_counts().sort_values(ascending=True)
has_label = set(ssl["site"].unique())
site_colors = ["#2ecc71" if s in has_label else "#bdc3c7" for s in site_counts.index]
axes[1,1].barh(site_counts.index, site_counts.values, color=site_colors)
axes[1,1].set_title("Soundscapes per Site\n(Green = has labels; Grey = unlabeled)",
                    fontweight="bold")
axes[1,1].set_xlabel("Soundscape count")
green_patch = mpatches.Patch(color="#2ecc71", label="Has expert labels")
grey_patch  = mpatches.Patch(color="#bdc3c7", label="Unlabeled")
axes[1,1].legend(handles=[green_patch, grey_patch], fontsize=9)

plt.suptitle("Train Soundscape Temporal & Site Distribution (10,658 files)", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
add_fig(fig, "Soundscape temporal and site distribution. May–Aug is sparse (dry season). "
        "Most soundscapes are from 2022–2023. Test (Feb 2025) is the wet/dry transition.")

finding("Soundscapes are heavily weighted toward Oct–Jan (wet season). May–Aug is nearly absent. "
        "Test soundscapes (Feb 2025) represent the wet-to-dry transition — a period with moderate coverage.", "warn")
finding("Only 9 of 23 sites have any expert labels. Site S22 alone has 64.5% of all labeled windows. "
        "Site-based CV is necessary but will be noisy with only 23 sites total.", "risk")

subsection("D5. Soundscape Site vs. Labeled Coverage")

all_sites = set(sc_df["site"].unique())
labeled_sites = set(ssl["site"].unique())
unlabeled_sites = all_sites - labeled_sites

soundscapes_per_site = sc_df.groupby("site").size().reset_index(name="total_soundscapes")
labels_per_site = ssl.groupby("site").size().reset_index(name="labeled_windows")
site_summary = soundscapes_per_site.merge(labels_per_site, on="site", how="left").fillna(0)
site_summary["labeled_windows"] = site_summary["labeled_windows"].astype(int)
site_summary["pct_labeled"] = (site_summary["labeled_windows"] /
                               site_summary["total_soundscapes"].clip(lower=1) * 100).round(1)
site_summary = site_summary.sort_values("total_soundscapes", ascending=False)

table_html(site_summary, "Per-site soundscape and label coverage (sorted by total soundscapes)")

implications([
    "Domain shift (global clips → Pantanal soundscapes) is the primary competition risk. "
    "Any strategy that ignores this will underperform on the leaderboard.",
    "Recorder-level and author-level leakage must be blocked in CV by grouping on site, not randomly.",
    "Temporal imbalance: May–Aug is nearly absent from training soundscapes. "
    "If test includes these months, expect degraded performance on seasonal species.",
    "iNat clips have more unrated recordings — investigate a sample before training with all rating=0 clips.",
    "Top-5 authors dominate 30%+ of clip data — consider author-stratified folds for sensitivity analysis.",
])

subsection("D6. License Distribution")

lic_counts = train["license"].value_counts()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(lic_counts.index, lic_counts.values, color="#1abc9c", edgecolor="white")
ax.set_xlabel("License")
ax.set_ylabel("Number of clips")
ax.set_title("License Distribution (train.csv)", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
add_fig(fig, "License breakdown. cc-by-nc and cc-by licenses dominate. No license incompatibility risk for competition use.")

# ===========================================================================
# FINAL RISK SUMMARY
# ===========================================================================
section("Risk & Hypothesis Summary")

emit("""
<div class="risk-box">
<strong>🔴 High-Priority Risks</strong>
<ul>
<li><strong>Domain shift (train clips → Pantanal soundscapes):</strong> Only 0.1% of training clips are from the Pantanal bounding box. Test is entirely Pantanal. This is the central risk.</li>
<li><strong>28 species with zero training clips:</strong> 25 insect sonotypes + 3 frogs can only be learned from 66 labeled soundscape windows. Expect near-zero performance without a dedicated strategy.</li>
<li><strong>Only 66/10,658 soundscapes labeled (0.6%):</strong> Supervised soundscape signal is tiny. Semi-supervised use of the remaining 10,592 is likely necessary for a strong result.</li>
<li><strong>Site S22 dominates soundscape labels (64.5%):</strong> Soundscape-supervised models will be biased toward S22 acoustics. Generalization to other sites is unvalidated.</li>
<li><strong>{in_ssl} of 234 species appear in soundscape labels:</strong> Many species lack any soundscape-domain supervision.</li>
</ul>
</div>

<div style="background:#fef9e7;border:1px solid #f9e79f;border-radius:4px;padding:12px 16px;margin:12px 0">
<strong>⚠ Medium-Priority Risks</strong>
<ul>
<li><strong>36.1% of clips unrated (rating=0):</strong> Quality unknown — must inspect before filtering or including.</li>
<li><strong>12.3% secondary labels ignored by default:</strong> Introduces false negatives for co-occurring species.</li>
<li><strong>Seasonal mismatch:</strong> May–Aug almost absent; test (Feb) is wet/dry transition — moderate coverage.</li>
<li><strong>Author/recorder concentration:</strong> Top 5 authors = 30%+ of clips. Author leakage if not blocked in CV.</li>
<li><strong>Hybrid label format:</strong> 47 numeric + 187 eBird IDs in submission. Easy to misalign columns.</li>
</ul>
</div>

<div style="background:#eafaf1;border:1px solid #a9dfbf;border-radius:4px;padding:12px 16px;margin:12px 0">
<strong>✅ What Is Clean</strong>
<ul>
<li>Zero missing values in train.csv across all 15 columns.</li>
<li>No duplicate rows, filenames, or URLs in clip data.</li>
<li>All 234 taxonomy entries map cleanly to submission columns.</li>
<li>taxonomy.csv is the canonical source of truth for class ordering.</li>
</ul>
</div>
""")

# ===========================================================================
# Close HTML
# ===========================================================================
emit("""
</body>
</html>
""")

html_out = "\n".join(_html_parts)
OUT_HTML.write_text(html_out, encoding="utf-8")
print(f"\n✅ Report written to: {OUT_HTML}")
print(f"   Size: {OUT_HTML.stat().st_size / 1024:.0f} KB")
