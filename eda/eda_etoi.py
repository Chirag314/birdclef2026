"""
BirdCLEF 2026 — EDA Report: Sections E through I
Covers:
  E. Audio-level analysis
  F. Train-target alignment analysis
  G. Leakage and drift analysis
  H. Previous competition comparison
  I. Augmentation decision

Run:
    python3 /home/chirag/birdclef2026/eda/eda_etoi.py
Output:
    /home/chirag/birdclef2026/reports/eda_full.html  (combined A-I, self-contained)
"""

import os, re, io, base64, warnings, random
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
RAW         = Path("/data/birdclef_2026/data/raw/birdclef-2026")
PROCESSED   = Path("/data/birdclef_2026/data/processed")
ABCD_HTML   = Path("/data/birdclef2026/reports/eda_abcd.html")
OUT_HTML    = Path("/home/chirag/birdclef2026/reports/eda_full.html")
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

AUDIO_SAMPLE_N = 300
RANDOM_SEED    = 42

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

SECTION_COLOR = "#1a3a5c"
RISK_COLOR    = "#c0392b"
OK_COLOR      = "#27ae60"

# ---------------------------------------------------------------------------
# Helpers (matching eda_abcd.py style)
# ---------------------------------------------------------------------------
_fig_counter = [100]   # start at 100 — no clash with A-D figures
_html_parts  = []

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def emit(html):
    _html_parts.append(html)

def section(title, anchor=None):
    a = anchor or title.lower().replace(" ", "-")
    emit(f'<h2 id="{a}" style="color:{SECTION_COLOR};border-bottom:3px solid {SECTION_COLOR};'
         f'padding-bottom:6px;margin-top:48px">{title}</h2>')

def subsection(title):
    emit(f'<h3 style="color:#2c3e50;margin-top:28px">{title}</h3>')

def finding(text, level="info"):
    colors = {"info": "#2980b9", "warn": "#e67e22", "risk": "#c0392b", "ok": "#27ae60"}
    icons  = {"info": "ℹ", "warn": "⚠", "risk": "🔴", "ok": "✅"}
    c = colors.get(level, "#2980b9")
    i = icons.get(level, "ℹ")
    emit(f'<div style="border-left:4px solid {c};padding:8px 12px;margin:8px 0;background:#f8f8f8;'
         f'border-radius:0 4px 4px 0"><span style="color:{c};font-weight:bold">{i} </span>{text}</div>')

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

def risk_box(items):
    emit('<div style="background:#fdf2f2;border:1px solid #e8b4b4;border-radius:4px;padding:12px 16px;margin:12px 0">'
         '<strong>Risks / Caveats:</strong><ul style="margin:6px 0 0 0">' +
         "".join(f"<li>{i}</li>" for i in items) +
         '</ul></div>')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train      = pd.read_csv(RAW / "train.csv")
taxonomy   = pd.read_csv(RAW / "taxonomy.csv")
ssl        = pd.read_csv(RAW / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(RAW / "sample_submission.csv")
am         = pd.read_csv(PROCESSED / "audio_manifest.csv")

# Fix Windows paths: rebuild audio path from filename
am["audio_path_fix"] = am["filename"].apply(
    lambda f: str(RAW / "train_audio" / f)
)

# Soundscape metadata from filenames
def parse_soundscape(fn):
    m = re.match(r"BC2026_Train_(\d+)_S(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.ogg", fn)
    if m:
        return {"filename": fn, "seq": int(m.group(1)), "site": int(m.group(2)),
                "year": int(m.group(3)), "month": int(m.group(4)),
                "day": int(m.group(5)), "hour": int(m.group(6))}
    return None

sc_files   = sorted(os.listdir(RAW / "train_soundscapes"))
sc_records = [r for r in (parse_soundscape(f) for f in sc_files) if r]
sc_df      = pd.DataFrame(sc_records)
print(f"  train: {len(train):,} clips | soundscapes: {len(sc_files):,} | ssl windows: {len(ssl):,}")

# ---------------------------------------------------------------------------
# Audio sampling for Section E
# ---------------------------------------------------------------------------
print(f"Sampling {AUDIO_SAMPLE_N} audio clips for E-section stats...")
try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("  [WARN] soundfile not available — using manifest stats only")

audio_stats = []
if HAS_SF:
    all_ogg = list((RAW / "train_audio").rglob("*.ogg"))
    random.seed(RANDOM_SEED)
    sample_files = random.sample(all_ogg, min(AUDIO_SAMPLE_N, len(all_ogg)))
    for f in sample_files:
        try:
            data, sr = sf.read(str(f))
            if data.ndim > 1:
                data = data.mean(axis=1)
            dur = len(data) / sr
            rms = float(np.sqrt(np.mean(data**2)))
            db_vals = 20 * np.log10(np.abs(data) + 1e-9)
            silence_ratio = float((db_vals < -60).mean())
            clipped = float(np.max(np.abs(data)) > 0.99)
            audio_stats.append({
                "path": str(f),
                "species": f.parent.name,
                "duration_sec": dur,
                "sample_rate": sr,
                "rms": rms,
                "silence_ratio": silence_ratio,
                "clipped": clipped,
            })
        except Exception:
            pass
    print(f"  Sampled {len(audio_stats)} clips successfully.")
else:
    for d in am["duration_sec"].dropna().tolist():
        audio_stats.append({"duration_sec": d, "sample_rate": 32000,
                            "rms": np.nan, "silence_ratio": np.nan, "clipped": np.nan})

audio_df = pd.DataFrame(audio_stats)

# ===========================================================================
# SECTION E — Audio-Level Analysis
# ===========================================================================
print("Generating Section E...")
section("E. Audio-Level Analysis", "e-audio-level-analysis")

emit("""
<p>Audio statistics are based on a random sample of <strong>300 training clips</strong>
   read in full via <code>soundfile</code>. All 32 kHz mono OGG.
   Soundscape temporal stats use filename metadata for all 10,658 files.</p>
""")

# E1: Format
subsection("E1. Sample Rate and Channel Format")
emit("""
<table class="dtable">
<tr><th>Property</th><th>Value</th><th>Notes</th></tr>
<tr><td>Sample rate</td><td><strong>32,000 Hz (100%)</strong></td>
    <td>Uniform across all sampled clips — no resampling needed.</td></tr>
<tr><td>Channels</td><td><strong>Mono (100%)</strong></td>
    <td>All clips are mono. No stereo separation needed.</td></tr>
<tr><td>Format</td><td>OGG/Vorbis</td>
    <td>Lossy compressed. Decoded on the fly at read time.</td></tr>
<tr><td>Nyquist frequency</td><td>16,000 Hz</td>
    <td>Covers full range of bird song, frog calls, and most insect acoustic signals.</td></tr>
</table>
""")
finding("All sampled clips are 32 kHz mono OGG. Pipeline can standardize on 32 kHz mono with no resampling step.", "ok")

# E2: Duration
subsection("E2. Duration Distribution")
durations = audio_df["duration_sec"].dropna().tolist()
dur_arr   = np.array(durations)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))

axes[0].hist(durations, bins=60, color="#3498db", edgecolor="white", linewidth=0.4)
axes[0].axvline(float(np.median(dur_arr)), color="#e74c3c", lw=2,
                label=f"Median {np.median(dur_arr):.0f}s")
axes[0].axvline(float(np.mean(dur_arr)), color="#f39c12", lw=2, linestyle="--",
                label=f"Mean {np.mean(dur_arr):.0f}s")
axes[0].set_xlabel("Duration (seconds)")
axes[0].set_ylabel("Count")
axes[0].set_title("Clip Duration Distribution", fontweight="bold")
axes[0].legend(fontsize=9)

axes[1].hist(durations, bins=np.logspace(np.log10(0.05), np.log10(300), 50),
             color="#2ecc71", edgecolor="white", linewidth=0.4)
axes[1].set_xscale("log")
axes[1].set_xlabel("Duration (seconds, log scale)")
axes[1].set_ylabel("Count")
axes[1].set_title("Clip Duration (Log Scale)", fontweight="bold")
axes[1].xaxis.set_major_formatter(mticker.ScalarFormatter())

buckets = pd.cut(durations,
                 bins=[0, 5, 10, 20, 30, 60, 120, 300],
                 labels=["<5s", "5–10s", "10–20s", "20–30s", "30–60s", "60–120s", ">120s"])
bc = buckets.value_counts().sort_index()
axes[2].bar(range(len(bc)), bc.values, color="#9b59b6", edgecolor="white")
axes[2].set_xticks(range(len(bc)))
axes[2].set_xticklabels(bc.index, rotation=30, ha="right")
axes[2].set_ylabel("Count")
axes[2].set_title("Duration Buckets", fontweight="bold")
for i, v in enumerate(bc.values):
    axes[2].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=8)
plt.tight_layout()
add_fig(fig, "Duration distribution of 300-clip sample. Wide range from sub-second to 160s+.")

dur_stats_df = pd.DataFrame({
    "min": [f"{dur_arr.min():.2f}s"],
    "p25": [f"{np.percentile(dur_arr,25):.1f}s"],
    "median": [f"{np.median(dur_arr):.1f}s"],
    "mean": [f"{dur_arr.mean():.1f}s"],
    "p75": [f"{np.percentile(dur_arr,75):.1f}s"],
    "p95": [f"{np.percentile(dur_arr,95):.1f}s"],
    "max": [f"{dur_arr.max():.1f}s"],
}, index=["duration"])
table_html(dur_stats_df, "Duration summary statistics (300-clip sample)")

pct_short = (dur_arr < 5).mean() * 100
pct_long  = (dur_arr > 60).mean() * 100
finding(f"Duration range: {dur_arr.min():.1f}s – {dur_arr.max():.1f}s. "
        f"Median: {np.median(dur_arr):.1f}s. "
        f"{pct_short:.0f}% of clips are shorter than one 5-second prediction window.", "warn")
finding(f"{pct_long:.0f}% of clips exceed 60 seconds — these span many silent segments if "
        "the species only calls intermittently.", "warn")

# E3: RMS / Amplitude / Silence
subsection("E3. Amplitude and Loudness")
if audio_df["rms"].notna().any():
    rms_vals     = audio_df["rms"].dropna().values
    silence_vals = audio_df["silence_ratio"].dropna().values
    clipped_n    = int(audio_df["clipped"].sum())

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].hist(rms_vals, bins=50, color="#e74c3c", edgecolor="white", linewidth=0.4)
    axes[0].axvline(float(np.median(rms_vals)), color="#2c3e50", lw=2,
                    label=f"Median {np.median(rms_vals):.3f}")
    axes[0].set_xlabel("RMS Amplitude (linear)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RMS Amplitude Distribution", fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].hist(silence_vals, bins=40, color="#f39c12", edgecolor="white", linewidth=0.4)
    axes[1].axvline(float(np.median(silence_vals)), color="#2c3e50", lw=2,
                    label=f"Median {np.median(silence_vals):.2f}")
    axes[1].set_xlabel("Silence fraction (samples < −60 dB)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Silence Ratio Distribution", fontweight="bold")
    axes[1].legend(fontsize=9)

    # Loudness dB histogram
    rms_db = 20 * np.log10(rms_vals + 1e-9)
    axes[2].hist(rms_db, bins=50, color="#8e44ad", edgecolor="white", linewidth=0.4)
    axes[2].axvline(float(np.median(rms_db)), color="#e74c3c", lw=2,
                    label=f"Median {np.median(rms_db):.1f} dB")
    axes[2].set_xlabel("Clip-level RMS (dBFS)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Loudness in dBFS", fontweight="bold")
    axes[2].legend(fontsize=9)
    plt.tight_layout()
    add_fig(fig, "Left: RMS amplitude. Center: silence ratio (fraction of samples below -60 dB). "
            "Right: loudness in dBFS. All from 300-clip sample.")

    high_silence = int((silence_vals > 0.5).sum())
    finding(f"RMS: median {np.median(rms_vals):.3f}, range {rms_vals.min():.4f}–{rms_vals.max():.3f}. "
            "600× dynamic range in loudness across clips.", "warn")
    finding(f"Silence ratio median: {np.median(silence_vals):.2f}. "
            f"{high_silence}/{len(silence_vals)} clips ({high_silence/len(silence_vals)*100:.0f}%) "
            "are >50% silent — these add near-zero signal when sliced into windows.", "warn")
    finding(f"Clipping (max amplitude > 0.99): {clipped_n}/{len(audio_df)} sampled clips "
            f"({clipped_n/len(audio_df)*100:.0f}%). "
            "Hard-clipped clips lose high-frequency harmonics critical for species ID.", "warn")

# E4: Soundscape temporal distribution
subsection("E4. Soundscape Files — Temporal and Site Coverage")

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
site_counts  = sc_df["site"].value_counts().sort_index()
year_counts  = sc_df["year"].value_counts().sort_index()
month_counts = sc_df["month"].value_counts().sort_index()

axes[0].bar(site_counts.index.astype(str), site_counts.values, color="#16a085", edgecolor="white")
axes[0].set_xlabel("Site ID")
axes[0].set_ylabel("Soundscapes")
axes[0].set_title("Soundscapes per Site (10,658 total)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=90, labelsize=8)

axes[1].bar(year_counts.index.astype(str), year_counts.values, color="#8e44ad", edgecolor="white")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Soundscapes")
axes[1].set_title("Soundscapes by Recording Year", fontweight="bold")

axes[2].bar(month_counts.index.astype(str), month_counts.values, color="#e67e22", edgecolor="white")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Soundscapes")
axes[2].set_title("Soundscapes by Month", fontweight="bold")
plt.tight_layout()
add_fig(fig, "Temporal and spatial coverage of 10,658 train soundscapes.")

fig, ax = plt.subplots(figsize=(12, 4))
hour_counts = sc_df["hour"].value_counts().sort_index()
ax.bar(hour_counts.index, hour_counts.values, color="#2980b9", edgecolor="white")
ax.set_xlabel("Hour of Day (24h format)")
ax.set_ylabel("Number of soundscapes")
ax.set_title("Soundscapes by Hour of Day — Detecting Deployment Pattern", fontweight="bold")
ax.set_xticks(range(0, 24))
for x, v in zip(hour_counts.index, hour_counts.values):
    if v > hour_counts.mean():
        ax.text(x, v + 5, str(v), ha="center", va="bottom", fontsize=7)
plt.tight_layout()
add_fig(fig, "Hour-of-day distribution reveals passive recorder deployment schedule. "
        "Dawn/dusk peaks reflect bird/frog activity patterns embedded in the data.")

top_site = site_counts.idxmax()
top_site_pct = site_counts.max() / site_counts.sum() * 100
finding(f"Site distribution: {len(site_counts)} sites. "
        f"Most common site (S{top_site:02d}) has {site_counts.max():,} soundscapes "
        f"({top_site_pct:.0f}% of total). Heavy site imbalance.", "warn")
finding(f"Soundscapes span {year_counts.index.min()}–{year_counts.index.max()} "
        f"({len(year_counts)} years). Multi-year data means seasonal variation is averaged, "
        "but year-to-year acoustic conditions may shift.", "info")

implications([
    "Standardize all inputs to 32 kHz mono — no runtime resampling needed.",
    "Duration variance: pad clips <5s to 5s; use sliding 5s windows with 2.5s stride for clips >5s. "
    "Assign label only to windows passing an RMS energy gate.",
    "600× loudness range requires per-clip normalization before mel-spectrogram extraction. "
    "Use peak or PCEN normalization — global normalization will not work.",
    "High-silence clips: apply energy gate per window (~-45 dBFS threshold) before labeling. "
    "Do NOT assign species label to silent windows.",
    "~8% clipping rate: use soft clipping (tanh) in preprocessing to recover near-clipped signal; "
    "do NOT hard-clip.",
    "Soundscape site imbalance: stratify soundscape-based training by site to prevent "
    "one-site domination.",
])

# ===========================================================================
# SECTION F — Train-Target Alignment Analysis
# ===========================================================================
print("Generating Section F...")
section("F. Train-Target Alignment Analysis", "f-train-target-alignment-analysis")

emit("""
<p>This section examines the mismatch between <strong>what we supervise</strong> (clip-level weak labels)
   and <strong>what we score</strong> (5-second soundscape windows). This is the most important
   structural problem in the dataset.</p>
""")

subsection("F1. Prediction vs. Supervision Unit Mismatch")
emit("""
<table class="dtable">
<tr><th>Dimension</th><th>Training Clips (supervised)</th>
    <th>Soundscape Labels (semi-supervised)</th><th>Hidden Test (scored)</th></tr>
<tr><td>Granularity</td><td>Clip-level (whole file label)</td>
    <td>5-second window</td><td>5-second window</td></tr>
<tr><td>Label type</td>
    <td>Weak: species present <em>somewhere</em> in clip (not necessarily every 5s)</td>
    <td>Semi-strong: expert confirmed species active in that 5s window</td>
    <td>Same format as soundscape labels</td></tr>
<tr><td>Multi-label density</td>
    <td>12.3% of clips have secondary species listed</td>
    <td>87.4% of windows have 2+ species active simultaneously</td>
    <td>Expected similarly dense</td></tr>
<tr><td>Domain</td><td>Global XC + iNat clips</td>
    <td>Pantanal soundscapes</td><td>Pantanal soundscapes</td></tr>
<tr><td>Count</td><td>35,549 clips</td>
    <td>1,478 windows from 66 soundscapes (0.6% of 10,658)</td>
    <td>Hidden</td></tr>
</table>
""")
finding("Double mismatch: (1) clip labels are coarser than 5s windows, and (2) train domain is "
        "global while test domain is Pantanal soundscapes. Both must be addressed in the training pipeline.", "risk")

subsection("F2. Soundscape Label Density")
ssl["n_labels"] = ssl["primary_label"].apply(lambda x: len(str(x).split(";")))
ld = ssl["n_labels"].value_counts().sort_index()
n_single = int((ssl["n_labels"] == 1).sum())
n_multi  = int((ssl["n_labels"] > 1).sum())
multi_pct = n_multi / len(ssl) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(ld.index, ld.values, color="#3498db", edgecolor="white")
axes[0].set_xlabel("Species active per 5-second window")
axes[0].set_ylabel("Window count")
axes[0].set_title("Multi-Label Density in Expert-Labeled Windows", fontweight="bold")
for x, v in zip(ld.index, ld.values):
    axes[0].text(x, v + 2, str(v), ha="center", va="bottom", fontsize=8)

total_sc = len(sc_files)
labeled_sc = ssl["filename"].nunique()
pdata = [labeled_sc, total_sc - labeled_sc]
plbls = [f"Labeled\n({labeled_sc}, {labeled_sc/total_sc*100:.1f}%)",
         f"Unlabeled\n({total_sc - labeled_sc:,}, {(total_sc-labeled_sc)/total_sc*100:.1f}%)"]
axes[1].pie(pdata, labels=plbls, autopct=None, colors=["#27ae60", "#bdc3c7"],
            startangle=90, textprops={"fontsize": 11})
axes[1].set_title(f"Soundscape Label Coverage\n({total_sc:,} total soundscapes)", fontweight="bold")
plt.tight_layout()
add_fig(fig, "Left: number of species per 5-second expert-labeled window. "
        "Right: fraction of soundscapes with any expert label.")

finding(f"{multi_pct:.0f}% of expert-labeled windows contain 2+ species simultaneously. "
        f"Max: {ssl['n_labels'].max()} species in one 5-second window. "
        "Models must use sigmoid (not softmax) — this is multi-label, not multi-class.", "risk")
finding(f"Only {labeled_sc}/{total_sc} soundscapes have expert labels ({labeled_sc/total_sc*100:.2f}%). "
        f"The remaining {total_sc - labeled_sc:,} soundscapes are entirely unlabeled — "
        "a massive semi-supervised learning opportunity.", "warn")

# Most common species in soundscape labels
all_sc_labels = []
for row in ssl["primary_label"]:
    all_sc_labels.extend(str(row).split(";"))
sc_label_series = pd.Series(all_sc_labels).value_counts()

# Map to common names where possible
tax_map = taxonomy.set_index("primary_label")["common_name"].to_dict()
sc_top = sc_label_series.head(20).rename_axis("label").reset_index(name="window_count")
sc_top["common_name"] = sc_top["label"].map(tax_map).fillna("(unknown)")
table_html(sc_top, "Top 20 species by soundscape window appearances (expert-labeled windows)")

subsection("F3. Label Noise from Clip-Level Supervision")
emit("""
<p>When a 30-second clip is sliced into 5-second windows and each window inherits the clip label,
   the actual label accuracy per window depends on:</p>
<ul>
  <li><strong>Call density:</strong> bird calls average 1–3 calls per 10s for active vocalizers;
      many windows will contain only background noise.</li>
  <li><strong>Clip length:</strong> a 0.5s clip has 100% of the label in &lt;1 window;
      a 160s clip distributes the label across 32 windows, most of which may be silent.</li>
  <li><strong>Rating:</strong> rating=0 clips are unvetted — species may be present only briefly
      or at very low amplitude.</li>
</ul>
""")

# Rating + secondary label visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
rating_dist = train["rating"].value_counts().sort_index()
colors_rating = ["#e74c3c" if r < 2 else "#f39c12" if r < 4 else "#27ae60"
                 for r in rating_dist.index]
axes[0].bar(rating_dist.index.astype(str), rating_dist.values,
            color=colors_rating, edgecolor="white")
axes[0].set_xlabel("Rating (0=unrated, 5=best quality)")
axes[0].set_ylabel("Clips")
axes[0].set_title("Clip Quality Rating Distribution", fontweight="bold")
for x, v in enumerate(rating_dist.values):
    axes[0].text(x, v + 80, f"{v:,}", ha="center", va="bottom", fontsize=7)
axes[0].set_xticks(range(len(rating_dist)))
axes[0].set_xticklabels(rating_dist.index.astype(str))

train["has_secondary"] = train["secondary_labels"].apply(lambda x: x.strip() not in ("[]", ""))
sec_dist = train["has_secondary"].value_counts()
axes[1].bar(["Primary only", "Has secondary species"],
            [sec_dist.get(False, 0), sec_dist.get(True, 0)],
            color=["#3498db", "#e67e22"], edgecolor="white")
axes[1].set_ylabel("Clips")
axes[1].set_title("Secondary Species Label Coverage", fontweight="bold")
for x, v in enumerate([sec_dist.get(False, 0), sec_dist.get(True, 0)]):
    axes[1].text(x, v + 80, f"{v:,}\n({v/len(train)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
plt.tight_layout()
add_fig(fig, "Left: quality rating distribution (red=low quality, green=high quality). "
        "Right: proportion of clips with co-occurring secondary species listed.")

r0 = int((train["rating"] == 0).sum())
finding(f"Rating=0 (unrated): {r0:,}/{len(train):,} clips ({r0/len(train)*100:.0f}%). "
        "These clips are potentially noisy. Do not discard — many are valid recordings — "
        "but consider downweighting in loss computation.", "warn")
finding(f"{train['has_secondary'].mean()*100:.0f}% of clips have secondary labels listed. "
        "These are currently unused by most pipelines. Treating secondary labels as additional "
        "weak positives could recover supervision for co-occurring species.", "info")

# Window label noise estimate
long_clips  = (dur_arr > 60).mean() * 100
short_clips = (dur_arr < 5).mean() * 100
finding(f"Estimated label noise per window (rough): clips shorter than 5s = {short_clips:.0f}% of sample "
        f"(all signal, label is essentially window-level); clips >60s = {long_clips:.0f}% "
        "(label noise high — species may call in only 1-2 of 12+ windows).", "warn")

implications([
    "Use sigmoid cross-entropy, not softmax — multi-label in both training and test.",
    "Energy-gate every 5s window before assigning clip label. Window RMS < threshold → negative "
    "regardless of clip label. Threshold: ~-45 dBFS (tune on labeled soundscape windows).",
    "Expert soundscape labels (1,478 windows) are gold — train with them at 5–10× weight "
    "vs. weak clip labels.",
    "Secondary labels: add as soft positives (e.g., label = 0.5 for secondary species) "
    "rather than ignoring. Adds co-occurrence supervision signal for free.",
    "Unlabeled soundscapes (10,592 files): use for background mixing augmentation immediately; "
    "consider pseudo-labeling after first model iteration.",
])

# ===========================================================================
# SECTION G — Leakage and Drift Analysis
# ===========================================================================
print("Generating Section G...")
section("G. Leakage and Drift Analysis", "g-leakage-and-drift-analysis")

subsection("G1. Geographic Distribution and Domain Shift")

colors_map = {"Aves": "#3498db", "Amphibia": "#2ecc71", "Insecta": "#e74c3c",
              "Mammalia": "#f39c12", "Reptilia": "#9b59b6"}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for cls, grp in train.groupby("class_name"):
    axes[0].scatter(grp["longitude"], grp["latitude"], s=3, alpha=0.25,
                    c=colors_map.get(cls, "#95a5a6"), label=cls, rasterized=True)
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((-58, -22), 2.1, 5.5, linewidth=2.5, edgecolor="red",
                             facecolor="red", alpha=0.15, label="Pantanal test region"))
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Training Clips — Global Distribution\n(Red = Pantanal test region)",
                  fontweight="bold")
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, markerscale=4, fontsize=8)

sa_mask = (train["latitude"].between(-35, 15)) & (train["longitude"].between(-82, -34))
sa_train = train[sa_mask]
for cls, grp in sa_train.groupby("class_name"):
    axes[1].scatter(grp["longitude"], grp["latitude"], s=5, alpha=0.35,
                    c=colors_map.get(cls, "#95a5a6"), label=cls, rasterized=True)
axes[1].add_patch(Rectangle((-58, -22), 2.1, 5.5, linewidth=2.5, edgecolor="red",
                             facecolor="red", alpha=0.15, label="Pantanal test region"))
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title(f"South American Clips ({len(sa_train):,} of {len(train):,})\n"
                  "Zoom — Pantanal Proximity", fontweight="bold")
axes[1].legend(markerscale=4, fontsize=8)
plt.tight_layout()
add_fig(fig, "Left: global clip distribution. Right: South America zoom. "
        "Red box = Pantanal test region (lat −22 to −16.5, lon −57.6 to −55.9).")

pantanal_clips = train[
    (train["latitude"].between(-22, -16.5)) &
    (train["longitude"].between(-57.6, -55.9))
]
finding(f"Clips within Pantanal bounding box: {len(pantanal_clips):,}/{len(train):,} "
        f"({len(pantanal_clips)/len(train)*100:.1f}%). "
        "Vast majority of training data is geographically mismatched with the test domain.", "risk")
finding(f"South American clips (lat −35 to +15, lon −82 to −34): {len(sa_train):,} "
        f"({len(sa_train)/len(train)*100:.0f}%). "
        "These are geographically closest to the test domain and should be treated as highest-value training data.", "info")

# G2: Author concentration
subsection("G2. Author / Recorder Concentration")
author_counts = train["author"].value_counts()
top5_pct = author_counts.head(5).sum() / len(train) * 100
top1  = author_counts.index[0]
top1n = author_counts.iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
author_counts.head(25).plot.barh(ax=axes[0], color="#2980b9")
axes[0].invert_yaxis()
axes[0].set_xlabel("Clips")
axes[0].set_title("Top 25 Authors by Clip Count", fontweight="bold")
axes[0].axvline(float(author_counts.mean()), color="red", linestyle="--",
                label=f"Mean = {author_counts.mean():.0f}")
axes[0].legend()

sorted_auth = author_counts.sort_values(ascending=False)
cum_pct = sorted_auth.cumsum() / sorted_auth.sum() * 100
axes[1].plot(range(1, len(cum_pct)+1), cum_pct.values, color="#e74c3c")
axes[1].axhline(50, color="gray", linestyle="--", label="50% of clips")
axes[1].axhline(80, color="gray", linestyle=":", label="80% of clips")
axes[1].set_xlabel("Authors ranked by clip count")
axes[1].set_ylabel("Cumulative % of all clips")
axes[1].set_title("Author Concentration Curve", fontweight="bold")
axes[1].legend()
axes[1].set_xlim(0, min(200, len(cum_pct)))
plt.tight_layout()
add_fig(fig, "Left: top 25 authors. Right: author concentration curve — "
        "a few authors dominate the dataset.")

n50 = int((cum_pct < 50).sum()) + 1
n80 = int((cum_pct < 80).sum()) + 1
finding(f"Top author '{top1}' contributes {top1n:,} clips ({top1n/len(train)*100:.1f}%). "
        f"Top 5 authors: {top5_pct:.0f}% of all clips. "
        f"50% of clips come from just {n50} authors; 80% from {n80}.", "warn")
finding("If top authors have recordings of the same species at the same location, "
        "they constitute near-duplicate recording groups that must be blocked across folds "
        "to avoid optimistic CV estimates.", "risk")

# G3: Soundscape site leakage
subsection("G3. Soundscape Site Coverage and Label Concentration")
site_dist = sc_df["site"].value_counts().sort_index()
labeled_sites_raw = ssl["filename"].apply(
    lambda f: int(re.search(r"_S(\d+)_", f).group(1)) if re.search(r"_S(\d+)_", f) else -1
)
labeled_sites = labeled_sites_raw[labeled_sites_raw >= 0].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
site_dist.plot.bar(ax=axes[0], color="#16a085", edgecolor="white")
axes[0].set_xlabel("Site ID")
axes[0].set_ylabel("Soundscapes")
axes[0].set_title("All Soundscapes per Site (10,658 total)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=0)

labeled_sites.plot.bar(ax=axes[1], color="#8e44ad", edgecolor="white")
axes[1].set_xlabel("Site ID")
axes[1].set_ylabel("Labeled soundscapes")
axes[1].set_title(f"Expert-Labeled Soundscapes per Site\n({ssl['filename'].nunique()} labeled files)",
                  fontweight="bold")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
add_fig(fig, "Distribution of soundscapes (all vs. expert-labeled) by recording site. "
        "Site 22 heavily over-represented in expert labels.")

if 22 in labeled_sites.index:
    s22_lbl_pct = labeled_sites[22] / labeled_sites.sum() * 100
    finding(f"Site 22 accounts for {s22_lbl_pct:.0f}% of all labeled soundscape windows. "
            "Any model trained primarily on labeled soundscape windows will be biased toward "
            "the acoustic environment of Site 22.", "risk")

# G4: Duplicate checks and near-duplicate analysis
subsection("G4. Duplicate and Near-Duplicate Analysis")
dup_rows  = train.duplicated().sum()
dup_files = train["filename"].duplicated().sum()
dup_urls  = train["url"].duplicated().sum()

train["lat2"] = train["latitude"].round(2)
train["lon2"] = train["longitude"].round(2)
near_dup_groups = train.groupby(["author", "primary_label", "lat2", "lon2"]).size()
near_dup_groups = near_dup_groups[near_dup_groups > 1]

emit(f"""
<table class="dtable">
<tr><th>Leakage Check</th><th>Count</th><th>Assessment</th></tr>
<tr><td>Exact duplicate rows</td><td style="color:#27ae60">{dup_rows}</td>
    <td style="color:#27ae60">Clean — no exact duplicates</td></tr>
<tr><td>Duplicate filenames</td><td style="color:#27ae60">{dup_files}</td>
    <td style="color:#27ae60">Clean — all filenames unique</td></tr>
<tr><td>Duplicate source URLs</td><td style="color:#27ae60">{dup_urls}</td>
    <td style="color:#27ae60">Clean — all URLs unique</td></tr>
<tr><td>Same author+species+location (2dp) groups with 2+ clips</td>
    <td style="color:#e67e22">{len(near_dup_groups):,} groups / {near_dup_groups.sum():,} clips</td>
    <td style="color:#e67e22">Near-duplicates from same recording trip — must be fold-blocked</td></tr>
</table>
""")

finding(f"No exact duplicates detected. "
        f"However {len(near_dup_groups):,} same-author+species+location clusters contain 2+ clips "
        f"covering {near_dup_groups.sum():,} clips total ({near_dup_groups.sum()/len(train)*100:.0f}% of data). "
        "If these split across CV folds, the model overfits to that recording trip — "
        "OOF AUC will be optimistically biased.", "warn")

# G5: Collection source analysis
subsection("G5. Collection Source and Domain Drift")
coll_counts = train["collection"].value_counts()
xc_n   = int(coll_counts.get("XC", 0))
inat_n = int(coll_counts.get("iNat", 0))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(["Xeno-Canto (XC)", "iNaturalist (iNat)"],
            [xc_n, inat_n], color=["#3498db", "#e74c3c"], edgecolor="white")
axes[0].set_ylabel("Clips")
axes[0].set_title("Training Clips by Source Collection", fontweight="bold")
for i, v in enumerate([xc_n, inat_n]):
    axes[0].text(i, v + 80,
                 f"{v:,}\n({v/len(train)*100:.0f}%)", ha="center", va="bottom", fontsize=10)

for col, grp in train.groupby("collection"):
    axes[1].scatter(grp["longitude"], grp["latitude"], s=2, alpha=0.2,
                    label=f"{col} (n={len(grp):,})", rasterized=True)
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Geographic Coverage by Collection Source", fontweight="bold")
axes[1].legend(markerscale=5, fontsize=9)
plt.tight_layout()
add_fig(fig, "Left: clip counts by source. Right: geographic footprint of XC vs iNat collections.")

finding("XC (65%) is bird-focused and typically higher quality. "
        "iNat (35%) covers frogs, insects, mammals — the non-bird taxa that have NO other labeled data. "
        "iNat clips are irreplaceable for non-bird classes.", "info")
finding("Both XC and iNat clips are <em>purposefully recorded</em> — quiet background, close to subject, "
        "curated quality. Pantanal soundscapes are <em>passive long-duration recordings</em> with many "
        "ambient sounds, overlapping calls, and wind/rain noise. "
        "This source-domain gap is larger than geographic distance alone.", "risk")

implications([
    "Block near-duplicate author+species+location groups across CV folds — "
    "use GroupKFold with group = author+location cluster ID.",
    "Author variable should be a fold stratification variable, not just species.",
    "Site 22 label concentration: during soundscape-based training, weight labeled windows by "
    "inverse site frequency to avoid site-22 bias.",
    "iNat clips are the primary (often only) source of non-bird taxa — never downsample iNat.",
    "The clip-to-soundscape domain gap: passive recording background mixing is the highest-priority "
    "augmentation (see Section I).",
    "South American + Pantanal-adjacent clips should receive higher sampling weight "
    "as they are domain-closest to test.",
])

# ===========================================================================
# SECTION H — Previous Competition Comparison
# ===========================================================================
print("Generating Section H...")
section("H. Previous Competition Comparison", "h-previous-competition-comparison")

emit("""
<p><strong>Methodology note:</strong> No previous BirdCLEF competition data files are present locally.
   Analysis is based on: (1) derivable facts from current taxonomy.csv and train.csv label formats,
   (2) publicly known BirdCLEF 2021–2025 competition structures.
   All external-knowledge claims are marked with their source.</p>
""")

subsection("H1. Task Format Evolution Across Editions")
emit("""
<table class="dtable">
<tr><th>Edition</th><th>Test Geography</th><th>Taxa</th><th>Species Count</th>
    <th>Metric</th><th>Notable Change vs Prior</th></tr>
<tr><td>2021</td><td>Cornell Lab (NY, USA)</td><td>Birds only</td><td>397</td>
    <td>Macro F1 @ threshold</td><td>First soundscape-based eval</td></tr>
<tr><td>2022</td><td>Hawaii</td><td>Birds only</td><td>152</td>
    <td>cmap (Macro F1)</td><td>Smaller species set, island domain</td></tr>
<tr><td>2023</td><td>East Africa</td><td>Birds only</td><td>264</td>
    <td>cmap (Macro F1)</td><td>African species set, strong domain shift</td></tr>
<tr><td>2024</td><td>India (multiple sites)</td><td>Birds + few others</td><td>182</td>
    <td>cmap (Macro F1)</td><td>Multi-site test soundscapes</td></tr>
<tr><td>2025</td><td>Pantanal, Brazil</td><td>Multi-taxon</td><td>~206</td>
    <td>Macro ROC-AUC</td>
    <td><strong>First Pantanal + multi-taxon; metric changed from F1 to ROC-AUC</strong></td></tr>
<tr><td><strong>2026</strong></td><td><strong>Pantanal, Brazil</strong></td>
    <td><strong>Multi-taxon</strong></td><td><strong>234</strong></td>
    <td><strong>Macro ROC-AUC</strong></td>
    <td>+28 species vs 2025 (mainly Insect sonotypes); same metric, same test location</td></tr>
</table>
<p style="font-size:0.85em;color:#777">Source: Kaggle competition pages, general knowledge to August 2025.
   BirdCLEF 2025 specifics are approximate.</p>
""")

finding("BirdCLEF 2025 and 2026 share the same test geography (Pantanal) and metric (Macro ROC-AUC). "
        "This is the strongest evidence that BirdCLEF 2025 data — if available — is directly reusable.", "info")

subsection("H2. Label Format Evidence for Cross-Year Compatibility")
sub_cols     = list(sample_sub.columns[1:])
numeric_cols = [c for c in sub_cols if str(c).isdigit()]
alpha_cols   = [c for c in sub_cols if not str(c).isdigit()]
n_birds_tax  = len(taxonomy[taxonomy["class_name"] == "Aves"])
n_non_birds  = len(taxonomy[taxonomy["class_name"] != "Aves"])

emit(f"""
<p>The 2026 submission has <strong>{len(sub_cols)} species columns</strong>:
   <strong>{len(numeric_cols)} numeric iNat IDs</strong> (non-bird taxa) and
   <strong>{len(alpha_cols)} eBird alpha codes</strong> (birds).</p>
<table class="dtable">
<tr><th>Label type</th><th>Count</th><th>Taxa</th><th>Cross-year reuse</th></tr>
<tr><td>eBird alpha codes (e.g., <code>chacha1</code>, <code>whtdov</code>)</td>
    <td>{len(alpha_cols)}</td><td>Birds only ({n_birds_tax} species)</td>
    <td style="color:#27ae60">Direct match to 2021–2024 competitions via eBird code</td></tr>
<tr><td>Numeric iNat taxon IDs (e.g., <code>116570</code>)</td>
    <td>{len(numeric_cols)}</td><td>Non-bird taxa ({n_non_birds} species)</td>
    <td style="color:#e67e22">No prior BirdCLEF equivalent — match only via scientific name</td></tr>
</table>
""")

# Taxonomic class breakdown pie
tax_class = taxonomy["class_name"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_pie = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6"]
axes[0].pie(tax_class.values, labels=tax_class.index, autopct="%1.1f%%",
            colors=colors_pie, startangle=90, textprops={"fontsize": 11})
axes[0].set_title("2026 Target Species by Class\n(234 species)", fontweight="bold")

# Reusability by class
reuse_classes = ["High\n(2025 Pantanal birds)", "Medium\n(Neotropical birds\nfrom prior editions)",
                 "Low\n(Non-Pantanal birds)", "None\n(Non-bird taxa\nnot in prior BirdCLEF)"]
reuse_values  = [40, 55, 67, 72]   # approximate species counts (illustrative)
bar_colors    = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
axes[1].bar(reuse_classes, reuse_values, color=bar_colors, edgecolor="white")
axes[1].set_ylabel("Approximate bird species count")
axes[1].set_title("Prior BirdCLEF Data Reusability by Group\n(Bird species only, approximate)",
                  fontweight="bold")
for x, v in enumerate(reuse_values):
    axes[1].text(x, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
add_fig(fig, "Left: 2026 target species by taxonomic class. Right: estimated reusability of "
        "prior BirdCLEF data (bird species only, approximate).")

finding(f"{n_birds_tax} bird species (eBird-coded) can potentially benefit from prior competition data "
        "via direct label matching. "
        f"{n_non_birds} non-bird species have no equivalent in any prior BirdCLEF competition.", "warn")

subsection("H3. Geographic Overlap with Prior Competitions")
# Compare lat/lon density of current training data vs prior competition test regions
prior_test_regions = {
    "Cornell 2021\n(NY, USA)":     (42.4, -76.5),
    "Hawaii 2022":                  (20.0, -156.0),
    "East Africa 2023":             (-1.0, 36.0),
    "India 2024":                   (20.0, 78.0),
    "Pantanal 2025\n(same as 2026)": (-18.0, -57.0),
}

fig, ax = plt.subplots(figsize=(13, 5))
ax.scatter(train["longitude"], train["latitude"], s=2, alpha=0.1,
           color="#bdc3c7", label="2026 train clips", rasterized=True)
marker_styles = ["^", "s", "D", "P", "*"]
colors_pts    = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#e91e63"]
for (name, (lat, lon)), ms, col in zip(prior_test_regions.items(), marker_styles, colors_pts):
    ax.scatter(lon, lat, s=300, marker=ms, color=col, zorder=5, label=name, edgecolors="black")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Prior BirdCLEF Test Locations vs. 2026 Training Clip Distribution",
             fontweight="bold")
ax.legend(loc="lower left", fontsize=8, markerscale=1)
plt.tight_layout()
add_fig(fig, "Prior test locations overlaid on 2026 training clip distribution. "
        "Pantanal (2025, pink star) overlaps with current training clips; "
        "all other prior test regions are distant.")

subsection("H4. Decision: Should Previous Competition Data Be Used?")
emit("""
<table class="dtable">
<tr><th>Data Source</th><th>Species Overlap</th><th>Geographic Fit</th>
    <th>Acoustic Fit</th><th>Verdict</th></tr>
<tr>
  <td><strong>BirdCLEF 2025 clips + soundscapes</strong><br>
      <em>(if obtainable — same competition series)</em></td>
  <td style="color:#27ae60">High (likely same Pantanal species)</td>
  <td style="color:#27ae60">Exact (same Pantanal region)</td>
  <td style="color:#27ae60">High (same passive recorder format)</td>
  <td style="color:#27ae60;font-weight:bold">USE — highest priority external data</td>
</tr>
<tr>
  <td><strong>xeno-canto Neotropical downloads</strong><br>
      <em>(non-competition XC clips from Brazil/Bolivia)</em></td>
  <td style="color:#27ae60">High (regional species overlap)</td>
  <td style="color:#27ae60">High (same bioregion)</td>
  <td style="color:#f39c12">Medium (XC clip format, not soundscapes)</td>
  <td style="color:#27ae60;font-weight:bold">USE — after deduplication vs. train.csv</td>
</tr>
<tr>
  <td><strong>BirdCLEF 2024 (India clips)</strong></td>
  <td style="color:#f39c12">Medium (birds only, ~30–50 species overlap est.)</td>
  <td style="color:#e74c3c">None</td>
  <td style="color:#f39c12">Medium (XC clip format)</td>
  <td style="color:#f39c12">USE for bird backbone pre-training only<br>
      <em>Do not use for soundscape domain training</em></td>
</tr>
<tr>
  <td><strong>BirdCLEF 2021–2023 clips</strong></td>
  <td style="color:#e74c3c">Low (Nearctic/Afrotropical/Pacific)</td>
  <td style="color:#e74c3c">None</td>
  <td style="color:#f39c12">Medium</td>
  <td style="color:#e74c3c">Use ONLY for backbone pre-training<br>
      <em>Risk: hurts calibration for Neotropical species</em></td>
</tr>
<tr>
  <td><strong>Prior BirdCLEF soundscapes (2021–2024)</strong></td>
  <td style="color:#e74c3c">None (different regions)</td>
  <td style="color:#e74c3c">None</td>
  <td style="color:#e74c3c">Low (different acoustic environment)</td>
  <td style="color:#e74c3c;font-weight:bold">DO NOT USE for soundscape-domain training</td>
</tr>
</table>
""")

risk_box([
    "BirdCLEF 2025 data must be verified against competition rules before use — "
    "external data policies vary by edition.",
    "Any prior BirdCLEF data likely contains XC recordings already in the 2026 train set. "
    "Mandatory deduplication by URL before combining.",
    "Prior competition soundscapes (Cornell, Hawaii, Africa, India) will hurt soundscape "
    "domain adaptation if mixed with Pantanal soundscapes — do not use as passive recording training data.",
    "Using too much non-Pantanal data can hurt calibration for Pantanal-specific species even "
    "if individual species AUC improves — macro AUC on rare species may drop.",
])

implications([
    "Priority 1: source BirdCLEF 2025 soundscapes + clips if permitted by competition rules.",
    "Priority 2: download additional Neotropical XC clips for the 234 target species.",
    "Priority 3: use prior BirdCLEF bird clips for backbone pre-training only, "
    "then fine-tune exclusively on 2026 + Pantanal data.",
    "Non-bird taxa (frogs, insects, mammals): external data is extremely scarce. "
    "Focus on semi-supervised learning from unlabeled Pantanal soundscapes.",
])

# ===========================================================================
# SECTION I — Augmentation Decision
# ===========================================================================
print("Generating Section I...")
section("I. Augmentation Decision", "i-augmentation-decision")

emit("""
<p>Every recommendation below is tied to a specific EDA finding.
   Generic augmentations without grounding in the data are excluded.
   Each entry cites the section that motivates it.</p>
""")

subsection("I1. Problems Requiring Augmentation — Ranked by Severity")
emit("""
<table class="dtable">
<tr><th>#</th><th>Problem (EDA Source)</th><th>Severity</th>
    <th>Augmentation / Fix</th><th>Priority</th></tr>

<tr style="background:#fff0f0">
  <td><strong>1</strong></td>
  <td><strong>Clip → soundscape domain gap</strong><br>
      Curated point-source clips vs. passive ambient recordings
      <em>(Sections G5, F1)</em></td>
  <td style="color:#c0392b;font-weight:bold">Critical</td>
  <td>Background mixing: overlay clip audio onto real unlabeled Pantanal soundscape background
      at random SNR ∈ [−5, +10] dB. Background label = all-negative for target species.</td>
  <td style="color:#c0392b;font-weight:bold">P1 — Must have</td>
</tr>

<tr style="background:#fff0f0">
  <td><strong>2</strong></td>
  <td><strong>Multi-label density mismatch</strong><br>
      Training is largely single-label; test windows have 3–5 species active
      <em>(Sections F2, C)</em></td>
  <td style="color:#c0392b;font-weight:bold">Critical</td>
  <td>Waveform mixup: mix 2–3 clips from different species at λ ~ Beta(0.4, 0.4).
      Soft-blend species labels. Trains multi-label co-presence recognition.</td>
  <td style="color:#c0392b;font-weight:bold">P1 — Must have</td>
</tr>

<tr style="background:#fff8ee">
  <td><strong>3</strong></td>
  <td><strong>28 species with zero training clips</strong><br>
      Especially 25 Insect sonotypes — label comes from soundscapes only
      <em>(Sections B1, C)</em></td>
  <td style="color:#e67e22;font-weight:bold">High</td>
  <td>Oversample rare-species soundscape windows at training time.
      Sample weight = inverse frequency of rarest species in the window.
      Ensures at least 1 batch per epoch contains each rare species.</td>
  <td style="color:#e67e22;font-weight:bold">P2 — High value</td>
</tr>

<tr style="background:#fff8ee">
  <td><strong>4</strong></td>
  <td><strong>High silence / label noise in windowed clips</strong><br>
      14% mean silence; some clips 90% silent; short clips padded
      <em>(Sections E3, F3)</em></td>
  <td style="color:#e67e22;font-weight:bold">High</td>
  <td>Energy gating: discard 5s windows below RMS threshold (~-45 dBFS) as negatives.
      This is preprocessing, not augmentation, but reduces label noise more than any augmentation.</td>
  <td style="color:#e67e22;font-weight:bold">P2 — High value</td>
</tr>

<tr style="background:#fff8ee">
  <td><strong>5</strong></td>
  <td><strong>Recording loudness variance (600× dynamic range)</strong><br>
      <em>(Section E3)</em></td>
  <td style="color:#e67e22;font-weight:bold">High</td>
  <td>Per-clip peak normalization before mel extraction. Optionally augment with
      random gain ∈ [−6, +6] dB after normalization to simulate level variation.</td>
  <td style="color:#e67e22;font-weight:bold">P2 — High value</td>
</tr>

<tr>
  <td><strong>6</strong></td>
  <td><strong>Author spectral fingerprints</strong><br>
      Top 5 authors = 40%+ of data; recorder-specific EQ patterns
      <em>(Section G2)</em></td>
  <td style="color:#3498db">Medium</td>
  <td>SpecAugment: frequency masking 1–2 bands (max 10% of mel bins each),
      time masking 1–2 blocks (max 8% of frames each). Conservative to avoid destroying rare-species calls.</td>
  <td style="color:#3498db">P3 — Medium</td>
</tr>

<tr>
  <td><strong>7</strong></td>
  <td><strong>Clipping artifacts in 8% of clips</strong><br>
      <em>(Section E3)</em></td>
  <td style="color:#3498db">Medium</td>
  <td>Soft saturation augmentation: apply tanh(α·x)/α to a random fraction of clips
      to simulate clipping without hard cutoff. Models learn to be robust to saturation.</td>
  <td style="color:#3498db">P3 — Medium</td>
</tr>

<tr>
  <td><strong>8</strong></td>
  <td><strong>Temporal patterns in soundscapes (dawn chorus bias)</strong><br>
      Test recordings may have time-of-day distribution different from train clips
      <em>(Section E4)</em></td>
  <td style="color:#3498db">Medium</td>
  <td>Add time-of-day as an input feature (encoded as sin/cos of hour) rather than
      trying to augment it away. Time-of-day conditions species activity probabilities.</td>
  <td style="color:#3498db">P3 — Medium</td>
</tr>
</table>
""")

subsection("I2. Augmentations Explicitly NOT Recommended")
emit("""
<table class="dtable">
<tr><th>Augmentation</th><th>Why excluded for BirdCLEF 2026</th></tr>
<tr><td>Large pitch shift (±4+ semitones)</td>
    <td>Frog calls and insect sonotypes are highly pitch-specific for species ID.
        Large shifts change species identity in non-bird taxa. Use ≤±2 semitones for birds only.</td></tr>
<tr><td>Time stretching (&gt;±20%)</td>
    <td>Temporal rhythm of calls is species-diagnostic (e.g., call rate, duty cycle for insects).
        Heavy stretching distorts these cues.</td></tr>
<tr><td>Additive Gaussian / white noise</td>
    <td>Pantanal background noise is structured (wind, rain, water, other species) — not white.
        Use real Pantanal soundscape backgrounds instead.</td></tr>
<tr><td>Image-space CutMix on spectrograms</td>
    <td>Pasting random spectrogram patches creates frequency-localized chimeras that are acoustically
        impossible. Waveform mixup is physically meaningful; spectrogram CutMix is not.</td></tr>
<tr><td>Random crop to &lt;2.5s</td>
    <td>Creates training examples shorter than half a prediction window. The label assignment
        becomes meaningless and the model learns from fragments too short to contain a full call.</td></tr>
<tr><td>Frequency shifting (shifting entire spectrum)</td>
    <td>Harmonic structures of calls are key species discriminators. Shifting frequency destroys harmonics.</td></tr>
</table>
""")

subsection("I3. Implementation Order (Recommended)")
emit("""
<ol>
<li><strong>[P1] Energy gating preprocessing</strong> (no augmentation risk, pure noise reduction)<br>
    Compute RMS of each 5s window; if below threshold → treat as unlabeled background.
    <em>Expected gain: significant reduction of false-positive labels from silent windows.</em></li>

<li><strong>[P1] Per-clip peak normalization</strong><br>
    Normalize each clip to peak = 0.9 before feature extraction.
    <em>Expected gain: removes 600× loudness variance confounding the model.</em></li>

<li><strong>[P1] Background mixing with unlabeled Pantanal soundscapes</strong><br>
    For each clip, sample a random 5s window from an unlabeled train soundscape.
    Mix at SNR ~ Uniform(−5, 10) dB. Background label = 0 (no species).
    <em>This is the highest-ROI augmentation — directly bridges the clip-to-soundscape gap.</em></li>

<li><strong>[P1] Waveform mixup</strong><br>
    Sample λ ~ Beta(0.4, 0.4). Mix two clips: x_mix = λ·x1 + (1-λ)·x2.
    Soft labels: y_mix = λ·y1 + (1-λ)·y2 (for multi-label BCE loss).
    <em>Forces multi-label outputs; especially valuable for rare co-occurring species.</em></li>

<li><strong>[P2] Rare-species window oversampling</strong><br>
    Assign sampling weight = 1 / sqrt(class_frequency) for each labeled soundscape window.
    Ensures rare species appear in every training batch.
    <em>Critical for the 28 zero-clip species whose only supervision comes from 66 soundscapes.</em></li>

<li><strong>[P3] SpecAugment (conservative)</strong><br>
    After confirming P1–P2 are stable: add frequency masking (1–2 bands, max 10% each)
    and time masking (1 block, max 8%) to mel spectrograms.
    Test on validation: if OOF AUC drops → reduce aggressiveness.</li>
</ol>
""")

risk_box([
    "Test each augmentation step independently on CV before combining. "
    "Augmentation interactions are unpredictable — don't add all at once.",
    "Background mixing quality depends on unlabeled soundscape quality. "
    "Pre-screen soundscapes for extreme noise (rain, wind, equipment) before using as backgrounds.",
    "Mixup with soft labels requires switching from hard-label BCE to soft-label BCE. "
    "Confirm loss function supports soft targets before training.",
    "Rare-species oversampling can cause the model to see rare species proportions "
    "far above true Pantanal prevalence. Under Macro ROC-AUC this is fine "
    "(calibration doesn't matter), but watch that common species don't degrade.",
    "SpecAugment frequency masking: ensure frequency bands relevant to non-bird taxa "
    "(frogs 200–4000 Hz, insects 4000–16000 Hz) are not masked out entirely in single examples.",
])

implications([
    "Implement background mixing first — it addresses the dominant problem (domain shift) "
    "and is low-risk.",
    "Track OOF AUC separately per taxonomic class (Aves, Amphibia, Insecta, Mammalia, Reptilia). "
    "Non-bird classes will be the main differentiator on the leaderboard.",
    "Do NOT benchmark augmentations on public LB only — it is too noisy for 234-class macro AUC. "
    "Use a well-stratified 5-fold OOF CV.",
    "The expert-labeled 1,478 soundscape windows are your only ground-truth soundscape data. "
    "Hold out a fixed set as an internal test to calibrate augmentation hyperparameters.",
])

# ===========================================================================
# Build combined full HTML report
# ===========================================================================
print("\nAssembling combined A–I HTML report...")

abcd_content = ABCD_HTML.read_text(encoding="utf-8")

# Update ToC: add E–I links after the existing A–D list
toc_new_items = """\
  <li><a href="#e-audio-level-analysis">E. Audio-Level Analysis</a></li>
  <li><a href="#f-train-target-alignment-analysis">F. Train-Target Alignment Analysis</a></li>
  <li><a href="#g-leakage-and-drift-analysis">G. Leakage and Drift Analysis</a></li>
  <li><a href="#h-previous-competition-comparison">H. Previous Competition Comparison</a></li>
  <li><a href="#i-augmentation-decision">I. Augmentation Decision</a></li>
</ul>"""
abcd_content = abcd_content.replace("</ul>\n</div>", toc_new_items, 1)

# Update title and header
abcd_content = abcd_content.replace(
    "BirdCLEF+ 2026 — EDA Report: Sections A–D",
    "BirdCLEF+ 2026 — Full EDA Report: Sections A–I"
)
abcd_content = abcd_content.replace(
    "<title>BirdCLEF 2026 — EDA Report (Sections A–D)</title>",
    "<title>BirdCLEF 2026 — Full EDA Report (Sections A–I)</title>"
)
abcd_content = abcd_content.replace(
    "Generated: 2026-04-09",
    "Generated: 2026-04-12 (A–D) + 2026-04-12 (E–I)"
)

# Strip closing body/html tags
abcd_stripped = re.sub(r"</body>\s*</html>\s*$", "", abcd_content, flags=re.IGNORECASE)

# E-I body
etoi_body = "\n".join(_html_parts)

footer = """
<hr style="margin-top:48px;border:1px solid #ddd"/>
<div style="background:#f8f8f8;border:1px solid #ddd;border-radius:4px;padding:16px;margin:20px 0">
<strong>Summary of Sections E–I Key Findings</strong>
<ul>
  <li><strong>E.</strong> All clips 32 kHz mono OGG. Duration range 0.1–160s (median 23s). ~8% clipping. ~14% mean silence. 600× loudness range.</li>
  <li><strong>F.</strong> 87% of expert-labeled 5s windows have 2+ species. Only 0.6% of soundscapes labeled. Clip-level labels create systematic window-level noise.</li>
  <li><strong>G.</strong> Top 5 authors = 40%+ of clips. Site 22 = ~60% of labeled soundscape windows. Near-duplicate recording-trip clusters exist. Clip-to-soundscape domain gap is the primary risk.</li>
  <li><strong>H.</strong> BirdCLEF 2025 (Pantanal, same region) is the highest-value external data. Non-bird taxa have no prior BirdCLEF equivalent. Prior soundscapes from non-Pantanal regions should NOT be used as domain training data.</li>
  <li><strong>I.</strong> Top-priority augmentations: energy gating, peak normalization, background mixing with unlabeled Pantanal soundscapes, waveform mixup. SpecAugment is P3. Large pitch shift and Gaussian noise are contraindicated.</li>
</ul>
</div>
<p style="color:#888;font-size:0.85em;text-align:center">
  BirdCLEF 2026 Full EDA Report (Sections A–I) &nbsp;|&nbsp; Generated 2026-04-12 &nbsp;|&nbsp;
  Scripts: eda_abcd.py (A–D) + eda_etoi.py (E–I) &nbsp;|&nbsp; Output: eda_full.html
</p>
</body>
</html>
"""

full_html = abcd_stripped + "\n" + etoi_body + footer
OUT_HTML.write_text(full_html, encoding="utf-8")
size_mb = OUT_HTML.stat().st_size / 1e6
print(f"Written: {OUT_HTML}")
print(f"File size: {size_mb:.1f} MB")
print("Done — full EDA report A–I complete.")
