"""
BirdCLEF 2026 — Modeling Hypotheses & Strategy Report Generator
================================================================
Synthesizes all EDA findings (Sections A–I) into a ranked modeling
strategy HTML report. Run from repo root:

    python3 reports/generate_modeling_report.py

Output: reports/modeling_report.html
"""

from pathlib import Path
import datetime

OUT = Path(__file__).parent / "modeling_report.html"

DATE = datetime.date.today().isoformat()

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def tag(t, content, **attrs):
    attr_str = "".join(f' {k.rstrip("_")}="{v}"' for k, v in attrs.items())
    return f"<{t}{attr_str}>{content}</{t}>"

def h1(text, anchor=""):
    a = f' id="{anchor}"' if anchor else ""
    return f'<h1{a}>{text}</h1>'

def h2(text, anchor=""):
    a = f' id="{anchor}"' if anchor else ""
    return f'<h2{a}>{text}</h2>'

def h3(text):
    return f"<h3>{text}</h3>"

def p(text, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f"<p{c}>{text}</p>"

def ul(items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f"<ul>{lis}</ul>"

def table(headers, rows, col_classes=None):
    ths = "".join(f"<th>{h}</th>" for h in headers)
    tr_head = f"<thead><tr>{ths}</tr></thead>"
    body_rows = []
    for i, row in enumerate(rows):
        tds = ""
        for j, cell in enumerate(row):
            cls = col_classes[j] if col_classes and j < len(col_classes) else ""
            c = f' class="{cls}"' if cls else ""
            tds += f"<td{c}>{cell}</td>"
        cls = "alt" if i % 2 == 1 else ""
        body_rows.append(f'<tr class="{cls}">{tds}</tr>')
    tr_body = f"<tbody>{''.join(body_rows)}</tbody>"
    return f'<div class="table-wrap"><table>{tr_head}{tr_body}</table></div>'

def badge(text, color):
    colors = {
        "red":    ("#c0392b", "#fdf0f0"),
        "orange": ("#e67e22", "#fef9e7"),
        "blue":   ("#2980b9", "#eaf4fb"),
        "green":  ("#27ae60", "#eafaf1"),
        "grey":   ("#7f8c8d", "#f4f4f4"),
    }
    fg, bg = colors.get(color, ("#333", "#eee"))
    return f'<span class="badge" style="color:{fg};background:{bg};border:1px solid {fg}">{text}</span>'

def infobox(content, color="blue"):
    colors = {
        "blue":   ("#2980b9", "#eaf4fb"),
        "orange": ("#e67e22", "#fef9e7"),
        "green":  ("#27ae60", "#eafaf1"),
        "red":    ("#c0392b", "#fdf0f0"),
        "yellow": ("#f39c12", "#fffde7"),
    }
    fg, bg = colors.get(color, ("#333", "#f9f9f9"))
    return (f'<div class="infobox" style="border-left:4px solid {fg};'
            f'background:{bg};padding:12px 16px;margin:12px 0;border-radius:4px">'
            f'{content}</div>')

def section_div(content):
    return f'<div class="section">{content}</div>'

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #2c3e50;
    background: #f5f6fa;
}

.page-wrap {
    max-width: 1100px;
    margin: 0 auto;
    background: white;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

/* Cover */
.cover {
    background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
    color: white;
    padding: 60px 48px 48px;
}
.cover h1 {
    font-size: 2.4em;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: white;
    border: none;
    margin-bottom: 6px;
    padding: 0;
}
.cover .subtitle {
    font-size: 1.15em;
    opacity: 0.85;
    margin-bottom: 4px;
}
.cover .date { font-size: 0.9em; opacity: 0.65; margin-top: 8px; }
.cover-divider {
    border: none;
    border-top: 2px solid rgba(255,255,255,0.35);
    margin: 20px 0;
    width: 200px;
}
.cover-stats {
    margin-top: 28px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}
.cover-stat {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 6px;
    padding: 10px 16px;
    flex: 1 1 240px;
}
.cover-stat .label { font-size: 0.78em; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.5px; }
.cover-stat .value { font-size: 0.95em; font-weight: 600; margin-top: 2px; }

/* Navigation */
nav {
    background: #1a5276;
    padding: 0 24px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    flex-wrap: wrap;
    gap: 0;
}
nav a {
    color: rgba(255,255,255,0.8);
    text-decoration: none;
    padding: 10px 14px;
    font-size: 0.82em;
    font-weight: 500;
    display: inline-block;
    transition: background 0.15s;
    white-space: nowrap;
}
nav a:hover { background: rgba(255,255,255,0.1); color: white; }

/* Main content */
.content { padding: 32px 48px 60px; }

.section {
    margin-bottom: 48px;
    padding-bottom: 32px;
    border-bottom: 1px solid #ecf0f1;
}
.section:last-child { border-bottom: none; }

h1 {
    font-size: 1.5em;
    font-weight: 700;
    color: #1a5276;
    margin: 32px 0 8px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2980b9;
}
h2 {
    font-size: 1.1em;
    font-weight: 700;
    color: #2c3e50;
    margin: 24px 0 8px;
}
h3 {
    font-size: 0.95em;
    font-weight: 700;
    color: #555;
    margin: 16px 0 6px;
    font-style: italic;
}
p { margin-bottom: 10px; }

ul { margin: 8px 0 12px 20px; }
li { margin-bottom: 5px; line-height: 1.55; }

/* Tables */
.table-wrap {
    overflow-x: auto;
    margin: 12px 0 20px;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88em;
}
thead tr {
    background: #1a5276;
    color: white;
}
thead th {
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.85em;
    white-space: nowrap;
}
tbody td {
    padding: 9px 12px;
    border-bottom: 1px solid #e8ecef;
    vertical-align: top;
    line-height: 1.5;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr.alt td { background: #f7f9fc; }
tbody tr:hover td { background: #eaf4fb; }

/* Priority row colors */
.row-critical td { background: #fdf0f0 !important; }
.row-high td     { background: #fef9e7 !important; }
.row-medium td   { background: #eaf4fb !important; }
.row-green td    { background: #eafaf1 !important; }
.row-excluded td { background: #fdf0f0 !important; }

/* Column utilities */
.col-num   { width: 36px; text-align: center; font-weight: 700; white-space: nowrap; }
.col-pri   { width: 110px; white-space: nowrap; font-weight: 700; }
.col-tag   { width: 100px; white-space: nowrap; font-weight: 700; }
.col-sev   { width: 90px; white-space: nowrap; font-weight: 700; }

/* Priority labels */
.pri-p1     { color: #c0392b; }
.pri-p2     { color: #e67e22; }
.pri-p3     { color: #2980b9; }
.pri-crit   { color: #c0392b; font-weight: 700; }
.pri-high   { color: #e67e22; font-weight: 700; }
.pri-medium { color: #2980b9; }

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    font-weight: 600;
    white-space: nowrap;
}

/* Info boxes */
.infobox { font-size: 0.9em; }
.infobox strong { display: block; margin-bottom: 4px; }

/* Key stats grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: #f7f9fc;
    border: 1px solid #d5e8f5;
    border-radius: 6px;
    padding: 12px 14px;
}
.stat-card .s-label { font-size: 0.75em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-card .s-value { font-size: 1em; font-weight: 700; color: #1a5276; margin-top: 2px; }

/* TOC */
.toc-list {
    list-style: none;
    margin: 0;
    padding: 0;
    columns: 2;
    column-gap: 24px;
}
.toc-list li {
    padding: 5px 0;
    border-bottom: 1px dotted #ddd;
    margin-bottom: 0;
}
.toc-list a {
    color: #2980b9;
    text-decoration: none;
    font-size: 0.92em;
}
.toc-list a:hover { text-decoration: underline; }
.toc-num { color: #aaa; margin-right: 6px; font-size: 0.85em; }

/* Summary box */
.summary-box {
    background: linear-gradient(135deg, #eaf4fb, #f0faf5);
    border: 1px solid #aed6f1;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 24px 0 0;
}
.summary-box h3 { color: #1a5276; margin-top: 0; font-size: 1em; }

/* Footer */
footer {
    background: #2c3e50;
    color: rgba(255,255,255,0.5);
    text-align: center;
    font-size: 0.78em;
    padding: 14px;
}

code {
    background: #f4f4f4;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: monospace;
    font-size: 0.9em;
}

@media print {
    nav { display: none; }
    .page-wrap { box-shadow: none; }
    .cover { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
}
"""

# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

parts = []

def emit(html):
    parts.append(html)

# ── HEAD ────────────────────────────────────────────────────────────────────
emit(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BirdCLEF 2026 — Modeling Report</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="page-wrap">
""")

# ── COVER ────────────────────────────────────────────────────────────────────
emit("""
<div class="cover">
  <h1>BirdCLEF 2026</h1>
  <div class="subtitle">Modeling Hypotheses &amp; Strategy Report</div>
  <div class="subtitle">Synthesis of EDA Sections A–I &nbsp;|&nbsp; Phase 6 Deliverable</div>
  <hr class="cover-divider">
  <div class="cover-stats">
    <div class="cover-stat"><div class="label">Target Species</div>
      <div class="value">234 (162 birds · 35 frogs · 28 insects · 8 mammals · 1 reptile)</div></div>
    <div class="cover-stat"><div class="label">Training Clips</div>
      <div class="value">35,549 — 206 of 234 species have clips</div></div>
    <div class="cover-stat"><div class="label">Zero-Clip Species</div>
      <div class="value">28 — supervised only via 66 labeled soundscape windows</div></div>
    <div class="cover-stat"><div class="label">Train Soundscapes</div>
      <div class="value">10,658 total — only 0.6% have expert window labels</div></div>
    <div class="cover-stat"><div class="label">Evaluation Metric</div>
      <div class="value">Macro-averaged ROC-AUC (skip classes absent in test)</div></div>
    <div class="cover-stat"><div class="label">Inference Constraint</div>
      <div class="value">CPU-only Kaggle notebook · 90-minute hard limit</div></div>
    <div class="cover-stat"><div class="label">Domain Gap</div>
      <div class="value">Global XC/iNat clips → Pantanal passive ARU recorders (SEVERE)</div></div>
    <div class="cover-stat"><div class="label">Deadline</div>
      <div class="value">June 3, 2026 — ~8 weeks remaining</div></div>
  </div>
</div>
""")

# ── NAV ──────────────────────────────────────────────────────────────────────
emit("""
<nav>
  <a href="#s1">§1 Structure</a>
  <a href="#s2">§2 Hypotheses</a>
  <a href="#s3">§3 Pipeline</a>
  <a href="#s4">§4 Validation</a>
  <a href="#s5">§5 Augmentation</a>
  <a href="#s6">§6 Prior Data</a>
  <a href="#s7">§7 Risk Register</a>
  <a href="#s8">§8 Experiment Plan</a>
  <a href="#s9">§9 Open Questions</a>
</nav>
""")

emit('<div class="content">')

# ── TOC ──────────────────────────────────────────────────────────────────────
emit(h1("Contents"))
emit("""
<ul class="toc-list">
  <li><a href="#s1"><span class="toc-num">§1</span> Competition Structure — What the Model Must Actually Do</a></li>
  <li><a href="#s2"><span class="toc-num">§2</span> Ranked Modeling Hypotheses (Top 12)</a></li>
  <li><a href="#s3"><span class="toc-num">§3</span> Data Pipeline Architecture</a></li>
  <li><a href="#s4"><span class="toc-num">§4</span> Validation Strategy</a></li>
  <li><a href="#s5"><span class="toc-num">§5</span> Augmentation Plan (Evidence-Based)</a></li>
  <li><a href="#s6"><span class="toc-num">§6</span> Prior Data Decision</a></li>
  <li><a href="#s7"><span class="toc-num">§7</span> Risk Register</a></li>
  <li><a href="#s8"><span class="toc-num">§8</span> Phased Experiment Plan</a></li>
  <li><a href="#s9"><span class="toc-num">§9</span> Open Questions &amp; Gaps</a></li>
</ul>
""")

# ============================================================================
# §1 — Competition Structure
# ============================================================================
emit(h1("§1 — Competition Structure: What the Model Must Actually Do", "s1"))
emit(p(
    "Before any modeling hypothesis can be tested, the prediction contract must be "
    "understood precisely. Misunderstanding the supervision unit vs. the scoring unit "
    "is the most common source of wasted effort in BirdCLEF-style competitions."
))

emit(h2("1.1 — Prediction Contract (Train vs. Score)"))
emit(table(
    ["Axis", "Training Side", "Scoring Side"],
    [
        ["Unit", "Whole clip (3–120 s)", "5-second window"],
        ["Label type", "Clip-level primary + secondary", "Per-window presence / absence"],
        ["Label count", "Mostly 1; 12.3% have secondary labels", "3–5 species per window (EDA §F)"],
        ["Supervision", "35K clips + 1,478 expert soundscape windows", "Hidden test soundscapes"],
        ["Geography", "Global (lat −55 to +70)", "Pantanal, Brazil only"],
        ["Recorder", "Close-mic / phone / field recorder", "Passive ARU deployment"],
    ]
))
emit(infobox(
    "<strong>Core tension:</strong> The model is trained on close-mic clips with whole-clip labels "
    "and must generalize to passive soundscape recorders predicting at 5-second resolution. "
    "This is a <em>three-way mismatch</em>: recording device, label granularity, and geographic domain. "
    "Every design decision must close at least one of these gaps.",
    "blue"
))

emit(h2("1.2 — Metric Implications"))
emit(ul([
    "<strong>Macro ROC-AUC</strong> — each of 234 classes gets equal weight regardless of frequency. "
    "A species appearing in 2 test windows matters as much as one appearing in 2,000.",
    "Metric is <strong>threshold-free</strong> — calibration of absolute probability values "
    "does not matter. Relative ranking within each class is what counts.",
    "<strong>Classes absent in test are skipped</strong> — but you cannot know which ones in advance. "
    "Do not deprioritize rare species.",
    "Improving common-species AUC by sacrificing rare-species AUC is net-neutral at best "
    "and net-negative under macro averaging.",
    "This metric strongly rewards models that <em>discriminate</em> within rare classes, "
    "not just models that avoid false positives overall.",
]))

emit(h2("1.3 — CPU Inference Constraint (90-min, no GPU)"))
emit(ul([
    "Hard limit on model size. EfficientNet-B4 and below are feasible; B6+ and large ViTs will time out.",
    "Mel spectrogram extraction must be batched — per-window librosa calls add ~2× overhead.",
    "ONNX export or <code>torch.compile</code> will likely be necessary for competitive inference speed.",
    "Top BC2024/2025 solutions used EfficientNet-B0 to B2 or small ConvNext variants with ONNX export.",
    "Plan model architecture around inference budget first, then optimize within that envelope.",
]))

# ============================================================================
# §2 — Ranked Modeling Hypotheses
# ============================================================================
emit(h1("§2 — Ranked Modeling Hypotheses (Top 12)", "s2"))
emit(p(
    "Each hypothesis is grounded in a specific EDA finding. Priority is assigned by "
    "<strong>expected AUC impact × implementation feasibility</strong>. "
    "Every hypothesis includes: the claim, the EDA evidence, the expected impact, and the test approach."
))

# Tier 1
emit(h2("Tier 1 — Critical (must test first)"))
emit(infobox(
    "These four hypotheses address the dominant risks identified in EDA. "
    "None of the other hypotheses should be tested before these are validated on OOF CV.",
    "red"
))

emit(table(
    ["#", "Hypothesis", "EDA Evidence", "Expected Impact", "How to Test"],
    [
        [
            '<span class="pri-crit">H1</span>',
            "<strong>Background mixing with unlabeled Pantanal soundscapes "
            "closes the clip→soundscape domain gap more than any other single intervention.</strong>",
            "Section G5, F1: clip-to-soundscape domain gap is the primary risk. "
            "10,592 unlabeled soundscapes provide real Pantanal noise backgrounds at no label cost.",
            badge("Large", "red"),
            "Baseline OOF AUC vs. +background mixing. "
            "Expected: +0.04–0.08 AUC. "
            "Monitor per-taxon AUC change."
        ],
        [
            '<span class="pri-crit">H2</span>',
            "<strong>Waveform mixup (λ~Beta(0.4,0.4)) with soft multi-label targets "
            "teaches the model co-presence of 3–5 species, matching the test window structure.</strong>",
            "Section F2, C: 87% of expert-labeled windows have 2+ co-active species. "
            "Training clips are mostly single-label. This gap hurts multi-label discrimination.",
            badge("High", "red"),
            "Baseline OOF AUC vs. +waveform mixup. "
            "Expected: +0.02–0.05. "
            "Check that rare-class AUC does not drop."
        ],
        [
            '<span class="pri-crit">H3</span>',
            "<strong>Energy gating — discarding silent 5s windows as unlabeled — "
            "reduces label noise more than any augmentation.</strong>",
            "Section E3, F3: 14% mean silence ratio; some clips 90% silent. "
            "Assigning clip-level labels to silent windows creates systematic false positives.",
            badge("High", "red"),
            "Train with and without energy gating. "
            "Compare OOF precision on silent-window positives. "
            "Threshold sweep: −40, −45, −50 dBFS."
        ],
        [
            '<span class="pri-crit">H4</span>',
            "<strong>Site-stratified CV avoids inflated OOF AUC caused by "
            "site-level acoustic leakage from recorder fingerprints and location clustering.</strong>",
            "Section G1, G2: top-5 authors = 40%+ of clips; "
            "site-level recorder fingerprints are learnable. "
            "Random CV splits leak site characteristics.",
            badge("High", "red"),
            "Compare random 5-fold OOF vs. site-stratified 5-fold OOF. "
            "Gap > 0.05 = leakage confirmed. "
            "Use site-stratified as the primary metric."
        ],
    ],
    col_classes=["col-num", "", "", "", ""]
))
# Apply row styles manually via post-processing below — done with class on tr instead
# We'll use a workaround: inject row-level class via raw HTML

emit(h2("Tier 2 — High Priority (test in first 2 weeks after baseline)"))

emit(table(
    ["#", "Hypothesis", "EDA Evidence", "Expected Impact", "How to Test"],
    [
        [
            '<span class="pri-high">H5</span>',
            "<strong>BirdCLEF 2025 Pantanal clips and soundscapes are the highest-value "
            "external data — adding them directly expands labeled Pantanal coverage.</strong>",
            "Section H: BC2025 covers the same Pantanal region, same ARU recorder type, "
            "same multi-taxon scope. Highest domain match of any prior competition.",
            badge("Med–High", "orange"),
            "Add BC2025 data (after URL dedup). "
            "Measure OOF AUC change. "
            "Flag: check if BC2025 soundscape sites overlap with BC2026 test sites."
        ],
        [
            '<span class="pri-high">H6</span>',
            "<strong>Backbone pre-training on all BirdCLEF clip audio (2021–2025), "
            "then fine-tuning on 2026+Pantanal data, "
            "gives better representations for rare species than training from scratch.</strong>",
            "Section H: prior competition clips share XC/iNat sources. "
            "Species-level acoustic features transfer across geographies even if domain differs.",
            badge("Med–High", "orange"),
            "Compare scratch training vs. pre-trained backbone. "
            "Focus comparison on rare-class AUC (< 10 clips). "
            "Expected gain concentrated in under-represented species."
        ],
        [
            '<span class="pri-high">H7</span>',
            "<strong>Per-clip peak normalization before mel extraction removes "
            "the 600× loudness variance that confounds spectral features with recording level.</strong>",
            "Section E3: RMS distribution spans 600× dynamic range across recordings. "
            "Without normalization, the model partially learns loudness as a species feature.",
            badge("Medium", "orange"),
            "Train with and without normalization. "
            "Mostly affects convergence speed and feature quality "
            "rather than final AUC. Low-risk, do by default."
        ],
        [
            '<span class="pri-high">H8</span>',
            "<strong>A multi-taxon head architecture (shared backbone, separate dense "
            "heads per taxon) outperforms a flat 234-class head "
            "by allocating capacity to each taxon's frequency range.</strong>",
            "Section A §3: frogs are 200–4000 Hz, insects 4000–16000 Hz, birds 1000–8000 Hz. "
            "A flat head must learn all three spectral regimes simultaneously "
            "with shared capacity.",
            badge("Medium", "orange"),
            "Flat head baseline vs. 5-head multi-taxon model. "
            "Report per-taxon AUC separately. "
            "Expected: +0.03–0.06 on Amphibia/Insecta."
        ],
    ],
    col_classes=["col-num", "", "", "", ""]
))

emit(h2("Tier 3 — Medium Priority (test after baseline is stable)"))

emit(table(
    ["#", "Hypothesis", "EDA Evidence", "Expected Impact", "How to Test"],
    [
        [
            '<span class="pri-medium">H9</span>',
            "<strong>Semi-supervised learning on 10,592 unlabeled soundscapes "
            "provides additional Pantanal-domain signal for rare and zero-clip species.</strong>",
            "Section B: 10,592 soundscapes are unlabeled but come from the exact "
            "deployment domain. Even without labels, they carry domain-specific acoustic patterns.",
            badge("Medium", "blue"),
            "Pseudo-label with baseline model (threshold > 0.5). "
            "Retrain on pseudo-labeled + real labeled. "
            "<em>Only start this after H1–H4 are validated.</em>"
        ],
        [
            '<span class="pri-medium">H10</span>',
            "<strong>Time-of-day conditioning (sin/cos hour encoding) "
            "improves prediction accuracy because species activity is time-dependent.</strong>",
            "Section E4: dawn chorus peak in soundscapes. "
            "Test row_id encodes datetime. "
            "Section D: seasonal patterns visible in soundscape metadata.",
            badge("Low–Med", "blue"),
            "Add <code>sin(2π·hour/24)</code>, <code>cos(2π·hour/24)</code> as input features. "
            "Free feature, low implementation cost. "
            "Expected: small gain on temporally-structured species."
        ],
        [
            '<span class="pri-medium">H11</span>',
            "<strong>Two-stage detection for insect sonotypes "
            "(detect parent taxon → sub-classify sonotype) "
            "recovers near-zero AUC on all 25 insect sonotype classes.</strong>",
            "Section B §5: 25 insect sonotypes have zero training clips. "
            "Their only supervision is the 66 labeled soundscape windows. "
            "Flat 234-class model produces near-random scores for these classes.",
            badge("Medium", "blue"),
            "First: visualize sonotype spectrograms — are they visually distinguishable? "
            "If yes: train parent-taxon detector, then sonotype sub-classifier. "
            "If no: accept low AUC; not worth engineering cost early."
        ],
        [
            '<span class="pri-medium">H12</span>',
            "<strong>Post-processing by Pantanal species prevalence "
            "reduces systematic over-prediction of globally-common XC species "
            "that are locally rare or absent in Pantanal.</strong>",
            "Section G5: train clip geographic distribution is global; test is Pantanal. "
            "Model will over-score species abundant globally but Pantanal-absent.",
            badge("Low–Med", "blue"),
            "Estimate species prevalence from labeled soundscape windows. "
            "Multiply raw scores by prior. "
            "Under ROC-AUC this only helps by re-ranking across windows; "
            "measure OOF before applying."
        ],
    ],
    col_classes=["col-num", "", "", "", ""]
))

emit(infobox(
    "<strong>Priority order:</strong> "
    "H1 → H2 → H3 → H4 → H7 → H5 → H6 → H8 → H9 → H10 → H11 → H12<br>"
    "<em>Do not test H9–H12 until H1–H4 are verified on OOF CV. "
    "Stacking unvalidated interventions produces uninterpretable results.</em>",
    "yellow"
))

# ============================================================================
# §3 — Data Pipeline
# ============================================================================
emit(h1("§3 — Data Pipeline Architecture", "s3"))

emit(h2("3.1 — Input Sources (Ranked by In-Domain Value)"))
emit(table(
    ["Source", "Size", "In-Domain?", "Labels?", "Recommended Use", "Risk"],
    [
        ["Train soundscapes (BC2026)", "10,658 ogg",
         badge("Pantanal ARU", "green"),
         "66 labeled (1,478 windows)",
         "Primary training + all as backgrounds",
         "Site S22 dominates labeled subset"],
        ["Train clips (BC2026)", "35,549 ogg",
         badge("Partial — global", "orange"),
         "Clip-level",
         "Training with windowing + energy gating",
         "Global geography ≠ Pantanal domain"],
        ["BC2025 clips + soundscapes", "~50K est.",
         badge("High — same region", "green"),
         "Clip + soundscape window",
         "Add directly after URL dedup",
         "Possible test-set site overlap"],
        ["Prior BC clips 2021–2024", "Large",
         badge("Low — diff regions", "orange"),
         "Clip-level",
         "Backbone pre-training only",
         "Hurts calibration if mixed directly"],
        ["Additional Neotropical XC", "Variable",
         badge("Medium — regional", "blue"),
         "Clip-level",
         "Rare/zero-clip species only",
         "May introduce non-Pantanal recordings"],
        ["Non-Pantanal soundscapes", "Various",
         badge("NO — wrong domain", "red"),
         "Various",
         "Do NOT use as domain training data",
         "Confuses domain adaptation"],
    ]
))

emit(h2("3.2 — Clip Processing Pipeline"))
emit(table(
    ["Step", "Action", "Why"],
    [
        ["1. Load &amp; resample",
         "Load OGG → resample to 32 kHz mono",
         "All BC2026 clips already 32 kHz. Minimal overhead."],
        ["2. Peak normalization",
         "Normalize peak to 0.9 before any windowing",
         "Removes 600× loudness variance. Apply universally."],
        ["3. Window extraction",
         "Slice into non-overlapping 5s windows. "
         "Short clips (&lt;5s): pad with zeros and treat as single window.",
         "Matches the 5s scoring unit exactly."],
        ["4. Energy gating",
         "Compute RMS of each window. "
         "If RMS &lt; −45 dBFS → mark as <code>background_only</code> (no species label). "
         "Do NOT assign primary_label to silent windows.",
         "Removes most systematic false-positive labels from silence."],
        ["5. Background mixing",
         "Sample random 5s window from unlabeled Pantanal soundscape. "
         "Mix at SNR ~ Uniform(−5, +10) dB. Background label = all-zeros.",
         "Highest-ROI step. Directly bridges clip→soundscape domain gap."],
        ["6. Waveform mixup",
         "With prob 0.5: sample second clip. "
         "Mix x = λ·x1 + (1-λ)·x2, λ ~ Beta(0.4, 0.4). "
         "Blend labels: y = λ·y1 + (1-λ)·y2 (soft labels, multi-label BCE).",
         "Trains multi-label co-presence matching test windows."],
        ["7. Mel spectrogram",
         "n_mels=128, hop_length=320 (10ms), n_fft=1024, fmin=50, fmax=14000 Hz. "
         "Log-amplitude. Result: 128×500 per 5s window.",
         "Covers frog (200–4k), bird (1–8k), insect (4–16k) ranges."],
        ["8. SpecAugment (P3 only)",
         "Freq masking 1–2 bands (≤10% mel bins), "
         "time masking 1 block (≤8% frames). "
         "<em>Only after P1–P2 are validated.</em>",
         "Breaks recorder-fingerprint memorization (author EQ patterns)."],
    ]
))

emit(h2("3.3 — Label Construction"))
emit(ul([
    "<strong>Clip windows:</strong> assign primary_label + secondary_labels to all non-silent windows. "
    "Accept clip-level label noise; energy gating handles the worst cases.",
    "<strong>Soundscape labeled windows:</strong> use exact per-window expert labels as gold standard. "
    "Oversample these relative to clips in the training loop.",
    "<strong>Unlabeled soundscape windows (as background mix source):</strong> label as all-zeros. "
    "Do NOT infer labels from clip overlap.",
    "<strong>Rare-species oversampling:</strong> weight each labeled soundscape window by "
    "<code>1/sqrt(rarest_class_freq)</code> to ensure all 234 classes appear in every epoch.",
    "<strong>Soft labels for mixup:</strong> BCE loss must support continuous [0,1] targets. "
    "Confirm framework compatibility before training.",
]))

# ============================================================================
# §4 — Validation
# ============================================================================
emit(h1("§4 — Validation Strategy", "s4"))
emit(p(
    "A realistic validation scheme is the single most important infrastructure decision. "
    "An optimistic CV scheme means every experiment looks better than it actually is. "
    "BirdCLEF competitions are notorious for site-level acoustic leakage inflating OOF scores."
))

emit(h2("4.1 — Why Random K-Fold Is Wrong Here"))
emit(ul([
    "<strong>Author leakage:</strong> Top-5 authors = 40%+ of clips. Random splits put clips "
    "from the same recording session in both train and validation. "
    "The model memorizes recorder fingerprints.",
    "<strong>Location leakage:</strong> Nearby recordings share forest background noise. "
    "GPS-clustered clips in both train/val look like generalization but isn't.",
    "<strong>Site leakage in soundscapes:</strong> Site S22 = 64% of labeled windows. "
    "A random split across S22 windows leaks site acoustics — not real generalization.",
    "<strong>Temporal leakage:</strong> Same recorder makes multiple trips to the same site in the same year. "
    "Near-duplicate recording clusters (EDA §G3) leak across random splits.",
]))

emit(h2("4.2 — Recommended Validation Scheme"))
emit(table(
    ["Scheme", "How to Split", "When to Use"],
    [
        ["Site-stratified 5-fold (clips)",
         "Group clips by GPS cluster (K-means on lat/lon → ~20 clusters). "
         "Each fold holds out complete geographic clusters.",
         "Primary CV for clip-based experiments"],
        ["Author-stratified 5-fold (clips)",
         "Group clips by author/recorder ID. "
         "Each fold holds out complete author groups.",
         "Cross-check: should give similar result to site-stratified"],
        ["Soundscape site holdout",
         "Reserve 3 non-S22 soundscape sites as held-out evaluation. "
         "Train on remaining 20 sites. "
         "S22 should not be the holdout — it is too dominant.",
         "Primary CV for soundscape-based experiments"],
        ["Taxonomic-stratified sanity check",
         "Ensure each fold contains examples from all 5 taxon classes. "
         "Verify Amphibia and Insecta are not concentrated in one fold.",
         "Sanity check on all fold configurations"],
    ]
))

emit(infobox(
    "<strong>Key rule:</strong> Always report OOF AUC separately per taxonomic class "
    "(Aves, Amphibia, Insecta, Mammalia, Reptilia). "
    "A high overall macro AUC driven by bird classes while Amphibia/Insecta AUC is near 0.5 "
    "is a false positive. The leaderboard will expose this.",
    "orange"
))

# ============================================================================
# §5 — Augmentation
# ============================================================================
emit(h1("§5 — Augmentation Plan (Evidence-Based)", "s5"))
emit(p(
    "Every augmentation below is tied to a specific EDA finding. "
    "Generic augmentations not supported by evidence are excluded. "
    "The list is ordered by implementation priority."
))

emit(h2("5.1 — Recommended Augmentations (Ordered by Priority)"))
emit(table(
    ["Priority", "Augmentation", "Problem It Addresses", "EDA Source", "Caveat"],
    [
        [badge("P1 Must Have", "red"),
         "<strong>Energy gating</strong> (preprocessing)",
         "14% mean silence; some clips 90% silent. "
         "Clip-level labels assigned to silent windows → systematic false positives.",
         "§E3, §F3",
         "Threshold tuning: test −45, −40, −50 dBFS on OOF before fixing."],
        [badge("P1 Must Have", "red"),
         "<strong>Peak normalization</strong> (preprocessing)",
         "600× loudness variance. Without normalization, model learns recording level as a species cue.",
         "§E3",
         "Apply per-clip before windowing. No hyperparameters."],
        [badge("P1 Must Have", "red"),
         "<strong>Background mixing</strong> (soundscape overlay)",
         "Clip→soundscape domain gap is the primary train/test mismatch. "
         "10,592 unlabeled Pantanal soundscapes provide real noise backgrounds.",
         "§G5, §F1",
         "Pre-screen soundscapes for extreme artifacts (rain, wind) first. "
         "SNR range: −5 to +10 dB."],
        [badge("P1 Must Have", "red"),
         "<strong>Waveform mixup</strong> (λ~Beta(0.4,0.4))",
         "Training is mostly single-label; test windows have 3–5 co-active species.",
         "§F2, §C",
         "Requires soft-label BCE loss. Confirm framework supports float targets."],
        [badge("P2 High Value", "orange"),
         "<strong>Rare-species oversampling</strong>",
         "28 zero-clip species; many with &lt;5 clips. Standard sampling skips them for entire epochs.",
         "§B, §C",
         "Weight = 1/sqrt(class_freq). Watch for common-class degradation."],
        [badge("P3 Medium", "blue"),
         "<strong>SpecAugment</strong> (conservative)",
         "Top-5 author recorder fingerprints = 40%+ of data. "
         "Frequency masking breaks EQ-pattern memorization.",
         "§G2",
         "Max 10% freq bins, max 8% time frames. Validate on OOF before enabling permanently."],
        [badge("P3 Medium", "blue"),
         "<strong>Soft saturation</strong> (tanh clipping sim)",
         "~8% of clips have clipping artifacts. Model should be robust to saturation distortion.",
         "§E3",
         "Apply to 20–30% of clips randomly only."],
        [badge("P3 Low", "blue"),
         "<strong>Time-of-day feature</strong> (not augmentation)",
         "Dawn chorus peak; species activity is time-conditional. Test row_id encodes datetime.",
         "§E4",
         "Encode as sin(2π·hour/24), cos(2π·hour/24). Free, low-risk feature."],
    ]
))

emit(h2("5.2 — Augmentations Explicitly Excluded"))
emit(table(
    ["Excluded Augmentation", "Reason"],
    [
        ["Large pitch shift (±4+ semitones)",
         "Frog and insect calls are pitch-specific for species ID. "
         "Large shifts change species identity in non-bird taxa. "
         "Use ≤±2 semitones for birds only."],
        ["Time stretching &gt;±20%",
         "Temporal rhythm of calls (rate, duty cycle) is species-diagnostic, "
         "especially for insects. Heavy stretching distorts these cues."],
        ["Additive Gaussian / white noise",
         "Pantanal background noise is structured (wind, rain, water, co-species) — not white. "
         "Use real soundscape backgrounds instead."],
        ["Spectrogram CutMix",
         "Pasting spectrogram patches creates frequency-localized chimeras that are acoustically "
         "impossible. Waveform mixup is physically meaningful; spectrogram CutMix is not."],
        ["Random crop to &lt;2.5s",
         "Creates training examples shorter than half a prediction window. "
         "Label assignment becomes meaningless for sub-2.5s fragments."],
        ["Frequency shifting (full spectrum)",
         "Harmonic structures are key species discriminators. Shifting destroys harmonic ratios."],
    ]
))

# ============================================================================
# §6 — Prior Data
# ============================================================================
emit(h1("§6 — Prior Data Decision", "s6"))
emit(p(
    "This section answers: which prior BirdCLEF competition data should be used, "
    "in what capacity, and what should be avoided. "
    "The answer is not binary — different datasets serve different purposes."
))

emit(table(
    ["Dataset", "Geography", "Taxon Overlap", "Domain Match", "Recommended Use", "Risk"],
    [
        ["BC2025 clips + soundscapes",
         "Pantanal, Brazil",
         badge("High", "green"),
         badge("Excellent", "green"),
         "Add directly to training after URL dedup",
         "Possible test-site overlap with BC2026 — check dates"],
        ["BC2024 clips",
         "Multiple regions",
         badge("Moderate", "orange"),
         badge("Poor", "red"),
         "Backbone pre-training only",
         "Hurts calibration if mixed directly with BC2026"],
        ["BC2023 / BC2022 clips",
         "Global / India / Africa",
         badge("Low", "red"),
         badge("Very poor", "red"),
         "Backbone pre-training only if needed",
         "Geographic and species mismatch"],
        ["BC2021 clips",
         "Cornell / North America",
         badge("Very low", "red"),
         badge("Very poor", "red"),
         "Skip or very broad pre-training only",
         "Unlikely to help; may hurt calibration"],
        ["Non-Pantanal soundscapes (any year)",
         "Various non-Pantanal",
         badge("Low", "red"),
         badge("Wrong domain", "red"),
         "Do NOT use as domain training data",
         "Confuses soundscape domain adaptation"],
    ]
))

emit(infobox(
    "<strong>Decision:</strong> Use BC2025 as in-domain training data (high priority). "
    "Use BC2021–2024 clips for backbone pre-training only. "
    "Do NOT use non-Pantanal soundscapes as soundscape domain training data.<br><br>"
    "<strong>Mandatory before combining any prior data:</strong> "
    "deduplicate by Xeno-Canto URL. Prior competitions draw from the same XC pool. "
    "Duplicate clips in both train and validation will inflate OOF AUC.",
    "green"
))

# ============================================================================
# §7 — Risk Register
# ============================================================================
emit(h1("§7 — Risk Register", "s7"))

emit(table(
    ["Risk", "Severity", "EDA Source", "Mitigation"],
    [
        ["Clip→soundscape domain gap",
         badge("CRITICAL", "red"),
         "§G5, §F1",
         "H1: Background mixing. H4: Site-stratified CV. "
         "Monitor per-site AUC during training."],
        ["Label noise from clip windowing",
         badge("HIGH", "red"),
         "§E3, §F3",
         "H3: Energy gating. Discard silent windows. "
         "Accept residual noise for non-silent windows."],
        ["28 zero-clip species (25 insect sonotypes + 3 frogs)",
         badge("HIGH", "red"),
         "§B §5, §C",
         "Oversample labeled soundscape windows. "
         "Accept near-zero AUC on insect sonotypes initially. "
         "H11: Two-stage detection in Phase 4."],
        ["Site S22 dominance in labeled soundscapes (64% of windows)",
         badge("HIGH", "red"),
         "§G4",
         "Hold out non-S22 sites for evaluation. "
         "Do not train final model exclusively on S22-derived labels."],
        ["Author/recorder fingerprint memorization",
         badge("MEDIUM", "orange"),
         "§G2",
         "H4: Author-stratified CV. P3 SpecAugment. "
         "Monitor gap between author-stratified and random CV OOF."],
        ["Seasonal mismatch (test ≈ Feb 2025, start of dry season)",
         badge("MEDIUM", "orange"),
         "§D, §E4",
         "H10: Add time-of-day and month features. "
         "Weight training examples less from extreme wet-season months."],
        ["Rating-zero clips (36% of training clips, quality unknown)",
         badge("MEDIUM", "orange"),
         "§A §4A",
         "Manually audit 50 random rating-0 clips. "
         "If systematic noise: quality-stratified sampling. "
         "If clean: keep all clips."],
        ["XC duplicate clips across BC2025–BC2026 inflating OOF",
         badge("MEDIUM", "orange"),
         "§H",
         "Deduplicate by XC URL before combining datasets."],
        ["CPU inference timeout on complex models",
         badge("HIGH", "red"),
         "§A §12",
         "Design around ≤ EfficientNet-B2 or ConvNext-Tiny. "
         "Profile inference early. ONNX export for speed."],
        ["Macro AUC masking poor non-bird taxon performance",
         badge("MEDIUM", "orange"),
         "§A §3",
         "Track OOF AUC per taxon separately. "
         "Target: all taxon classes ≥ 0.70. "
         "Non-bird AUC below 0.60 requires targeted intervention."],
    ]
))

# ============================================================================
# §8 — Experiment Plan
# ============================================================================
emit(h1("§8 — Phased Experiment Plan", "s8"))
emit(p(
    "Experiments are sequenced to maximize information gain per GPU-hour. "
    "Each phase must produce a validated OOF AUC before moving to the next. "
    "Do not stack unvalidated interventions."
))

emit(table(
    ["Phase", "Target Dates", "Goals", "Exit Criterion"],
    [
        ["<strong>Phase 0</strong><br>Infrastructure",
         "Weeks 1–2<br>(by Apr 26)",
         "<ul>"
         "<li>Clip windowing + energy gating pipeline</li>"
         "<li>Site-stratified 5-fold splits</li>"
         "<li>Batched mel spectrogram extraction</li>"
         "<li>Baseline EfficientNet-B0, flat 234-class</li>"
         "<li>OOF AUC tracking per taxon</li>"
         "</ul>",
         "Reproducible baseline OOF AUC &gt; 0.65<br>"
         "Inference time &lt; 60 min on CPU"],
        ["<strong>Phase 1</strong><br>Core Augmentation",
         "Weeks 2–3<br>(by May 3)",
         "<ul>"
         "<li>Peak normalization (H7)</li>"
         "<li>Background mixing P1 (H1)</li>"
         "<li>Waveform mixup P1 (H2)</li>"
         "<li>Rare-species oversampling (H3)</li>"
         "</ul>",
         "+0.03 OOF AUC vs Phase 0<br>"
         "Non-bird taxon AUC &gt; 0.60"],
        ["<strong>Phase 2</strong><br>Data Expansion",
         "Weeks 3–4<br>(by May 10)",
         "<ul>"
         "<li>Add BC2025 data + dedup (H5)</li>"
         "<li>Add Neotropical XC for zero-clip species</li>"
         "<li>SpecAugment P3 validation</li>"
         "</ul>",
         "+0.02 OOF AUC vs Phase 1"],
        ["<strong>Phase 3</strong><br>Architecture",
         "Weeks 4–5<br>(by May 17)",
         "<ul>"
         "<li>Multi-taxon head experiment (H8)</li>"
         "<li>Backbone pre-training on BC2021–2024 (H6)</li>"
         "<li>Time-of-day conditioning (H10)</li>"
         "</ul>",
         "+0.02 OOF AUC vs Phase 2<br>"
         "Decide: multi-taxon head yes/no"],
        ["<strong>Phase 4</strong><br>Semi-supervised",
         "Weeks 5–6<br>(by May 24)",
         "<ul>"
         "<li>Pseudo-labeling on unlabeled soundscapes (H9)</li>"
         "<li>Two-stage insect sonotype detection (H11)</li>"
         "<li>Model ensemble (2–3 seeds minimum)</li>"
         "</ul>",
         "+0.01–0.02 OOF AUC<br>"
         "Insect sonotype AUC &gt; 0.55"],
        ["<strong>Phase 5</strong><br>Final Polish",
         "Weeks 6–8<br>(by Jun 3)",
         "<ul>"
         "<li>Inference optimization (ONNX / torch.compile)</li>"
         "<li>Prevalence post-processing experiment (H12)</li>"
         "<li>Final ensemble submission</li>"
         "<li>Submission slot management</li>"
         "</ul>",
         "Submission time &lt; 85 min<br>"
         "Final ensemble &gt; Phase 4 OOF"],
    ]
))

emit(infobox(
    "<strong>Checkpoint rule:</strong> If a phase does not meet its exit criterion after 2 attempts, "
    "diagnose root cause before advancing. "
    "Do not move to the next phase on faith — each phase builds on validated results from the previous one.",
    "orange"
))

# ============================================================================
# §9 — Open Questions
# ============================================================================
emit(h1("§9 — Open Questions &amp; Gaps", "s9"))
emit(p(
    "These cannot be resolved from EDA alone and require either direct experimentation "
    "or information that becomes available later in the competition."
))

emit(table(
    ["Question", "Why It Matters", "How to Resolve"],
    [
        ["Which species actually appear in the test soundscapes?",
         "Classes absent in test are skipped from macro AUC. "
         "Knowing test class coverage would change rare-class priority.",
         "Cannot know until final scoring. "
         "Treat all 234 classes as potentially present."],
        ["How many test soundscapes are there?",
         "Determines inference time budget per soundscape "
         "and maximum architecture size.",
         "Check sample_submission row count. "
         "Estimate: ~50–200 soundscapes based on prior year competition sizes."],
        ["What quality are the rating-zero clips?",
         "36% of training clips are unrated. "
         "If many are low quality, they should be down-weighted or excluded.",
         "Listen to 50 random rating-0 clips manually. "
         "Run SNR estimation script on all rating-0 clips."],
        ["Are BC2025 soundscapes from the same physical sites as BC2026 test soundscapes?",
         "If yes, site-level acoustic leakage between BC2025 and BC2026 test is possible.",
         "Compare site metadata. "
         "If site IDs overlap: exclude overlapping BC2025 soundscapes from domain training."],
        ["Are insect sonotypes acoustically distinguishable in practice?",
         "If sonotype boundaries are visually/acoustically ambiguous even to experts, "
         "the 25-class insect block may be inherently unlearnable.",
         "Spectrogram visualization of sonotype examples. "
         "If visually indistinguishable: accept low AUC; skip H11 engineering cost."],
        ["What is the test soundscape recording date distribution?",
         "Feb 2025 is all we know from sample submission. "
         "If test spans multiple months, seasonal conditioning matters more.",
         "Infer from final submission row_ids as more test info becomes available."],
        ["Does BC2026 evaluation skip truly zero-positive classes or uses all 234?",
         "The skip policy determines effective N for macro average. "
         "If most 234 classes appear, rare classes matter more.",
         "Check competition FAQ / discussion forum. "
         "Assume all 234 classes are possible."],
    ]
))

# ── Summary box ──────────────────────────────────────────────────────────────
emit(f"""
<div class="summary-box">
  <h3>Three Headline Findings From This Analysis</h3>
  <ul>
    <li><strong>The domain gap (clip → soundscape) is the #1 problem.</strong>
    Background mixing with unlabeled Pantanal soundscapes (H1) is the highest-ROI single
    intervention. Do this before anything else.</li>
    <li><strong>Validation scheme determines everything.</strong>
    Site-stratified 5-fold CV is mandatory (H4).
    Random splits produce inflated OOF that doesn't match leaderboard position.</li>
    <li><strong>Non-bird taxa are the leaderboard differentiator.</strong>
    Frogs, insects, and mammals are 30% of targets.
    Track their AUC separately. Most competitors will under-optimize for them.</li>
  </ul>
</div>
""")

emit("</div>")  # end .content

# ── Footer ───────────────────────────────────────────────────────────────────
emit(f"""
<footer>
  BirdCLEF 2026 — Modeling Hypotheses &amp; Strategy Report &nbsp;|&nbsp;
  Generated {DATE} &nbsp;|&nbsp;
  Synthesized from EDA Sections A–I &nbsp;|&nbsp;
  reports/modeling_report.html
</footer>
""")

emit("</div></body></html>")  # end .page-wrap

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
html = "\n".join(parts)
OUT.write_text(html, encoding="utf-8")
size_kb = OUT.stat().st_size / 1000
print(f"Written: {OUT}  ({size_kb:.0f} KB)")
