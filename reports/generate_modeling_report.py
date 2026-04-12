"""
BirdCLEF 2026 — Modeling Hypotheses & Strategy Report Generator
================================================================
Synthesizes all EDA findings (Sections A–I) into a ranked modeling
strategy PDF. Run from repo root:

    python reports/generate_modeling_report.py

Output: reports/modeling_report.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor
import datetime

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUT_PDF = Path(__file__).parent / "modeling_report.pdf"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_RED    = HexColor("#c0392b")
C_ORANGE = HexColor("#e67e22")
C_BLUE   = HexColor("#2980b9")
C_GREEN  = HexColor("#27ae60")
C_GREY   = HexColor("#7f8c8d")
C_DARK   = HexColor("#2c3e50")
C_LIGHT  = HexColor("#ecf0f1")
C_YELLOW = HexColor("#f39c12")
C_BG_RED    = HexColor("#fdf0f0")
C_BG_ORANGE = HexColor("#fef9e7")
C_BG_BLUE   = HexColor("#eaf4fb")
C_BG_GREEN  = HexColor("#eafaf1")
C_ACCENT = HexColor("#1a5276")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
BASE = getSampleStyleSheet()

def make_style(name, parent="Normal", **kwargs):
    return ParagraphStyle(name, parent=BASE[parent], **kwargs)

TITLE_STYLE = make_style("Title", "Title",
    fontSize=26, textColor=C_ACCENT, spaceAfter=6, alignment=TA_CENTER,
    fontName="Helvetica-Bold")

SUBTITLE_STYLE = make_style("Subtitle", "Normal",
    fontSize=13, textColor=C_GREY, spaceAfter=4, alignment=TA_CENTER,
    fontName="Helvetica")

H1 = make_style("H1", "Heading1",
    fontSize=15, textColor=C_ACCENT, spaceBefore=18, spaceAfter=6,
    fontName="Helvetica-Bold", borderPad=2)

H2 = make_style("H2", "Heading2",
    fontSize=12, textColor=C_DARK, spaceBefore=12, spaceAfter=4,
    fontName="Helvetica-Bold")

H3 = make_style("H3", "Heading3",
    fontSize=10, textColor=C_DARK, spaceBefore=8, spaceAfter=3,
    fontName="Helvetica-BoldOblique")

BODY = make_style("Body", "Normal",
    fontSize=9.5, leading=14, spaceAfter=4, alignment=TA_JUSTIFY,
    fontName="Helvetica")

BODY_SMALL = make_style("BodySmall", "Normal",
    fontSize=8.5, leading=12, spaceAfter=3, alignment=TA_JUSTIFY,
    fontName="Helvetica")

BULLET = make_style("Bullet", "Normal",
    fontSize=9.5, leading=14, leftIndent=12, firstLineIndent=-8,
    spaceAfter=3, fontName="Helvetica")

CAPTION = make_style("Caption", "Normal",
    fontSize=8, textColor=C_GREY, alignment=TA_CENTER, spaceAfter=4,
    fontName="Helvetica-Oblique")

CODE_STYLE = make_style("Code", "Code",
    fontSize=8, leading=11, fontName="Courier", backColor=HexColor("#f4f4f4"),
    leftIndent=8, rightIndent=8, spaceAfter=6)

LABEL_RED    = make_style("LabelRed",    "Normal", fontSize=8.5, textColor=C_RED,    fontName="Helvetica-Bold")
LABEL_ORANGE = make_style("LabelOrange", "Normal", fontSize=8.5, textColor=C_ORANGE, fontName="Helvetica-Bold")
LABEL_BLUE   = make_style("LabelBlue",   "Normal", fontSize=8.5, textColor=C_BLUE,   fontName="Helvetica-Bold")
LABEL_GREEN  = make_style("LabelGreen",  "Normal", fontSize=8.5, textColor=C_GREEN,  fontName="Helvetica-Bold")

# ---------------------------------------------------------------------------
# Helper: shaded info box
# ---------------------------------------------------------------------------
class InfoBox(Flowable):
    def __init__(self, paragraphs, bg=None, border=None, padding=8):
        super().__init__()
        self.paragraphs = paragraphs
        self.bg = bg or C_LIGHT
        self.border = border or C_GREY
        self.padding = padding
        self._width = 160 * mm
        self._height = None

    def wrap(self, availWidth, availHeight):
        self._width = availWidth
        h = self.padding * 2
        for p in self.paragraphs:
            w, ph = p.wrap(availWidth - self.padding * 2, availHeight)
            h += ph + 2
        self._height = h
        return availWidth, h

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.bg)
        c.setStrokeColor(self.border)
        c.setLineWidth(0.8)
        c.roundRect(0, 0, self._width, self._height, 4, fill=1, stroke=1)
        y = self._height - self.padding
        for p in self.paragraphs:
            w, ph = p.wrap(self._width - self.padding * 2, 9999)
            y -= ph
            p.drawOn(c, self.padding, y)
            y -= 2
        c.restoreState()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def hr():
    return HRFlowable(width="100%", thickness=0.5, color=C_GREY, spaceAfter=6, spaceBefore=4)

def sp(h=6):
    return Spacer(1, h)

def para(text, style=BODY):
    return Paragraph(text, style)

def bullet(text):
    return Paragraph(f"• {text}", BULLET)

def bullets(items):
    return [bullet(i) for i in items]

def section_header(text):
    return [Paragraph(text, H1), hr(), sp(4)]

def subsection_header(text):
    return [Paragraph(text, H2)]

def table(data, col_widths, style_cmds=None, header_bg=C_ACCENT):
    base_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 8.5),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("LEADING",    (0, 0), (-1, -1), 12),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f7f9fc")]),
        ("GRID",       (0, 0), (-1, -1), 0.4, C_GREY),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ]
    if style_cmds:
        base_cmds.extend(style_cmds)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(base_cmds))
    return t

def p(text, **kw):
    """Shortcut for Paragraph with BODY style + optional overrides."""
    if kw:
        s = ParagraphStyle("tmp", parent=BODY, **kw)
    else:
        s = BODY
    return Paragraph(text, s)

# ===========================================================================
# Build document content
# ===========================================================================
def build_story():
    story = []

    # -----------------------------------------------------------------------
    # COVER PAGE
    # -----------------------------------------------------------------------
    story += [
        sp(30),
        Paragraph("BirdCLEF 2026", TITLE_STYLE),
        sp(4),
        Paragraph("Modeling Hypotheses & Strategy Report", make_style("ST2", "Normal",
            fontSize=17, textColor=C_DARK, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        sp(8),
        HRFlowable(width="60%", thickness=2, color=C_ACCENT, hAlign="CENTER"),
        sp(8),
        Paragraph("Synthesis of EDA Sections A–I", SUBTITLE_STYLE),
        Paragraph("Phase 6 — Ranked Modeling Ideas & Hypotheses", SUBTITLE_STYLE),
        sp(6),
        Paragraph(f"Generated: {datetime.date.today().isoformat()}", SUBTITLE_STYLE),
        sp(40),
    ]

    # Key stats box on cover
    kf_data = [
        ["Stat", "Value"],
        ["Target species", "234 (162 birds, 35 frogs, 28 insects, 8 mammals, 1 reptile)"],
        ["Training clips", "35,549 (206 of 234 species have clips)"],
        ["Zero-clip species", "28 — only supervised via 66 labeled soundscape windows"],
        ["Train soundscapes", "10,658 total — only 0.6% have expert window labels"],
        ["Prediction unit", "5-second windows → 234-class probability vector"],
        ["Evaluation metric", "Macro-averaged ROC-AUC (skip classes absent in test)"],
        ["Inference constraint", "CPU-only Kaggle notebook, 90-minute hard limit"],
        ["Domain gap", "Global XC/iNat clips → Pantanal passive recorders (severe)"],
        ["Competition deadline", "June 3, 2026 (~8 weeks remaining)"],
    ]
    story.append(table(kf_data,
        [45*mm, 115*mm],
        header_bg=C_ACCENT))
    story.append(sp(8))
    story.append(Paragraph(
        "This report synthesizes all EDA findings into a ranked, evidence-grounded "
        "set of modeling hypotheses, data pipeline recommendations, and a phased "
        "experiment plan. Every recommendation traces back to a specific EDA finding.",
        make_style("CoverNote", "Normal",
            fontSize=9, textColor=C_GREY, alignment=TA_CENTER, fontName="Helvetica-Oblique")))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # TABLE OF CONTENTS
    # -----------------------------------------------------------------------
    story += section_header("Contents")
    toc_items = [
        ("1", "Competition Structure — What the Model Must Actually Do"),
        ("2", "Ranked Modeling Hypotheses (Top 12)"),
        ("3", "Data Pipeline Architecture"),
        ("4", "Validation Strategy"),
        ("5", "Augmentation Plan (Evidence-Based)"),
        ("6", "Prior Data Decision"),
        ("7", "Risk Register"),
        ("8", "Phased Experiment Plan"),
        ("9", "Open Questions & Gaps"),
    ]
    toc_data = [["§", "Section"]] + toc_items
    story.append(table(toc_data, [15*mm, 145*mm], header_bg=C_DARK))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 1 — Competition Structure
    # -----------------------------------------------------------------------
    story += section_header("§1 — Competition Structure: What the Model Must Actually Do")

    story.append(para(
        "Before any modeling hypothesis can be tested, the prediction contract must be "
        "understood precisely. Misunderstanding the supervision unit vs. the scoring unit "
        "is the most common source of wasted effort in BirdCLEF-style competitions."
    ))
    story.append(sp(4))

    story += subsection_header("1.1 Prediction Contract")
    c1_data = [
        ["Axis", "Training side", "Scoring side"],
        ["Unit",      "Whole clip (3–120 s)",       "5-second window"],
        ["Label type","Clip-level primary + secondary","Per-window presence/absence"],
        ["Label count","Mostly 1; 12.3% have secondaries","3–5 species per window (EDA Section F)"],
        ["Supervision","35K clips + 1,478 soundscape windows","Hidden test soundscapes"],
        ["Geography",  "Global (lat −55 to +70)",   "Pantanal, Brazil only"],
        ["Recorder",   "Close-mic / phone / field recorder","Passive ARU deployment"],
    ]
    story.append(table(c1_data, [35*mm, 65*mm, 60*mm], header_bg=C_ACCENT))
    story.append(sp(4))
    story.append(para(
        "<b>Core tension:</b> The model is trained on close-mic clips with whole-clip labels "
        "and must generalize to passive soundscape recorders predicting at 5-second resolution. "
        "This is a three-way mismatch: recording device, label granularity, and geographic domain. "
        "Every design decision must close at least one of these gaps."
    ))
    story.append(sp(6))

    story += subsection_header("1.2 Metric Implications")
    story += bullets([
        "<b>Macro ROC-AUC</b> — each of 234 classes gets equal weight regardless of frequency. "
        "A species appearing in 2 test windows matters as much as one appearing in 2,000.",
        "Metric is <b>threshold-free</b> — calibration of absolute probability values "
        "does not matter. Relative ranking within each class is what counts.",
        "<b>Classes absent in test are skipped</b> — but you cannot know which ones. "
        "Do not deprioritize rare species.",
        "Improving common-species AUC by sacrificing rare-species AUC is net-neutral at best "
        "and often net-negative under macro averaging.",
        "This metric strongly rewards models that <i>discriminate</i> within rare classes, "
        "not just models that avoid false positives overall.",
    ])
    story.append(sp(6))

    story += subsection_header("1.3 CPU Inference Constraint (90-minute, no GPU)")
    story += bullets([
        "Hard limit on model size. EfficientNet-B4 and below are feasible; B6+ and large ViTs will time out.",
        "Mel spectrogram extraction must be batched — per-window librosa calls add ~2× overhead.",
        "ONNX export or torch.compile will likely be necessary for competitive inference speed.",
        "This constraint is known from BirdCLEF 2024 and 2025. Top solutions used EfficientNet-B0 to B2 "
        "or small ConvNext variants with ONNX export.",
        "Plan model architecture around inference budget first, then optimize within that envelope.",
    ])
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 2 — Ranked Modeling Hypotheses
    # -----------------------------------------------------------------------
    story += section_header("§2 — Ranked Modeling Hypotheses (Top 12)")

    story.append(para(
        "Each hypothesis is grounded in a specific EDA finding. Priority is assigned by "
        "<b>expected AUC impact × implementation feasibility</b>. "
        "Hypotheses are grouped into tiers: Critical (must test first), High (test in first 2 weeks), "
        "and Medium (test after baseline is established)."
    ))
    story.append(sp(6))

    # --- CRITICAL TIER ---
    story += subsection_header("Tier 1 — Critical (Expected AUC impact: High)")
    story.append(sp(4))

    hyp_data_t1 = [
        ["#", "Hypothesis", "EDA Evidence", "Expected Impact"],

        ["H1",
         "Background mixing with unlabeled Pantanal soundscapes "
         "closes the clip→soundscape domain gap more than any other single intervention.",
         "Section G5, F1: clip-to-soundscape domain gap is the primary risk. "
         "All 10,592 unlabeled soundscapes provide real Pantanal noise backgrounds.",
         "Large — directly addresses the largest identified source of train/test mismatch."],

        ["H2",
         "Waveform mixup (λ~Beta(0.4,0.4)) with soft multi-label targets teaches the "
         "model co-presence of 3–5 species, matching the test window structure.",
         "Section F2, C: expert-labeled windows show 87% have 2+ co-active species; "
         "training clips are mostly single-label. This gap hurts multi-label discrimination.",
         "High — synthetic co-presence training aligns supervision with test reality."],

        ["H3",
         "Clip windowing with energy gating reduces label noise more than any augmentation. "
         "Silent 5s windows should be treated as unlabeled, not positive.",
         "Section E3, F3: 14% mean silence ratio; some clips 90% silent. "
         "Naively assigning clip-level label to all windows creates systematic false positives.",
         "High — removes the most reliable source of corrupt training signal."],

        ["H4",
         "A validation scheme that separates by recording site (not random split) "
         "avoids inflated OOF AUC due to site-level acoustic leakage.",
         "Section G1, G2: top-5 authors contribute 40%+ of clips; site-level recorder "
         "fingerprints are learnable. Random CV splits leak site characteristics.",
         "High — a site-stratified OOF split gives a more realistic estimate of "
         "leaderboard position and prevents over-fitting to site acoustics."],
    ]
    story.append(table(hyp_data_t1,
        [10*mm, 60*mm, 60*mm, 30*mm],
        style_cmds=[
            ("BACKGROUND", (0, 1), (-1, 4), C_BG_RED),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, -1), C_RED),
        ]))
    story.append(sp(8))

    # --- HIGH TIER ---
    story += subsection_header("Tier 2 — High Priority (Expected AUC impact: Medium–High)")
    story.append(sp(4))

    hyp_data_t2 = [
        ["#", "Hypothesis", "EDA Evidence", "Expected Impact"],

        ["H5",
         "BirdCLEF 2025 Pantanal clips and soundscapes are the highest-value "
         "external data. Adding them to training directly expands labeled Pantanal coverage.",
         "Section H: BC2025 covers the same Pantanal region, same recorder type, "
         "same multi-taxon scope. Species overlap is high. This is the only prior "
         "competition data that should be used as in-domain training data.",
         "Medium–High — increases labeled soundscape coverage from 66 to potentially "
         "hundreds of files if BC2025 has more labeled windows."],

        ["H6",
         "Backbone pre-training on all available BirdCLEF clip audio (2021–2025), "
         "then fine-tuning exclusively on 2026+Pantanal data, gives better "
         "representations for rare species than training from scratch.",
         "Section H: prior competition clips share XC/iNat sources. Species-level "
         "features (call structure, harmonics) transfer even across geographies.",
         "Medium–High — especially for rare classes with < 5 clips in BC2026. "
         "Pre-trained backbone reduces overfitting on 1–5 example species."],

        ["H7",
         "Per-clip peak normalization before mel extraction removes the 600× "
         "loudness variance that currently confounds spectral features with recording level.",
         "Section E3: RMS distribution spans 600× dynamic range across recordings. "
         "Without normalization, the model partially learns recording level as a feature.",
         "Medium — straightforward preprocessing with guaranteed positive effect."],

        ["H8",
         "A multi-taxon head architecture (shared backbone, separate dense heads "
         "for Aves / Amphibia / Insecta / Mammalia) outperforms a flat 234-class head "
         "by allocating capacity to each taxon's frequency range.",
         "Section A §3: 72 of 234 species are non-birds. Frog calls are 200–4000 Hz, "
         "insects are 4000–16000 Hz, birds span 1000–8000 Hz. A flat head must learn "
         "all three spectral regimes simultaneously.",
         "Medium — especially improves Amphibia and Insecta AUC which will otherwise "
         "be dominated by bird-optimized features."],
    ]
    story.append(table(hyp_data_t2,
        [10*mm, 60*mm, 60*mm, 30*mm],
        style_cmds=[
            ("BACKGROUND", (0, 1), (-1, -1), C_BG_ORANGE),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, -1), C_ORANGE),
        ]))
    story.append(sp(8))
    story.append(PageBreak())

    # --- MEDIUM TIER ---
    story += subsection_header("Tier 3 — Medium Priority (Test after baseline is stable)")
    story.append(sp(4))

    hyp_data_t3 = [
        ["#", "Hypothesis", "EDA Evidence", "Expected Impact"],

        ["H9",
         "Semi-supervised learning on the 10,592 unlabeled soundscapes "
         "(pseudo-labeling or contrastive pre-training) provides additional "
         "Pantanal-domain signal for rare and zero-clip species.",
         "Section B: 10,592 soundscapes are unlabeled but come from the exact "
         "deployment domain. Even without labels, they carry domain-specific acoustic "
         "patterns useful for representation learning.",
         "Medium — depends heavily on baseline quality. If baseline is weak, "
         "pseudo-labels are too noisy. Start this after H1–H4 are validated."],

        ["H10",
         "Time-of-day conditioning (sin/cos encoding of hour) improves prediction "
         "accuracy because species activity probability is time-dependent "
         "(dawn chorus, nocturnal frogs, diurnal insects).",
         "Section E4: soundscape temporal patterns show dawn chorus peak. "
         "Section D: test files carry timestamps. This is a free feature derivable "
         "from the row_id without any additional data.",
         "Low–Medium — easy to add, risk-free, small but real signal for "
         "temporally-structured species."],

        ["H11",
         "The 25 insect sonotypes (parent taxon 47158) require a two-stage approach: "
         "(1) detect parent taxon presence, (2) sub-classify among sonotypes. "
         "A flat 234-class model will produce near-random scores for these 25 classes.",
         "Section B §5: 25 insect sonotypes have zero training clips. Their only "
         "supervision is the 66 labeled soundscape windows. Boundaries between sonotypes "
         "are acoustically defined, not biologically validated.",
         "Medium — recovers near-zero AUC on 25 classes to something positive. "
         "May not be worth the engineering cost early on; accept low AUC initially."],

        ["H12",
         "Post-processing the output distribution by species prevalence in "
         "Pantanal soundscapes (from the labeled windows) reduces systematic "
         "over-prediction of globally common XC species that are locally rare.",
         "Section G5: train clip geographic distribution is global; test is Pantanal. "
         "A model trained on global clips will systematically over-score species "
         "that are globally abundant but Pantanal-absent.",
         "Low–Medium — calibration-based post-processing. Under ROC-AUC it helps "
         "only indirectly (reranking across windows), but useful for threshold-based "
         "downstream use."],
    ]
    story.append(table(hyp_data_t3,
        [10*mm, 60*mm, 60*mm, 30*mm],
        style_cmds=[
            ("BACKGROUND", (0, 1), (-1, -1), C_BG_BLUE),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, -1), C_BLUE),
        ]))
    story.append(sp(8))

    story.append(InfoBox([
        para("<b>Priority summary:</b>  H1 (background mixing) → H2 (waveform mixup) → "
             "H3 (energy gating) → H4 (site-stratified CV) → H7 (normalization) → "
             "H5 (BC2025 data) → H6 (backbone pre-train) → H8 (multi-taxon head) → "
             "H9 (semi-supervised) → H10 (time-of-day) → H11 (sonotype 2-stage) → H12 (post-processing)",
             style=BODY_SMALL),
        para("Do not test H9–H12 until H1–H4 are verified on OOF CV. "
             "Stacking unvalidated interventions produces uninterpretable results.",
             style=make_style("WarnSmall", "Normal", fontSize=8.5, textColor=C_RED,
                              fontName="Helvetica-Oblique")),
    ], bg=HexColor("#fffde7"), border=C_YELLOW))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 3 — Data Pipeline
    # -----------------------------------------------------------------------
    story += section_header("§3 — Data Pipeline Architecture")

    story += subsection_header("3.1 Input Sources (Ranked by In-Domain Value)")
    src_data = [
        ["Source", "Size", "In-Domain?", "Labels?", "Recommended Use"],
        ["Train soundscapes (BC2026)", "10,658 ogg",
         "YES — Pantanal ARU", "66 labeled (1,478 windows)", "Primary training + all as backgrounds"],
        ["Train clips (BC2026)", "35,549 ogg",
         "Partial — global sources", "Clip-level", "Training with windowing + energy gating"],
        ["BC2025 clips + soundscapes", "~50K estimated",
         "High — same region", "Clip + soundscape window", "Add to training after dedup"],
        ["Prior BC clips (2021–2024)", "Large",
         "Low — different regions", "Clip-level", "Backbone pre-training only"],
        ["Additional Neotropical XC", "Variable",
         "Medium — regional match", "Clip-level", "For rare/zero-clip species only"],
        ["Non-Pantanal soundscapes", "Various",
         "NO — different domain", "Various", "Do NOT use as domain training data"],
    ]
    story.append(table(src_data, [38*mm, 20*mm, 22*mm, 24*mm, 50*mm],
        style_cmds=[
            ("BACKGROUND", (0, 1), (-1, 1), HexColor("#eafaf1")),
            ("BACKGROUND", (0, 2), (-1, 2), HexColor("#eafaf1")),
            ("BACKGROUND", (0, 3), (-1, 3), HexColor("#fef9e7")),
            ("BACKGROUND", (0, 4), (-1, 4), HexColor("#fef9e7")),
            ("BACKGROUND", (0, 5), (-1, 5), HexColor("#fef9e7")),
            ("BACKGROUND", (0, 6), (-1, 6), HexColor("#fdf0f0")),
        ]))
    story.append(sp(8))

    story += subsection_header("3.2 Clip Processing Pipeline")
    pipeline_steps = [
        ("<b>[1] Load & resample</b>", "Load OGG → resample to 32 kHz mono. "
         "All clips already 32 kHz, minimal resampling needed."),
        ("<b>[2] Peak normalization</b>", "Normalize peak to 0.9. Removes 600× loudness variance. "
         "Apply before any windowing."),
        ("<b>[3] Window extraction</b>", "Slice into non-overlapping 5s windows. "
         "Short clips (<5s): pad with zeros OR treat as single window."),
        ("<b>[4] Energy gating</b>", "Compute RMS of each window. "
         "If RMS < −45 dBFS → mark as background_only (no species label). "
         "Do NOT assign primary_label to silent windows."),
        ("<b>[5] Background mixing</b>", "Sample random 5s window from unlabeled Pantanal soundscape. "
         "Mix at SNR ~ Uniform(−5, +10) dB. Background label = all-zeros. "
         "This is the highest-ROI single step."),
        ("<b>[6] Waveform mixup</b>", "With prob 0.5: sample second clip. "
         "Mix x = λ·x1 + (1-λ)·x2, λ ~ Beta(0.4, 0.4). "
         "Blend labels: y = λ·y1 + (1-λ)·y2 (soft labels, multi-label BCE)."),
        ("<b>[7] Mel spectrogram</b>", "n_mels=128, hop_length=320 (10ms), "
         "n_fft=1024, fmin=50, fmax=14000. "
         "Result: 128×500 per 5s window. Log-scale amplitude."),
        ("<b>[8] SpecAugment (P3)</b>", "After P1–P2 are validated: "
         "freq masking 1–2 bands (≤10% mel bins), time masking 1 block (≤8% frames). "
         "Conservative to protect non-bird spectral cues."),
    ]
    pp_data = [["Step", "Action"]] + [[s, d] for s, d in pipeline_steps]
    story.append(table(pp_data, [42*mm, 118*mm],
        style_cmds=[
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]))
    story.append(PageBreak())

    story += subsection_header("3.3 Feature Extraction Recommendations")
    story += bullets([
        "<b>Sample rate:</b> 32 kHz mono — already the standard in the training data.",
        "<b>Window size:</b> 5 seconds (fixed) — matches the prediction unit exactly.",
        "<b>Mel bins:</b> 128 — sufficient for bird and frog frequency resolution; "
        "increase to 256 only if non-bird classes are the bottleneck.",
        "<b>fmin:</b> 50 Hz — captures low-frequency frog calls (Amphibia ≥ 200 Hz) "
        "and mammal calls without excessive low-frequency noise.",
        "<b>fmax:</b> 14,000 Hz — captures most insect sonotypes (peak 4–12 kHz) "
        "without wasting bins above 14 kHz where little target signal exists.",
        "<b>Pre-compute spectrograms</b> before training where possible. "
        "On-the-fly extraction from 35K OGG files causes significant I/O bottleneck.",
    ])
    story.append(sp(8))

    story += subsection_header("3.4 Label Construction")
    story += bullets([
        "<b>Clip clips → window labels:</b> assign primary_label + secondary_labels to all "
        "non-silent windows. Accept label noise; energy gating handles the worst cases.",
        "<b>Soundscape windows (labeled):</b> use exact per-window expert labels. "
        "These are the gold standard — prioritize them in the loss or oversample them.",
        "<b>Unlabeled soundscape windows (background mix source):</b> label as all-zeros "
        "when used as background. Do NOT infer labels from clip-level overlap.",
        "<b>Rare-species oversampling:</b> weight each labeled soundscape window by "
        "1/sqrt(rarest_class_freq) to ensure all 234 classes appear in every epoch.",
        "<b>Soft labels for mixup:</b> BCE loss must support continuous [0,1] targets. "
        "Confirm this in your chosen framework before training.",
    ])
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 4 — Validation Strategy
    # -----------------------------------------------------------------------
    story += section_header("§4 — Validation Strategy")

    story.append(para(
        "A realistic validation scheme is the single most important infrastructure decision. "
        "An optimistic CV scheme means every experiment looks better than it is, "
        "and real leaderboard position is a surprise. BirdCLEF competitions are notorious "
        "for site-level acoustic leakage inflating OOF scores."
    ))
    story.append(sp(6))

    story += subsection_header("4.1 Why Random K-Fold Is Wrong Here")
    story += bullets([
        "<b>Author leakage:</b> Top 5 authors contribute 40%+ of clips. A random split puts "
        "clips from the same recording session in both train and validation. "
        "The model partially memorizes recorder fingerprints.",
        "<b>Location leakage:</b> train clips have GPS coordinates. "
        "Nearby recordings sound similar (same forest, same background noise). "
        "A random split leaks location-level acoustic context.",
        "<b>Site leakage in soundscapes:</b> site S22 dominates labeled windows (64%). "
        "A random split of soundscape windows will have both train and validation from S22, "
        "making the model appear to generalize when it is just recognizing site acoustics.",
        "<b>Temporal leakage:</b> same recorder makes multiple trips to same site in same year. "
        "Near-duplicate recording clusters (EDA Section G3) will leak across random splits.",
    ])
    story.append(sp(6))

    story += subsection_header("4.2 Recommended Validation Scheme")
    val_data = [
        ["Scheme", "How to split", "When to use"],
        ["Site-stratified 5-fold (clips)",
         "Group clips by GPS cluster (K-means on lat/lon → 20 clusters). "
         "Each fold holds out complete geographic clusters.",
         "Primary CV for clip-based experiments"],
        ["Author-stratified 5-fold (clips)",
         "Group clips by author/recorder. "
         "Each fold holds out complete author groups.",
         "Cross-check: should give similar result to site-stratified"],
        ["Soundscape site holdout",
         "Reserve 3 soundscape sites (not S22) as held-out evaluation. "
         "Train on remaining 20 sites. S22 should not be the holdout — it is too dominant.",
         "Primary CV for soundscape-based experiments"],
        ["Taxonomic-stratified check",
         "Ensure each fold contains examples from all 5 taxon classes. "
         "Check that Amphibia and Insecta are not concentrated in one fold.",
         "Sanity check on all folds"],
    ]
    story.append(table(val_data, [40*mm, 80*mm, 40*mm]))
    story.append(sp(6))

    story.append(InfoBox([
        para("<b>Key rule:</b> Report OOF AUC separately per taxonomic class "
             "(Aves, Amphibia, Insecta, Mammalia, Reptilia). "
             "A high overall macro AUC driven by bird classes while frog/insect AUC is "
             "near 0.5 is a false positive. The leaderboard will expose this.",
             style=BODY_SMALL),
    ], bg=C_BG_ORANGE, border=C_ORANGE))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 5 — Augmentation Plan
    # -----------------------------------------------------------------------
    story += section_header("§5 — Augmentation Plan (Evidence-Based)")

    story.append(para(
        "Each augmentation below is tied to a specific EDA finding. "
        "Generic augmentations not supported by evidence are excluded. "
        "The list is ordered by implementation priority."
    ))
    story.append(sp(6))

    aug_data = [
        ["Priority", "Augmentation", "Problem it addresses (EDA source)", "Caveat"],
        ["P1 — Must have",
         "Energy gating\n(preprocessing)",
         "14% mean silence; some clips 90% silent.\n"
         "Clip-level labels assigned to silent windows → false positives.\n(Section E3, F3)",
         "Threshold tuning: test −45, −40, −50 dBFS on OOF."],
        ["P1 — Must have",
         "Peak normalization\n(preprocessing)",
         "600× loudness variance across recordings.\n"
         "Without normalization, model learns level as a species cue.\n(Section E3)",
         "Apply per-clip, before any windowing."],
        ["P1 — Must have",
         "Background mixing\n(soundscape overlay)",
         "Clip→soundscape domain gap is the primary train/test mismatch.\n"
         "Real Pantanal noise backgrounds are available from 10,592 unlabeled soundscapes.\n"
         "(Section G5, F1)",
         "Pre-screen soundscapes for extreme artifacts (rain, wind) first. "
         "SNR range: −5 to +10 dB."],
        ["P1 — Must have",
         "Waveform mixup\n(λ~Beta(0.4,0.4))",
         "Training clips are mostly single-label.\n"
         "Expert soundscape windows show 87% have 2+ co-active species.\n(Section F2, C)",
         "Requires soft-label BCE loss. Confirm framework supports it."],
        ["P2 — High value",
         "Rare-species oversampling",
         "28 zero-clip species; many with < 5 clips.\n"
         "Standard sampling ignores them for entire epochs.\n(Section B, C)",
         "Weight = 1/sqrt(class_freq). Watch common-class degradation."],
        ["P3 — Medium",
         "SpecAugment (conservative)",
         "Top-5 author recorder fingerprints = 40%+ of data.\n"
         "Frequency masking breaks EQ-pattern memorization.\n(Section G2)",
         "Max 10% freq bins, max 8% time frames. Test on OOF before enabling."],
        ["P3 — Medium",
         "Soft saturation\n(tanh clipping sim)",
         "~8% of clips have clipping artifacts.\n"
         "Model should be robust to saturation distortion.\n(Section E3)",
         "Apply to a random fraction (20–30%) of clips only."],
        ["P3 — Low",
         "Time-of-day feature\n(not an augmentation)",
         "Dawn chorus peak in soundscapes; species activity is time-conditional.\n"
         "Test row_id encodes datetime.\n(Section E4)",
         "Encode as sin(2π·hour/24), cos(2π·hour/24). Free feature."],
    ]
    story.append(table(aug_data, [25*mm, 30*mm, 75*mm, 30*mm],
        style_cmds=[
            ("FONTNAME", (0, 1), (0, 4), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, 4), C_RED),
            ("FONTNAME", (0, 5), (0, 6), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 5), (0, 6), C_ORANGE),
            ("FONTNAME", (0, 7), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 7), (0, -1), C_BLUE),
        ]))
    story.append(sp(8))

    story += subsection_header("5.1 Augmentations Explicitly Excluded")
    excl_data = [
        ["Excluded augmentation", "Reason"],
        ["Large pitch shift (±4+ semitones)",
         "Frog and insect calls are pitch-specific for species ID. Large shifts change species identity."],
        ["Time stretching > ±20%",
         "Call temporal rhythm (rate, duty cycle) is species-diagnostic, especially for insects."],
        ["Additive Gaussian / white noise",
         "Pantanal background is structured (water, wind, co-species). Use real soundscape backgrounds."],
        ["Spectrogram CutMix",
         "Pasting spectrogram patches creates acoustically impossible frequency chimeras. "
         "Waveform mixup is physically meaningful; spectrogram CutMix is not."],
        ["Random crop to < 2.5s",
         "Creates training examples shorter than half a prediction window. "
         "Label assignment becomes meaningless for sub-2.5s fragments."],
        ["Frequency shifting (full spectrum)",
         "Harmonic structures are key species discriminators. Shifting destroys harmonic ratios."],
    ]
    story.append(table(excl_data, [55*mm, 105*mm],
        style_cmds=[("BACKGROUND", (0, 1), (-1, -1), C_BG_RED)]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 6 — Prior Data Decision
    # -----------------------------------------------------------------------
    story += section_header("§6 — Prior Data Decision")

    story.append(para(
        "This section answers: which prior BirdCLEF competition data should be used, "
        "in what capacity, and what should be avoided. "
        "The answer is not binary — different datasets serve different purposes."
    ))
    story.append(sp(6))

    prior_data = [
        ["Dataset", "Geography", "Taxon overlap", "Domain match", "Recommended use", "Risk"],
        ["BC2025 clips + soundscapes",
         "Pantanal, Brazil",
         "High (same region species)",
         "Excellent — same ARU deployment",
         "Add directly to training after dedup by URL",
         "Possible test-set overlap with BC2026 soundscapes — check dates"],
        ["BC2024 clips",
         "Multiple regions",
         "Moderate",
         "Poor — different recording contexts",
         "Backbone pre-training only",
         "Hurts calibration if mixed directly"],
        ["BC2023 / BC2022 clips",
         "Global / India / Africa",
         "Low",
         "Very poor",
         "Backbone pre-training only if needed",
         "Geographic and species mismatch"],
        ["BC2021 clips",
         "Cornell / North America",
         "Very low",
         "Very poor",
         "Skip or use only for very broad pre-training",
         "Unlikely to help; may hurt"],
        ["BC2024 soundscapes (non-Pantanal)",
         "Other regions",
         "Low",
         "Very poor",
         "Do NOT use as domain training data",
         "Confuses domain adaptation; hurts soundscape AUC"],
    ]
    story.append(table(prior_data, [25*mm, 22*mm, 22*mm, 22*mm, 35*mm, 34*mm],
        style_cmds=[
            ("BACKGROUND", (0, 1), (-1, 1), HexColor("#eafaf1")),
            ("BACKGROUND", (0, 5), (-1, 5), HexColor("#fdf0f0")),
        ]))
    story.append(sp(8))

    story.append(InfoBox([
        para("<b>Decision:</b> Use BC2025 as in-domain training data (high priority). "
             "Use BC2021–2024 clips for backbone pre-training only. "
             "Do NOT use non-Pantanal soundscapes as soundscape training data.",
             style=BODY_SMALL),
        para("<b>Mandatory step before combining any prior data:</b> "
             "deduplicate by Xeno-Canto URL. Prior competitions draw from the same XC pool. "
             "Duplicate clips in both train and val will inflate OOF AUC.",
             style=make_style("WarnSmall2", "Normal", fontSize=8.5, textColor=C_RED,
                              fontName="Helvetica-Oblique")),
    ], bg=C_BG_GREEN, border=C_GREEN))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 7 — Risk Register
    # -----------------------------------------------------------------------
    story += section_header("§7 — Risk Register")

    risk_data = [
        ["Risk", "Severity", "Source (EDA)", "Mitigation"],
        ["Clip→soundscape domain gap",
         "CRITICAL", "G5, F1",
         "Background mixing (H1). Site-stratified CV (H4). "
         "Monitor per-site AUC during training."],
        ["Label noise from clip windowing",
         "HIGH", "E3, F3",
         "Energy gating (H3). Discard silent windows. "
         "Accept some noise for non-silent windows; model will average out."],
        ["28 zero-clip species (25 insect sonotypes + 3 frogs)",
         "HIGH", "B §5, C",
         "Oversample labeled soundscape windows (H3). "
         "Accept near-zero AUC on insect sonotypes initially. "
         "Two-stage detection for sonotypes later (H11)."],
        ["Site S22 dominance in labeled soundscapes (64% of windows)",
         "HIGH", "G4",
         "Hold out non-S22 sites for evaluation. "
         "Do not train final model exclusively on S22-derived labels."],
        ["Author/recorder fingerprint memorization",
         "MEDIUM", "G2",
         "Author-stratified CV (4.2). SpecAugment frequency masking (P3). "
         "Monitor OOF gap between author-stratified and random CV."],
        ["Seasonal mismatch (test = Feb 2025 ≈ start of dry season)",
         "MEDIUM", "D, E4",
         "Add time-of-day and month features (H10). "
         "Weight training on wet-season data slightly less if test is dry season."],
        ["Rating-zero clips degrading training (36% of clips unrated)",
         "MEDIUM", "A §4A",
         "Audit: listen to 50 random rating-0 clips. "
         "If systematic noise → train with quality-stratified sampling. "
         "If clean → keep all clips."],
        ["XC duplicate clips across BC2025–BC2026 inflating OOF",
         "MEDIUM", "H",
         "Deduplicate by XC URL before combining datasets."],
        ["CPU inference timeout on complex models",
         "HIGH", "A §12",
         "Design around ≤ EfficientNet-B2 or ConvNext-Tiny equivalent. "
         "Profile inference time early. ONNX export for speed."],
        ["Macro AUC masking poor non-bird taxon performance",
         "MEDIUM", "A §3",
         "Track OOF AUC per taxon. Target: all taxon classes ≥ 0.70. "
         "Non-bird AUC below 0.60 requires targeted intervention."],
    ]
    story.append(table(risk_data, [42*mm, 20*mm, 18*mm, 80*mm],
        style_cmds=[
            ("BACKGROUND", (0, 1), (-1, 1), C_BG_RED),
            ("BACKGROUND", (0, 2), (-1, 2), C_BG_RED),
            ("BACKGROUND", (0, 3), (-1, 3), C_BG_RED),
            ("BACKGROUND", (0, 4), (-1, 4), C_BG_RED),
            ("BACKGROUND", (0, 5), (-1, 5), C_BG_ORANGE),
            ("BACKGROUND", (0, 6), (-1, 6), C_BG_ORANGE),
            ("BACKGROUND", (0, 7), (-1, 7), C_BG_ORANGE),
            ("BACKGROUND", (0, 8), (-1, 8), C_BG_ORANGE),
            ("BACKGROUND", (0, 9), (-1, 9), C_BG_ORANGE),
            ("BACKGROUND", (0, 10), (-1, 10), C_BG_ORANGE),
            ("TEXTCOLOR", (1, 1), (1, 4), C_RED),
            ("TEXTCOLOR", (1, 5), (1, -1), C_ORANGE),
            ("FONTNAME",  (1, 1), (1, -1), "Helvetica-Bold"),
        ]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 8 — Phased Experiment Plan
    # -----------------------------------------------------------------------
    story += section_header("§8 — Phased Experiment Plan")

    story.append(para(
        "Experiments are sequenced to maximize information gain per GPU-hour. "
        "Each phase must produce a validated OOF AUC before moving to the next. "
        "Do not stack unvalidated interventions."
    ))
    story.append(sp(6))

    phase_data = [
        ["Phase", "Dates (approx)", "Goal", "Exit criterion"],
        ["Phase 0: Infrastructure",
         "Weeks 1–2\n(by Apr 26)",
         "• Clip windowing + energy gating pipeline\n"
         "• Site-stratified 5-fold splits\n"
         "• Mel spectrogram extraction (batched)\n"
         "• Baseline EfficientNet-B0 flat 234-class\n"
         "• OOF AUC tracking per taxon",
         "Reproducible baseline OOF AUC > 0.65\n"
         "Inference time < 60 min on CPU"],
        ["Phase 1: Core Augmentation",
         "Weeks 2–3\n(by May 3)",
         "• Peak normalization (H7)\n"
         "• Background mixing P1 (H1)\n"
         "• Waveform mixup P1 (H2)\n"
         "• Rare-species oversampling (H3)",
         "+0.03 OOF AUC vs Phase 0\n"
         "Non-bird taxon AUC > 0.60"],
        ["Phase 2: Data Expansion",
         "Weeks 3–4\n(by May 10)",
         "• Add BC2025 data + dedup (H5)\n"
         "• Add Neotropical XC for zero-clip species\n"
         "• SpecAugment P3 validation",
         "+0.02 OOF AUC vs Phase 1"],
        ["Phase 3: Architecture",
         "Weeks 4–5\n(by May 17)",
         "• Multi-taxon head experiment (H8)\n"
         "• Backbone pre-training on BC2021–2024 (H6)\n"
         "• Time-of-day conditioning (H10)",
         "+0.02 OOF AUC vs Phase 2\n"
         "Decide: multi-taxon head yes/no"],
        ["Phase 4: Semi-supervised",
         "Weeks 5–6\n(by May 24)",
         "• Pseudo-labeling on unlabeled soundscapes (H9)\n"
         "• Two-stage insect sonotype detection (H11)\n"
         "• Model ensemble (2–3 seeds minimum)",
         "+0.01–0.02 OOF AUC\n"
         "Insect sonotype AUC > 0.55"],
        ["Phase 5: Final polish",
         "Weeks 6–8\n(by Jun 3)",
         "• Inference optimization (ONNX / torch.compile)\n"
         "• Prevalence post-processing experiment (H12)\n"
         "• Final ensemble submission\n"
         "• Submission slot management",
         "Submission time < 85 min\n"
         "Final ensemble > Phase 4 OOF"],
    ]
    story.append(table(phase_data, [30*mm, 25*mm, 80*mm, 25*mm]))
    story.append(sp(8))

    story.append(InfoBox([
        para("<b>Checkpoint rule:</b> If a phase does not meet its exit criterion after 2 attempts, "
             "diagnose root cause before moving forward. "
             "Do not advance to the next phase on faith — every phase builds on the previous one.",
             style=BODY_SMALL),
    ], bg=C_BG_ORANGE, border=C_ORANGE))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # SECTION 9 — Open Questions & Gaps
    # -----------------------------------------------------------------------
    story += section_header("§9 — Open Questions & Gaps")

    story.append(para(
        "These are things that cannot be resolved from EDA alone and require either "
        "direct experimentation or access to the hidden test set."
    ))
    story.append(sp(6))

    oq_data = [
        ["Question", "Why it matters", "How to resolve"],
        ["Which species appear in the test soundscapes?",
         "Classes absent in test are skipped from macro AUC. "
         "Knowing test class coverage would change rare-class priority.",
         "Cannot know until final scoring. "
         "Treat all 234 classes as potentially present."],
        ["How many test soundscapes are there?",
         "Determines inference time budget per soundscape and architecture size ceiling.",
         "Check sample_submission row count. "
         "Estimate: ~50–200 soundscapes based on prior year competition sizes."],
        ["What quality are the rating-zero clips?",
         "36% of training clips are unrated. If many are low quality, "
         "they should be down-weighted or excluded.",
         "Listen to 50 random rating-0 clips manually. "
         "Run SNR estimation script on all rating-0 clips."],
        ["Are BC2025 soundscapes from the same physical sites as BC2026 soundscapes?",
         "If yes, site-level acoustic leakage between BC2025 and BC2026 test is possible.",
         "Compare site metadata if available. "
         "If site IDs overlap: exclude overlapping BC2025 soundscapes from domain training."],
        ["Are insect sonotypes actually distinguishable in practice?",
         "If sonotype boundaries are acoustically ambiguous even to experts, "
         "the 25-class insect block may be inherently unlearnable.",
         "Spectrogram visualization of sonotype examples. "
         "If visually indistinguishable → accept low AUC; not worth engineering cost."],
        ["What is the test soundscape recording date distribution?",
         "Feb 2025 is all we know from sample submission. "
         "If test spans multiple months, seasonal conditioning matters more.",
         "Infer from final submission row_ids after competition reveals more samples."],
        ["Does BC2026 evaluation skip truly zero-positive classes or uses all 234?",
         "The skip policy determines effective N for macro average. "
         "If test set happens to contain most of the 234 classes, rare classes matter more.",
         "Community discussion / competition FAQ. "
         "Assume all 234 classes are possible."],
    ]
    story.append(table(oq_data, [45*mm, 60*mm, 55*mm]))
    story.append(sp(12))

    # -----------------------------------------------------------------------
    # FOOTER SUMMARY BOX
    # -----------------------------------------------------------------------
    story.append(hr())
    story.append(sp(4))
    story.append(Paragraph("Report Summary", H2))
    story.append(para(
        "This report synthesizes EDA Sections A–I into a concrete, prioritized strategy. "
        "The three most important takeaways:"
    ))
    story += bullets([
        "<b>The domain gap (clip → soundscape) is the #1 problem.</b> "
        "Background mixing with unlabeled Pantanal soundscapes (H1) is the "
        "highest-ROI single intervention. Do this before anything else.",
        "<b>Validation scheme determines everything.</b> "
        "Site-stratified 5-fold CV is mandatory. "
        "Random splits will produce inflated OOF that doesn't match LB.",
        "<b>Non-bird taxa are the leaderboard differentiator.</b> "
        "Frogs, insects, and mammals are 30% of targets. "
        "Track their AUC separately. Most competitors will under-optimize for them.",
    ])
    story.append(sp(6))
    story.append(Paragraph(
        f"BirdCLEF 2026 — Modeling Hypotheses & Strategy Report  |  "
        f"Generated {datetime.date.today().isoformat()}  |  "
        f"reports/modeling_report.pdf",
        make_style("Footer", "Normal",
            fontSize=7.5, textColor=C_GREY, alignment=TA_CENTER,
            fontName="Helvetica-Oblique")))

    return story


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
def main():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title="BirdCLEF 2026 — Modeling Hypotheses & Strategy Report",
        author="BirdCLEF 2026 Analysis Pipeline",
        subject="Ranked modeling ideas and hypotheses synthesized from EDA Sections A–I",
    )

    story = build_story()
    doc.build(story)
    size_kb = OUT_PDF.stat().st_size / 1000
    print(f"Written: {OUT_PDF}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
