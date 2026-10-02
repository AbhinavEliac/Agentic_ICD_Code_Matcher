"""Custom CSS stylesheet and design tokens for the Streamlit clinical interface."""

CUSTOM_CSS = """
<style>
/* ------------------------------------------------------------- */
/* Modern Clinical AI Design System                             */
/* ------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

code, pre, .mono-text {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Main Container spacing */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Custom Header Banner */
.clinic-header {
    background: linear-gradient(135deg, #091322 0%, #152238 50%, #0c334d 100%);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}

.clinic-header h1, .clinic-title {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0 !important;
    display: flex;
    align-items: center;
    gap: 12px;
}

.clinic-subtitle {
    font-size: 14px !important;
    color: #e2e8f0 !important;
    margin: 0 !important;
    line-height: 1.5;
}


/* Status Pill Badges */
.pill-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.badge-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-info {
    background-color: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-danger {
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-secondary {
    background-color: rgba(148, 163, 184, 0.15);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

/* Metric Cards */
.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.kpi-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}

.kpi-value {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin: 4px 0 !important;
}

.kpi-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Primary Diagnosis Card */
.primary-card {
    background: linear-gradient(180deg, rgba(14, 165, 233, 0.12) 0%, #111e33 100%) !important;
    border: 1.5px solid rgba(56, 189, 248, 0.6) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 16px 0 !important;
    box-shadow: 0 4px 15px -3px rgba(14, 165, 233, 0.25) !important;
}

.primary-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.25);
    padding-bottom: 8px;
}

.primary-code {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
    letter-spacing: 0.5px !important;
}

.primary-desc {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin-bottom: 10px !important;
}

/* Secondary Diagnosis Card */
.secondary-card {
    background: #142032 !important;
    border: 1px solid #2d3f58 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    transition: border-color 0.15s ease;
}

.secondary-card:hover {
    border-color: #38bdf8 !important;
}

/* Evidence Highlight Box */
.evidence-box {
    background: #09121f !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 10px 14px !important;
    margin: 10px 0 !important;
    font-size: 13px !important;
    color: #e2e8f0 !important;
}

.evidence-quote {
    font-style: italic !important;
    color: #a7f3d0 !important;
}


/* Abstention Card */
.abstention-card {
    background: rgba(239, 68, 68, 0.05);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-left: 4px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* Verification Checklist */
.audit-check-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    margin-bottom: 6px;
    color: #cbd5e1;
}

.check-pass {
    color: #10b981;
    font-weight: 700;
}

.check-fail {
    color: #ef4444;
    font-weight: 700;
}

/* PDF Page Card */
.pdf-page-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
    font-size: 12px;
    line-height: 1.6;
    max-height: 250px;
    overflow-y: auto;
}

/* Multimodal Ingestion Styling */
.modality-card {
    background: linear-gradient(135deg, #132238 0%, #1a2f4c 100%);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.modality-badge-pdf {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
}

.modality-badge-image {
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
}

.modality-badge-txt {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
}

.modality-badge-manual {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
}

.section-chip {
    display: inline-block;
    background: #0f172a;
    color: #38bdf8;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
}
</style>
"""


def get_css() -> str:
    """Return the custom CSS snippet."""
    return CUSTOM_CSS
