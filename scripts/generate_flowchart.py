"""Generate crisp SVG and high-resolution PNG workflow flowchart diagrams."""

from pathlib import Path

import pymupdf

OUTPUT_DIR = Path("docs/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SVG_PATH = OUTPUT_DIR / "workflow_flowchart.svg"
PNG_PATH = OUTPUT_DIR / "workflow_flowchart.png"

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1150" width="1600" height="1150">
  <defs>
    <!-- Gradients -->
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="50%" stop-color="#818cf8"/>
      <stop offset="100%" stop-color="#c084fc"/>
    </linearGradient>

    <linearGradient id="cardGradCyan" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0f2b48"/>
      <stop offset="100%" stop-color="#0a1d30"/>
    </linearGradient>

    <linearGradient id="cardGradPurple" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#261647"/>
      <stop offset="100%" stop-color="#180e2e"/>
    </linearGradient>

    <linearGradient id="cardGradAmber" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38260b"/>
      <stop offset="100%" stop-color="#211606"/>
    </linearGradient>

    <linearGradient id="cardGradEmerald" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#093022"/>
      <stop offset="100%" stop-color="#051f16"/>
    </linearGradient>

    <!-- Filters -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feOffset dx="0" dy="4" result="offset"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.4"/>
      </feComponentTransfer>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <!-- Arrow Markers -->
    <marker id="arrowCyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8"/>
    </marker>
    <marker id="arrowAmber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#fbbf24"/>
    </marker>
    <marker id="arrowEmerald" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#34d399"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#a78bfa"/>
    </marker>
    <marker id="arrowRose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#fb7185"/>
    </marker>
  </defs>

  <!-- Solid Canvas Background -->
  <rect width="1600" height="1150" fill="#080c14"/>

  <!-- Subtle Blueprint Grid -->
  <g opacity="0.06">
    <path d="M 0,0 L 1600,0 M 0,50 L 1600,50 M 0,100 L 1600,100 M 0,150 L 1600,150 M 0,200 L 1600,200 M 0,250 L 1600,250 M 0,300 L 1600,300 M 0,350 L 1600,350 M 0,400 L 1600,400 M 0,450 L 1600,450 M 0,500 L 1600,500 M 0,550 L 1600,550 M 0,600 L 1600,600 M 0,650 L 1600,650 M 0,700 L 1600,700 M 0,750 L 1600,750 M 0,800 L 1600,800 M 0,850 L 1600,850 M 0,900 L 1600,900 M 0,950 L 1600,950 M 0,1000 L 1600,1000 M 0,1050 L 1600,1050 M 0,1100 L 1600,1100 M 0,1150 L 1600,1150" stroke="#38bdf8" stroke-width="1"/>
    <path d="M 0,0 L 0,1150 M 50,0 L 50,1150 M 100,0 L 100,1150 M 150,0 L 150,1150 M 200,0 L 200,1150 M 250,0 L 250,1150 M 300,0 L 300,1150 M 350,0 L 350,1150 M 400,0 L 400,1150 M 450,0 L 450,1150 M 500,0 L 500,1150 M 550,0 L 550,1150 M 600,0 L 600,1150 M 650,0 L 650,1150 M 700,0 L 700,1150 M 750,0 L 750,1150 M 800,0 L 800,1150 M 850,0 L 850,1150 M 900,0 L 900,1150 M 950,0 L 950,1150 M 1000,0 L 1000,1150 M 1050,0 L 1050,1150 M 1100,0 L 1100,1150 M 1150,0 L 1150,1150 M 1200,0 L 1200,1150 M 1250,0 L 1250,1150 M 1300,0 L 1300,1150 M 1350,0 L 1350,1150 M 1400,0 L 1400,1150 M 1450,0 L 1450,1150 M 1500,0 L 1500,1150 M 1550,0 L 1550,1150 M 1600,0 L 1600,1150" stroke="#38bdf8" stroke-width="1"/>
  </g>

  <!-- ==================== HEADER SECTION ==================== -->
  <g transform="translate(60, 42)">
    <text x="0" y="28" font-family="'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="28" font-weight="800" fill="#f8fafc" letter-spacing="0.5">AGENTIC ICD-10-CM AUTONOMOUS CODING PIPELINE</text>
    <text x="0" y="54" font-family="'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="14" font-weight="400" fill="#94a3b8">Deterministic-First LangGraph Workflow &bull; Offline GGUF Inference &bull; Hybrid BM25+FAISS Retrieval &bull; Strict UHDDS Compliance</text>

    <!-- Top Badges -->
    <g transform="translate(980, 8)">
      <rect x="0" y="0" width="140" height="28" rx="14" fill="#0369a1" fill-opacity="0.25" stroke="#38bdf8" stroke-width="1.2"/>
      <text x="70" y="18" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#38bdf8" text-anchor="middle">100% AIR-GAPPED</text>

      <rect x="152" y="0" width="165" height="28" rx="14" fill="#065f46" fill-opacity="0.25" stroke="#34d399" stroke-width="1.2"/>
      <text x="234" y="18" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#34d399" text-anchor="middle">ZERO HALLUCINATION</text>

      <rect x="328" y="0" width="170" height="28" rx="14" fill="#581c87" fill-opacity="0.25" stroke="#c084fc" stroke-width="1.2"/>
      <text x="413" y="18" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#c084fc" text-anchor="middle">UHDDS &amp; HIPAA VERIFIED</text>
    </g>
  </g>

  <!-- ==================== PHASE 1: INGESTION & DOCUMENT VERIFICATION ==================== -->
  <g transform="translate(50, 115)">
    <!-- Container -->
    <rect x="0" y="0" width="345" height="955" rx="16" fill="#0d1424" stroke="#1e293b" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="345" height="42" rx="16" fill="#131e33"/>
    <circle cx="24" cy="21" r="7" fill="#38bdf8"/>
    <text x="40" y="26" font-family="'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#38bdf8" letter-spacing="0.5">PHASE 1: INGESTION &amp; DOC CHECK</text>

    <!-- Node 0: Document Input Card -->
    <g transform="translate(20, 58)">
      <rect x="0" y="0" width="305" height="150" rx="12" fill="url(#cardGradCyan)" stroke="#0284c7" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="70" height="20" rx="4" fill="#0284c7" fill-opacity="0.3"/>
      <text x="49" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#38bdf8" text-anchor="middle">INPUT</text>
      <text x="92" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">Clinical Documentation</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Inpatient Discharge Summaries</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Formats: Plain Text (.txt) or PDF (.pdf)</text>
      <text x="16" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Batch Concurrency: &ge; 10 documents</text>
      <rect x="14" y="112" width="277" height="24" rx="4" fill="#0369a1" fill-opacity="0.25"/>
      <text x="152" y="128" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="600" fill="#38bdf8" text-anchor="middle">BoundedDocumentGate (asyncio.Semaphore)</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="208" x2="172" y2="242" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowCyan)"/>

    <!-- Node 1: validate_document -->
    <g transform="translate(20, 244)">
      <rect x="0" y="0" width="305" height="150" rx="12" fill="url(#cardGradCyan)" stroke="#0284c7" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#0284c7" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#38bdf8" text-anchor="middle">NODE 1</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">validate_document</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Verifies payload integrity &amp; existence</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Non-zero byte &amp; non-empty string checks</text>
      <text x="16" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Rejects unreadable or corrupted files</text>
      <rect x="14" y="112" width="277" height="24" rx="4" fill="#dc2626" fill-opacity="0.2" stroke="#f43f5e" stroke-width="0.8"/>
      <text x="152" y="128" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="600" fill="#fca5a5" text-anchor="middle">Fail &rarr; Short-circuits to Abstention</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="394" x2="172" y2="428" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowCyan)"/>

    <!-- Node 2: extract_text -->
    <g transform="translate(20, 430)">
      <rect x="0" y="0" width="305" height="195" rx="12" fill="url(#cardGradCyan)" stroke="#0284c7" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#0284c7" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#38bdf8" text-anchor="middle">NODE 2</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">extract_text</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; PyMuPDF thread-pool offloading</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#cbd5e1">&bull; Scanned image detection &amp; OCR alerts</text>
      <text x="16" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="600" fill="#38bdf8">SectionDetector Segmentation:</text>
      <text x="24" y="118" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#93c5fd">&bull; Discharge / Principal Diagnoses</text>
      <text x="24" y="136" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#93c5fd">&bull; History of Present Illness (HPI)</text>
      <text x="24" y="154" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#93c5fd">&bull; Hospital Course &amp; Assessment</text>
      <text x="16" y="180" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#67e8f9">&bull; Whitespace normalization &amp; offsets</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="625" x2="172" y2="665" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowCyan)"/>

    <!-- Document State Box -->
    <g transform="translate(20, 668)">
      <rect x="0" y="0" width="305" height="120" rx="12" fill="#0b172b" stroke="#38bdf8" stroke-dasharray="4 4" stroke-width="1.2"/>
      <text x="16" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="700" fill="#38bdf8">Parsed Document State</text>
      <text x="16" y="52" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#94a3b8">&bull; Clean text spans &amp; character offsets</text>
      <text x="16" y="72" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#94a3b8">&bull; Section boundaries mapped</text>
      <text x="16" y="92" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#94a3b8">&bull; Thread-safe state ready for LLM agent</text>
    </g>
  </g>

  <!-- Connector Phase 1 to Phase 2 -->
  <path d="M 395, 728 L 415, 728 C 425, 728 425, 175 435, 175 L 440, 175" fill="none" stroke="#818cf8" stroke-width="2.5" marker-end="url(#arrowPurple)"/>

  <!-- ==================== PHASE 2: CLINICAL UNDERSTANDING & UHDDS ==================== -->
  <g transform="translate(440, 115)">
    <!-- Container -->
    <rect x="0" y="0" width="345" height="955" rx="16" fill="#0d1424" stroke="#1e293b" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="345" height="42" rx="16" fill="#1b1433"/>
    <circle cx="24" cy="21" r="7" fill="#a78bfa"/>
    <text x="40" y="26" font-family="'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#a78bfa" letter-spacing="0.5">PHASE 2: CLINICAL NLP &amp; UHDDS</text>

    <!-- Node 3: extract_diagnoses Card -->
    <g transform="translate(20, 58)">
      <rect x="0" y="0" width="305" height="205" rx="12" fill="url(#cardGradPurple)" stroke="#7c3aed" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#7c3aed" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#c084fc" text-anchor="middle">NODE 3</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">extract_diagnoses</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#e2e8f0">&bull; Local LLM: Offline GGUF inference</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#e2e8f0">&bull; Mandatory verbatim text quote</text>
      <text x="16" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#e2e8f0">&bull; Exact start &amp; end character offsets</text>

      <!-- Ban Warning -->
      <rect x="14" y="112" width="277" height="46" rx="6" fill="#dc2626" fill-opacity="0.22" stroke="#ef4444" stroke-width="1"/>
      <text x="152" y="130" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="700" fill="#fca5a5" text-anchor="middle">&#10008; STRICT BAN: NO ICD CODES HERE</text>
      <text x="152" y="147" font-family="'Segoe UI', Roboto, sans-serif" font-size="9.5" fill="#fecaca" text-anchor="middle">LLM is strictly forbidden from guessing codes</text>

      <text x="16" y="184" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#c084fc">&bull; Subsumption removes broad duplicate terms</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="263" x2="172" y2="295" stroke="#a78bfa" stroke-width="2" marker-end="url(#arrowPurple)"/>

    <!-- Node 4: analyze_context Card -->
    <g transform="translate(20, 298)">
      <rect x="0" y="0" width="305" height="200" rx="12" fill="url(#cardGradPurple)" stroke="#7c3aed" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#7c3aed" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#c084fc" text-anchor="middle">NODE 4</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">analyze_context</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="700" fill="#e9d5ff">Clinical Context Dimensions:</text>

      <text x="16" y="80" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#38bdf8">Negation:</tspan> Affirmative vs Negated ("denies")</text>
      <text x="16" y="100" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#38bdf8">Temporality:</tspan> Current vs Historical PMH</text>
      <text x="16" y="120" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#38bdf8">Certainty:</tspan> Confirmed vs Suspected vs Ruled Out</text>
      <text x="16" y="140" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#38bdf8">Acuity:</tspan> Acute, Chronic, Acute-on-Chronic</text>

      <rect x="14" y="156" width="277" height="28" rx="4" fill="#581c87" fill-opacity="0.25"/>
      <text x="152" y="174" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="600" fill="#e9d5ff" text-anchor="middle">Verifies documented inpatient management</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="498" x2="172" y2="532" stroke="#a78bfa" stroke-width="2" marker-end="url(#arrowPurple)"/>

    <!-- Node 5: classify_diagnoses Card -->
    <g transform="translate(20, 535)">
      <rect x="0" y="0" width="305" height="215" rx="12" fill="url(#cardGradPurple)" stroke="#7c3aed" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#7c3aed" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#c084fc" text-anchor="middle">NODE 5</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">classify_diagnoses</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="700" fill="#e9d5ff">UHDDS Role Disambiguation:</text>
      <text x="16" y="80" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#34d399">PRIMARY:</tspan> Exactly ONE chief reason for admission</text>
      <text x="16" y="102" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#38bdf8">SECONDARY:</tspan> Active co-existing comorbidities</text>
      <text x="16" y="124" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#e2e8f0">&bull; <tspan font-weight="700" fill="#f87171">EXCLUDED:</tspan> Inactive PMH or Ruled-Out</text>

      <rect x="14" y="142" width="277" height="56" rx="6" fill="#1e1b4b" stroke="#818cf8" stroke-width="1"/>
      <text x="24" y="162" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#a5b4fc">INVARIANT: Single Primary Diagnosis</text>
      <text x="24" y="180" font-family="'Segoe UI', Roboto, sans-serif" font-size="9.5" fill="#c7d2fe">Multiple primary ties &rarr; demote or flag for query</text>
    </g>
  </g>

  <!-- Connector Phase 2 to Phase 3 -->
  <path d="M 785, 642 L 805, 642 C 815, 642 815, 175 825, 175 L 830, 175" fill="none" stroke="#fbbf24" stroke-width="2.5" marker-end="url(#arrowAmber)"/>

  <!-- ==================== PHASE 3: RETRIEVAL & CONSTRAINED RANKING ==================== -->
  <g transform="translate(830, 115)">
    <!-- Container -->
    <rect x="0" y="0" width="345" height="955" rx="16" fill="#0d1424" stroke="#1e293b" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="345" height="42" rx="16" fill="#241908"/>
    <circle cx="24" cy="21" r="7" fill="#fbbf24"/>
    <text x="40" y="26" font-family="'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#fbbf24" letter-spacing="0.5">PHASE 3: RETRIEVAL &amp; RANKING</text>

    <!-- Catalog Box -->
    <g transform="translate(20, 58)">
      <rect x="0" y="0" width="305" height="140" rx="12" fill="#20180a" stroke="#d97706" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="70" height="20" rx="4" fill="#d97706" fill-opacity="0.3"/>
      <text x="49" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#fbbf24" text-anchor="middle">CATALOG</text>
      <text x="92" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">Local ICD-10 Dataset</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#fde68a">&bull; Official CMS / CDC Order Files</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#fde68a">&bull; Terminal billable leaf codes (HIPAA)</text>
      <text x="16" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#fde68a">&bull; Authoritative Excludes1 &amp; Inclusions</text>
      <rect x="14" y="108" width="277" height="20" rx="4" fill="#78350f" fill-opacity="0.3"/>
      <text x="152" y="122" font-family="'Segoe UI', Roboto, sans-serif" font-size="9.5" font-weight="600" fill="#f59e0b" text-anchor="middle">Zero cloud API &bull; 100% Local Dataset</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="198" x2="172" y2="230" stroke="#fbbf24" stroke-width="2" marker-end="url(#arrowAmber)"/>

    <!-- Node 6: retrieve_candidates Card -->
    <g transform="translate(20, 233)">
      <rect x="0" y="0" width="305" height="260" rx="12" fill="url(#cardGradAmber)" stroke="#d97706" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#d97706" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#fbbf24" text-anchor="middle">NODE 6</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">retrieve_candidates</text>

      <!-- Dual retrieval cards -->
      <g transform="translate(14, 48)">
        <rect x="0" y="0" width="132" height="62" rx="6" fill="#1b1204" stroke="#fbbf24" stroke-width="0.8"/>
        <text x="10" y="20" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#fbbf24">BM25 Lexical</text>
        <text x="10" y="36" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#fef3c7">Keyword matching</text>
        <text x="10" y="50" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#fef3c7">Clinical synonyms</text>
      </g>
      <g transform="translate(158, 48)">
        <rect x="0" y="0" width="132" height="62" rx="6" fill="#1b1204" stroke="#fbbf24" stroke-width="0.8"/>
        <text x="10" y="20" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#fbbf24">FAISS Dense</text>
        <text x="10" y="36" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#fef3c7">BGE-Small vectors</text>
        <text x="10" y="50" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#fef3c7">Cosine similarity</text>
      </g>

      <text x="16" y="136" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="600" fill="#fef3c7">Hybrid Fusion Formula:</text>
      <text x="16" y="156" font-family="monospace" font-size="10" fill="#fde68a">Score = 0.6*Dense + 0.4*BM25</text>
      <text x="16" y="180" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#fde047">&bull; Reciprocal Rank Fusion (RRF)</text>
      <text x="16" y="200" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#fde047">&bull; Top-K candidate pool extraction (K=5..10)</text>

      <rect x="14" y="214" width="277" height="30" rx="4" fill="#451a03" stroke="#f59e0b" stroke-width="0.8"/>
      <text x="152" y="233" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="600" fill="#fed7aa" text-anchor="middle">Catalog Boundary: Out-of-catalog codes purged</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="493" x2="172" y2="526" stroke="#fbbf24" stroke-width="2" marker-end="url(#arrowAmber)"/>

    <!-- Node 7: rank_candidates Card -->
    <g transform="translate(20, 530)">
      <rect x="0" y="0" width="305" height="225" rx="12" fill="url(#cardGradAmber)" stroke="#d97706" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#d97706" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#fbbf24" text-anchor="middle">NODE 7</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">rank_candidates</text>
      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#fef3c7">&bull; Evaluates candidate pool vs clinical quotes</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#fef3c7">&bull; Aligns clinical specificity (e.g. STEMI)</text>
      <text x="16" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#fef3c7">&bull; Computes confidence score &amp; justification</text>

      <rect x="14" y="114" width="277" height="46" rx="6" fill="#78350f" fill-opacity="0.5" stroke="#f59e0b" stroke-width="1"/>
      <text x="152" y="132" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="700" fill="#fed7aa" text-anchor="middle">&#10004; CONSTRAINED POOL INVARIANT</text>
      <text x="152" y="149" font-family="'Segoe UI', Roboto, sans-serif" font-size="9.5" fill="#ffedd5" text-anchor="middle">Ranker strictly confined to retrieved candidates</text>

      <text x="16" y="182" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#fde68a">&bull; Rejects out-of-pool choices</text>
      <text x="16" y="202" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#fde68a">&bull; Low confidence (&lt;0.65) &rarr; flags for abstention</text>
    </g>
  </g>

  <!-- Connector Phase 3 to Phase 4 -->
  <path d="M 1175, 642 L 1195, 642 C 1205, 642 1205, 175 1215, 175 L 1220, 175" fill="none" stroke="#34d399" stroke-width="2.5" marker-end="url(#arrowEmerald)"/>

  <!-- ==================== PHASE 4: DETERMINISTIC GUARDRAILS & STORAGE ==================== -->
  <g transform="translate(1220, 115)">
    <!-- Container -->
    <rect x="0" y="0" width="345" height="955" rx="16" fill="#0d1424" stroke="#1e293b" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="345" height="42" rx="16" fill="#0b241b"/>
    <circle cx="24" cy="21" r="7" fill="#34d399"/>
    <text x="40" y="26" font-family="'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#34d399" letter-spacing="0.5">PHASE 4: GUARDRAILS &amp; STORAGE</text>

    <!-- Node 8: validate_codes Card -->
    <g transform="translate(20, 58)">
      <rect x="0" y="0" width="305" height="230" rx="12" fill="url(#cardGradEmerald)" stroke="#059669" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#059669" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#34d399" text-anchor="middle">NODE 8</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">validate_codes</text>
      <rect x="200" y="14" width="90" height="20" rx="4" fill="#047857"/>
      <text x="245" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" font-weight="700" fill="#a7f3d0" text-anchor="middle">100% NON-LLM</text>

      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#6ee7b7">Deterministic Verification Engine:</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#d1fae5">&bull; <tspan font-weight="600" fill="#34d399">Catalog Check:</tspan> Code exists in local ICD dataset</text>
      <text x="16" y="96" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#d1fae5">&bull; <tspan font-weight="600" fill="#34d399">Leaf Specificity:</tspan> Must be terminal billable code</text>
      <text x="16" y="114" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#d1fae5">&bull; <tspan font-weight="600" fill="#34d399">Excludes1 Rules:</tspan> Detects mutually exclusive pairs</text>
      <text x="16" y="132" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#d1fae5">&bull; <tspan font-weight="600" fill="#34d399">Duplicate Consolidation:</tspan> Merges duplicate codes</text>
      <text x="16" y="150" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#d1fae5">&bull; <tspan font-weight="600" fill="#34d399">Anti-Hallucination Specificity:</tspan> Demotes over-specific</text>

      <rect x="14" y="172" width="277" height="42" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="0.8"/>
      <text x="152" y="188" font-family="'Segoe UI', Roboto, sans-serif" font-size="9.5" font-weight="700" fill="#a7f3d0" text-anchor="middle">AUDIT TRAIL GENERATION</text>
      <text x="152" y="204" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#d1fae5" text-anchor="middle">Every check logs rule code, evidence quote, timestamp</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="288" x2="172" y2="320" stroke="#34d399" stroke-width="2" marker-end="url(#arrowEmerald)"/>

    <!-- Node 9: evaluate_confidence & Abstentions -->
    <g transform="translate(20, 324)">
      <rect x="0" y="0" width="305" height="195" rx="12" fill="url(#cardGradEmerald)" stroke="#059669" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="60" height="20" rx="4" fill="#059669" fill-opacity="0.3"/>
      <text x="44" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#34d399" text-anchor="middle">NODE 9</text>
      <text x="82" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">evaluate_confidence</text>

      <text x="16" y="58" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" fill="#d1fae5">&bull; Composite Confidence Scoring (Extraction + Retrieval + Ranking)</text>
      <text x="16" y="78" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="600" fill="#6ee7b7">Explicit Abstention Engine:</text>
      <text x="24" y="98" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" fill="#fca5a5">&bull; INSUFFICIENT_CLINICAL_EVIDENCE</text>
      <text x="24" y="114" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" fill="#fca5a5">&bull; MULTIPLE_AMBIGUOUS_PRIMARY</text>
      <text x="24" y="130" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" fill="#fca5a5">&bull; SPECIFICITY_REQUIRED / UNSPECIFIED</text>
      <text x="24" y="146" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" fill="#fca5a5">&bull; EXCLUDES1_CONFLICT_DETECTED</text>

      <rect x="14" y="160" width="277" height="22" rx="4" fill="#14532d" stroke="#22c55e" stroke-width="0.8"/>
      <text x="152" y="175" font-family="'Segoe UI', Roboto, sans-serif" font-size="9.5" font-weight="600" fill="#bbf7d0" text-anchor="middle">Responsible AI: Refuses to guess under ambiguity</text>
    </g>

    <!-- Down Arrow -->
    <line x1="172" y1="519" x2="172" y2="550" stroke="#34d399" stroke-width="2" marker-end="url(#arrowEmerald)"/>

    <!-- Node 10: finalize_output & Storage -->
    <g transform="translate(20, 554)">
      <rect x="0" y="0" width="305" height="240" rx="12" fill="url(#cardGradEmerald)" stroke="#059669" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="14" y="14" width="65" height="20" rx="4" fill="#059669" fill-opacity="0.3"/>
      <text x="46" y="28" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700" fill="#34d399" text-anchor="middle">NODE 10</text>
      <text x="88" y="29" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="700" fill="#f8fafc">finalize_output</text>

      <text x="16" y="56" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="#6ee7b7">Delivery &amp; Persistence Targets:</text>

      <!-- Target 1: Pydantic Payload -->
      <g transform="translate(14, 66)">
        <rect x="0" y="0" width="277" height="38" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="0.8"/>
        <text x="12" y="17" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="700" fill="#f8fafc">Pydantic CodingResult Schema</text>
        <text x="12" y="31" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#a7f3d0">Validated JSON payload &bull; Internal prompts omitted</text>
      </g>

      <!-- Target 2: SQLite DB -->
      <g transform="translate(14, 112)">
        <rect x="0" y="0" width="277" height="38" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="0.8"/>
        <text x="12" y="17" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="700" fill="#f8fafc">SQLite Encounters Vault (WAL Mode)</text>
        <text x="12" y="31" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#a7f3d0">Indexed by encounter, ICD code, billable status</text>
      </g>

      <!-- Target 3: UI & API -->
      <g transform="translate(14, 158)">
        <rect x="0" y="0" width="277" height="48" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="0.8"/>
        <text x="12" y="17" font-family="'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="700" fill="#f8fafc">Streamlit UI &amp; FastAPI REST Service</text>
        <text x="12" y="31" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#a7f3d0">&bull; Interactive clinical cards &amp; analytics</text>
        <text x="12" y="43" font-family="'Segoe UI', Roboto, sans-serif" font-size="9" fill="#a7f3d0">&bull; Endpoints: /code/text, /code/pdf, /batch-pdf</text>
      </g>

      <!-- Terminal Indicator -->
      <rect x="70" y="214" width="165" height="20" rx="10" fill="#10b981"/>
      <text x="152" y="228" font-family="'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="800" fill="#022c22" text-anchor="middle">&#10004; PIPELINE COMPLETE</text>
    </g>
  </g>

  <!-- ==================== FOOTER ==================== -->
  <g transform="translate(60, 1100)">
    <line x1="0" y1="0" x2="1480" y2="0" stroke="#1e293b" stroke-width="1"/>
    <text x="0" y="24" font-family="'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="500" fill="#64748b">Author: Abhinav Gupta &bull; Email: abhinavgupta15.ag@gmail.com &bull; GitHub: https://github.com/AbhinavEliac/Agentic_ICD_Code_Matcher</text>
    <text x="1480" y="24" font-family="'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="500" fill="#64748b" text-anchor="end">Engineered with Python 3.13 &bull; LangGraph &bull; GPT4All &bull; FAISS &bull; PyMuPDF &bull; Streamlit</text>
  </g>
</svg>
"""

with open(SVG_PATH, "w", encoding="utf-8") as f:
    f.write(svg_content.strip())
print(f"SVG generated successfully: {SVG_PATH}")

doc = pymupdf.open(stream=svg_content.encode("utf-8"), filetype="svg")
pix = doc[0].get_pixmap(dpi=150)
pix.save(PNG_PATH)
print(f"High-resolution PNG generated successfully: {PNG_PATH} ({pix.width}x{pix.height})")
