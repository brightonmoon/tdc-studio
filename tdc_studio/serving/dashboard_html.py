"""Embedded Interactive Biomedical Web Dashboard for TDC-Studio Serving Service."""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TDC-Studio | Bio-MLOps ADMET & DTI Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script type="text/javascript" language="javascript" src="https://peter-ertl.com/jsme/JSME_2020-06-11/jsme/jsme.nocache.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    :root {
      --bg-primary: #0a0f1d;
      --bg-secondary: #111827;
      --bg-card: #1f2937;
      --bg-card-hover: #283548;
      --border-color: #374151;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --badge-safe: rgba(16, 185, 129, 0.15);
      --badge-warning: rgba(245, 158, 11, 0.15);
      --badge-danger: rgba(239, 68, 68, 0.15);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-primary);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    /* Modal styles */
    .modal-overlay {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      display: none; justify-content: center; align-items: center;
      z-index: 1000; backdrop-filter: blur(4px);
    }
    .modal-overlay.active { display: flex; }
    .modal-content {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      width: 92%; max-width: 950px;
      max-height: 90vh; overflow-y: auto;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
      padding: 1.5rem;
    }
    .modal-header {
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 0.75rem; margin-bottom: 1rem;
    }
    .modal-title { font-size: 1.1rem; font-weight: 600; color: #93c5fd; }
    .modal-close {
      background: none; border: none; color: var(--text-muted); font-size: 1.5rem;
      cursor: pointer; line-height: 1;
    }
    .modal-close:hover { color: var(--text-main); }
    .btn-action-nav {
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.4);
      color: #60a5fa;
      padding: 0.35rem 0.85rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.2s;
    }
    .btn-action-nav:hover {
      background: rgba(59, 130, 246, 0.3);
      color: #93c5fd;
    }
    .table-container {
      width: 100%;
      overflow-x: auto;
      max-height: 320px;
      border: 1px solid var(--border-color);
      border-radius: 8px;
    }
    .screening-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.75rem;
      text-align: left;
    }
    .screening-table th {
      background: var(--bg-card);
      color: #93c5fd;
      padding: 0.5rem 0.6rem;
      position: sticky;
      top: 0;
      font-weight: 600;
      border-bottom: 1px solid var(--border-color);
      white-space: nowrap;
    }
    .screening-table td {
      padding: 0.45rem 0.6rem;
      border-bottom: 1px solid rgba(55, 65, 81, 0.4);
      white-space: nowrap;
    }
    .screening-table tr:hover {
      background: rgba(255, 255, 255, 0.02);
    }
    header {
      background: linear-gradient(180deg, #111827 0%, rgba(17, 24, 39, 0.8) 100%);
      border-bottom: 1px solid var(--border-color);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }
    .logo-container { display: flex; align-items: center; gap: 0.75rem; }
    .logo-badge {
      background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
      color: white;
      font-weight: 700;
      font-size: 1.1rem;
      padding: 0.35rem 0.75rem;
      border-radius: 8px;
      letter-spacing: 0.5px;
    }
    .logo-title { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.5px; }
    .logo-subtitle { font-size: 0.75rem; color: var(--text-muted); }
    .status-bar { display: flex; align-items: center; gap: 1rem; }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      background: rgba(16, 185, 129, 0.1);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 0.3rem 0.8rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 500;
    }
    .pulse-dot {
      width: 8px; height: 8px; background-color: #10b981; border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
    }
    .nav-link {
      color: var(--text-muted); text-decoration: none; font-size: 0.85rem; font-weight: 500;
      transition: color 0.2s;
    }
    .nav-link:hover { color: var(--text-main); }
    main { max-width: 1400px; margin: 0 auto; width: 100%; padding: 2rem; flex: 1; }
    .grid-layout { display: grid; grid-template-columns: 420px 1fr; gap: 2rem; }
    @media (max-width: 1024px) { .grid-layout { grid-template-columns: 1fr; } }
    .card {
      background-color: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .card-title {
      font-size: 1.05rem; font-weight: 600; margin-bottom: 1rem;
      display: flex; align-items: center; gap: 0.5rem; color: #e5e7eb;
    }
    .input-group { margin-bottom: 1.25rem; }
    label { display: block; font-size: 0.85rem; font-weight: 500; color: var(--text-muted); margin-bottom: 0.4rem; }
    textarea, input[type="text"] {
      width: 100%;
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 0.75rem;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      transition: border-color 0.2s;
    }
    textarea:focus, input[type="text"]:focus {
      outline: none; border-color: var(--primary);
    }
    .sample-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
    .pill-btn {
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .pill-btn:hover { background-color: var(--bg-card-hover); color: var(--text-main); border-color: #6b7280; }
    .btn-primary {
      width: 100%;
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      color: white;
      border: none;
      padding: 0.85rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: transform 0.1s, opacity 0.2s;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }
    .btn-primary:hover { opacity: 0.95; transform: translateY(-1px); }
    .btn-secondary {
      width: 100%;
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 0.75rem;
      border-radius: 8px;
      font-weight: 500;
      font-size: 0.85rem;
      cursor: pointer;
      margin-top: 0.6rem;
      transition: all 0.2s;
    }
    .btn-secondary:hover { background-color: var(--bg-card-hover); }
    .score-banner {
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }
    .score-val { font-size: 2.25rem; font-weight: 700; color: #60a5fa; }
    .cluster-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
    .cluster-card {
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 1rem;
    }
    .cluster-header {
      font-size: 0.85rem; font-weight: 600; color: #93c5fd; text-transform: uppercase;
      letter-spacing: 0.5px; border-bottom: 1px solid rgba(55, 65, 81, 0.8);
      padding-bottom: 0.4rem; margin-bottom: 0.75rem;
    }
    .indicator-row {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 0.8rem; padding: 0.35rem 0; border-bottom: 1px dashed rgba(55, 65, 81, 0.4);
    }
    .indicator-name { color: var(--text-muted); }
    .indicator-decision { font-weight: 500; font-size: 0.75rem; padding: 0.15rem 0.45rem; border-radius: 4px; }
    .tag-safe { background-color: var(--badge-safe); color: #34d399; }
    .tag-warning { background-color: var(--badge-warning); color: #fbbf24; }
    .tag-danger { background-color: var(--badge-danger); color: #f87171; }
    .pbpk-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 0.75rem; }
    @media (max-width: 640px) { .pbpk-grid { grid-template-columns: 1fr; } }
    .pbpk-stat {
      background: var(--bg-card); border: 1px solid var(--border-color);
      border-radius: 8px; padding: 0.75rem; text-align: center;
    }
    .pbpk-stat-title { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }
    .pbpk-stat-val { font-size: 1.15rem; font-weight: 600; color: #38bdf8; margin: 0.2rem 0; }
    .pbpk-stat-sub { font-size: 0.7rem; color: var(--text-muted); }
    .xai-preview {
      background-color: white; border-radius: 8px; padding: 1rem;
      display: flex; justify-content: center; align-items: center; min-height: 200px;
    }
    .xai-preview svg { max-width: 100%; height: auto; }
    .opt-card {
      background: var(--bg-card); border: 1px solid var(--border-color);
      border-radius: 8px; padding: 1rem; margin-top: 0.75rem;
    }
    .opt-smiles {
      font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
      background: #111827; padding: 0.4rem; border-radius: 4px; word-break: break-all; margin: 0.4rem 0;
    }
    .opt-tags { display: flex; gap: 0.5rem; flex-wrap: wrap; font-size: 0.75rem; }
    .loading-spinner {
      display: inline-block; width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3);
      border-radius: 50%; border-top-color: white; animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    footer {
      border-top: 1px solid var(--border-color); padding: 1rem 2rem;
      text-align: center; font-size: 0.8rem; color: var(--text-muted); background: var(--bg-secondary);
    }
    .retro-badge-champion {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.75rem;
    }
    .retro-badge-alt {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-weight: 500;
      font-size: 0.75rem;
    }
    .retro-tab-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 0.4rem 0.8rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .retro-tab-btn.active {
      background: rgba(59, 130, 246, 0.2);
      border-color: #3b82f6;
      color: #93c5fd;
      font-weight: 600;
    }
    .retro-tab-btn:hover {
      background: var(--bg-card-hover);
      color: var(--text-main);
    }
    .nav-tab-link {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      transition: all 0.2s;
      cursor: pointer;
    }
    .nav-tab-link:hover, .nav-tab-link.active {
      background: rgba(59, 130, 246, 0.15);
      color: #93c5fd;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-container">
      <div class="logo-badge">TDC</div>
      <div>
        <div class="logo-title">TDC-Studio</div>
        <div class="logo-subtitle">Next-Gen ADMET, DTI & Retrosynthesis Platform</div>
      </div>
    </div>
    <div style="display: flex; gap: 0.4rem; align-items: center;">
      <a class="nav-tab-link" onclick="scrollToSection('clusters-container')">🧪 ADMET & PBPK</a>
      <a class="nav-tab-link" onclick="scrollToSection('dti-card')">🎯 DTI Affinity</a>
      <a class="nav-tab-link" onclick="scrollToSection('retro-section')">🧭 Retrosynthesis Studio</a>
    </div>
    <div class="status-bar">
      <button class="btn-action-nav" onclick="openModal('batch-modal')">
        <span>📂 Batch Screening</span>
      </button>
      <div class="status-pill">
        <span class="pulse-dot"></span>
        <span id="service-status">Engine Active (SOTA)</span>
      </div>
      <a href="/docs" target="_blank" class="nav-link">Swagger API &rarr;</a>
    </div>
  </header>

  <main>
    <div class="grid-layout">
      <!-- Input Sidebar -->
      <div>
        <div class="card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div class="card-title" style="margin-bottom: 0;">🧪 Molecule Specification</div>
            <button class="pill-btn" style="color: #60a5fa; border-color: rgba(59, 130, 246, 0.4);" onclick="openJsmeModal()">
              ✏️ 2D Editor
            </button>
          </div>
          <div class="input-group">
            <label for="smiles-input">SMILES String</label>
            <textarea id="smiles-input" rows="3" placeholder="Enter valid SMILES string...">CC(=O)Oc1ccccc1C(=O)O</textarea>
            <div class="sample-pills">
              <span class="pill-btn" onclick="setSmiles('CC(=O)Oc1ccccc1C(=O)O')">Aspirin</span>
              <span class="pill-btn" onclick="setSmiles('CC(=O)NC1=CC=C(O)C=C1')">Acetaminophen</span>
              <span class="pill-btn" onclick="setSmiles('Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C')">Imatinib</span>
              <span class="pill-btn" onclick="setSmiles('COc1cc2ncnc(c2cc1OCCCN3CCOCC3)Nc4ccc(c(c4)Cl)F')">Gefitinib</span>
              <span class="pill-btn" onclick="setSmiles('CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4')">Remdesivir</span>
            </div>
            <button class="btn-secondary" style="margin-top: 0.6rem; display: flex; align-items: center; justify-content: center; gap: 0.4rem; padding: 0.5rem;" onclick="openJsmeModal()">
              <span>✏️ Draw / Edit 2D Structure (JSME)</span>
            </button>
          </div>

          <button id="btn-analyze" class="btn-primary" onclick="runFullPipeline()">
            <span>⚡ Run 22-Task Full ADMET & PBPK</span>
          </button>
          <button class="btn-secondary" onclick="runXAI()">
            <span>🔬 XAI Heatmap & Bioisosteres</span>
          </button>
          <button class="btn-secondary" onclick="runOptimizer()">
            <span>🛠️ Closed-Loop Lead Self-Optimization</span>
          </button>
          <button class="btn-secondary" style="border-color: rgba(16, 185, 129, 0.4); color: #34d399;" onclick="runRetroPlanner()">
            <span>🧭 Plan Retrosynthesis Routes (Retro*)</span>
          </button>
        </div>

        <div class="card" id="dti-card">
          <div class="card-title">🎯 DTI Target Affinity Prediction</div>
          <div class="input-group">
            <label for="target-seq">Target Amino Acid Sequence (FASTA)</label>
            <textarea id="target-seq" rows="3" placeholder="e.g. MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD...">MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD</textarea>
          </div>
          <button class="btn-primary" onclick="runDTI()">
            <span>🔗 Predict Binding (Kd, Ki, IC50)</span>
          </button>
          <div id="dti-result-panel" style="margin-top: 1rem; display: none;"></div>
        </div>

        <div class="card" id="retro-sidebar-card">
          <div class="card-title">🧭 Retrosynthesis Search Config</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.75rem;">
            <div>
              <label for="retro-top-k">Top-K Routes</label>
              <select id="retro-top-k" style="width: 100%; background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.5rem; font-size: 0.85rem;">
                <option value="1">1 (Optimal Champion)</option>
                <option value="2">2 Routes (Top 2)</option>
                <option value="3" selected>3 Routes (Top 3)</option>
                <option value="4">4 Routes (Top 4)</option>
                <option value="5">5 Routes (Top 5)</option>
              </select>
            </div>
            <div>
              <label for="retro-max-depth">Max Depth</label>
              <select id="retro-max-depth" style="width: 100%; background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.5rem; font-size: 0.85rem;">
                <option value="3">3 Steps</option>
                <option value="5" selected>5 Steps</option>
                <option value="7">7 Steps</option>
                <option value="10">10 Steps</option>
              </select>
            </div>
          </div>
          <div class="input-group">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
              <label for="retro-diversity" style="margin-bottom: 0;">Min Route Diversity (Jaccard)</label>
              <span id="retro-diversity-val" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #60a5fa;">0.25</span>
            </div>
            <input type="range" id="retro-diversity" min="0.0" max="0.8" step="0.05" value="0.25" oninput="document.getElementById('retro-diversity-val').innerText = this.value" style="width: 100%; accent-color: var(--primary);" />
          </div>
          <div class="input-group">
            <label for="retro-banned">Banned Building Blocks (Blacklist / Supply Disruption)</label>
            <textarea id="retro-banned" rows="2" placeholder="Comma-separated SMILES to ban from commercial stock..."></textarea>
            <div style="display: flex; gap: 0.4rem; margin-top: 0.4rem;">
              <button type="button" class="pill-btn" onclick="clearBannedBuildingBlocks()">Clear Banned</button>
            </div>
          </div>
          <button class="btn-primary" style="background: linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="runRetroPlanner()">
            <span>🚀 Plan Multi-Step Pathways (Retro*)</span>
          </button>
          <button class="btn-secondary" style="margin-top: 0.5rem;" onclick="runSingleStepDisconnection()">
            <span>⚡ Single-Step Disconnection Preview</span>
          </button>
        </div>
      </div>

      <!-- Results Main Panel -->
      <div>
        <!-- Drug-likeness & Summary -->
        <div id="summary-section" class="score-banner" style="display: none;">
          <div>
            <div style="font-size: 0.85rem; color: #93c5fd; font-weight: 600;">COMPOSITE DRUG-LIKENESS SCORE</div>
            <div id="summary-meta" style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;">Canonical SMILES: -</div>
            <div style="margin-top: 0.5rem;">
              <button class="pill-btn" style="color: #34d399; border-color: rgba(16, 185, 129, 0.4);" onclick="planRouteForCurrentMolecule()">
                🧭 Plan Synthesis Route &rarr;
              </button>
            </div>
          </div>
          <div style="text-align: right;">
            <div class="score-val" id="score-val">--</div>
            <div id="score-tier" style="font-size: 0.75rem; color: #34d399; font-weight: 600;">Optimal Drug-like</div>
          </div>
        </div>

        <!-- 5 ADMET Clusters -->
        <div id="clusters-container" class="cluster-grid">
          <!-- C1: Absorption -->
          <div class="cluster-card">
            <div class="cluster-header">C1. Absorption & Permeability</div>
            <div id="c1-items"><div style="color: var(--text-muted); font-size: 0.8rem;">Click 'Run 22-Task Full ADMET' to predict.</div></div>
          </div>
          <!-- C2: Distribution -->
          <div class="cluster-card">
            <div class="cluster-header">C2. Distribution & Penetration (Tri-Hybrid)</div>
            <div id="c2-items"><div style="color: var(--text-muted); font-size: 0.8rem;">Waiting for query...</div></div>
          </div>
          <!-- C3: Metabolism -->
          <div class="cluster-card">
            <div class="cluster-header">C3. CYP450 8-Head Matrix</div>
            <div id="c3-items"><div style="color: var(--text-muted); font-size: 0.8rem;">Waiting for query...</div></div>
          </div>
          <!-- C4: Excretion -->
          <div class="cluster-card">
            <div class="cluster-header">C4. Elimination & Clearance</div>
            <div id="c4-items"><div style="color: var(--text-muted); font-size: 0.8rem;">Waiting for query...</div></div>
          </div>
          <!-- C5: Safety -->
          <div class="cluster-card" style="grid-column: 1 / -1;">
            <div class="cluster-header">C5. Cardiotoxicity & Broad Safety Shield</div>
            <div id="c5-items" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.75rem;">
              <div style="color: var(--text-muted); font-size: 0.8rem;">Waiting for query...</div>
            </div>
          </div>
        </div>

        <!-- PBPK Engine Section -->
        <div id="pbpk-section" class="card" style="margin-top: 1.5rem; display: none;">
          <div class="card-title">🧮 Physiologically Based Pharmacokinetics (PBPK Simulation)</div>
          <div class="pbpk-grid">
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Steady-State Vdss</div>
              <div class="pbpk-stat-val" id="pbpk-vdss">--</div>
              <div class="pbpk-stat-sub">L/kg (Tissue Uptake)</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Elimination Half-Life</div>
              <div class="pbpk-stat-val" id="pbpk-thalf">--</div>
              <div class="pbpk-stat-sub">hours (In Vivo t1/2)</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Fraction Unbound (fu)</div>
              <div class="pbpk-stat-val" id="pbpk-fu">--</div>
              <div class="pbpk-stat-sub">Free Drug in Plasma</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Total Body Clearance</div>
              <div class="pbpk-stat-val" id="pbpk-cltot">--</div>
              <div class="pbpk-stat-sub">L/h/kg (Systemic)</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Hepatic Extraction (EH)</div>
              <div class="pbpk-stat-val" id="pbpk-eh">--</div>
              <div class="pbpk-stat-sub">Well-Stirred Ratio</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Max Oral Bioavailability</div>
              <div class="pbpk-stat-val" id="pbpk-fmax">--</div>
              <div class="pbpk-stat-sub">First-Pass Limit</div>
            </div>
          </div>

          <!-- Virtual Population Sub-panel -->
          <div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
              <div style="font-size: 0.85rem; font-weight: 600; color: #93c5fd;">👥 Virtual Population Monte Carlo (IIV)</div>
              <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
                <select id="vpop-subgroup" style="background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.3rem 0.6rem; font-size: 0.8rem;">
                  <option value="healthy_adults">Healthy Adults (70kg standard)</option>
                  <option value="renal_mild">Renal Impairment: Mild (eGFR 60-89)</option>
                  <option value="renal_moderate">Renal Impairment: Moderate (eGFR 30-59)</option>
                  <option value="renal_severe">Renal Impairment: Severe (eGFR &lt; 30)</option>
                  <option value="hepatic_child_pugh_a">Hepatic Impairment: Child-Pugh A (Mild)</option>
                  <option value="hepatic_child_pugh_b">Hepatic Impairment: Child-Pugh B (Mod)</option>
                  <option value="hepatic_child_pugh_c">Hepatic Impairment: Child-Pugh C (Severe)</option>
                  <option value="geriatric">Geriatric (Elderly &ge; 65yo)</option>
                </select>
                <div style="display: flex; align-items: center; gap: 0.25rem;">
                  <input type="number" id="vpop-dose" value="100" min="1" max="2000" style="width: 70px; background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.3rem; font-size: 0.8rem;" title="Dose (mg)" />
                  <span style="font-size: 0.75rem; color: var(--text-muted);">mg</span>
                </div>
                <button id="btn-vpop-run" class="pill-btn" style="background: #2563eb; color: white; border: none; font-weight: 600;" onclick="runVirtualPopulation()">Simulate (N=500)</button>
              </div>
            </div>
            <div id="vpop-results-area" style="display: none; background: rgba(0, 0, 0, 0.2); border-radius: 8px; padding: 1rem;">
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.5rem; margin-bottom: 0.75rem;">
                <div class="pbpk-stat"><div class="pbpk-stat-title">C_max (Median [90% CI])</div><div class="pbpk-stat-val" id="vpop-cmax" style="font-size: 0.9rem;">--</div><div class="pbpk-stat-sub">ug/mL</div></div>
                <div class="pbpk-stat"><div class="pbpk-stat-title">T_max (Median)</div><div class="pbpk-stat-val" id="vpop-tmax" style="font-size: 0.9rem;">--</div><div class="pbpk-stat-sub">hours</div></div>
                <div class="pbpk-stat"><div class="pbpk-stat-title">AUC_inf (Median [90% CI])</div><div class="pbpk-stat-val" id="vpop-auc" style="font-size: 0.9rem;">--</div><div class="pbpk-stat-sub">ug*h/mL</div></div>
                <div class="pbpk-stat"><div class="pbpk-stat-title">t_1/2 (Median [90% CI])</div><div class="pbpk-stat-val" id="vpop-thalf" style="font-size: 0.9rem;">--</div><div class="pbpk-stat-sub">hours</div></div>
              </div>
              <div id="vpop-chart-container" style="background: #111827; border-radius: 6px; padding: 0.75rem; text-align: center;">
                <svg id="vpop-svg-chart" viewBox="0 0 600 180" style="width: 100%; height: 180px; overflow: visible;"></svg>
                <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted);">
                  <div style="display: flex; align-items: center; gap: 0.3rem;"><span style="display: inline-block; width: 12px; height: 12px; background: rgba(59, 130, 246, 0.3); border-radius: 2px;"></span> 90% Confidence Interval (5th - 95th Percentile)</div>
                  <div style="display: flex; align-items: center; gap: 0.3rem;"><span style="display: inline-block; width: 14px; height: 3px; background: #60a5fa;"></span> Median In Vivo Concentration</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Retrosynthesis & Route Feasibility Studio -->
        <div id="retro-section" class="card" style="margin-top: 1.5rem; display: none;">
          <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
            <div>
              <div class="card-title" style="margin-bottom: 0.2rem;">🧭 Retrosynthesis & Multi-Route Pathway Explorer</div>
              <div style="font-size: 0.75rem; color: var(--text-muted);">Neural A* (Retro*) Search & USPTO Single-Step Disconnection | Multi-Objective Pareto Ranking (Yield, Cost, Depth)</div>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span id="retro-badge-status" class="indicator-decision tag-safe">Ready</span>
            </div>
          </div>

          <!-- Feasibility KPI Metrics Banner -->
          <div id="retro-kpi-banner" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin-bottom: 1.25rem;">
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Search Status</div>
              <div class="pbpk-stat-val" id="retro-kpi-status" style="font-size: 1rem; color: #34d399;">--</div>
              <div class="pbpk-stat-sub" id="retro-kpi-depth-sub">Depth: --</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Est. Cumulative Yield</div>
              <div class="pbpk-stat-val" id="retro-kpi-yield" style="color: #60a5fa;">--%</div>
              <div class="pbpk-stat-sub">Overall Synthesis Efficiency</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Total Est. Cost</div>
              <div class="pbpk-stat-val" id="retro-kpi-cost" style="color: #fbbf24;">$--</div>
              <div class="pbpk-stat-sub">Per gram of product</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Stock Building Blocks</div>
              <div class="pbpk-stat-val" id="retro-kpi-bbs" style="color: #a78bfa;">--</div>
              <div class="pbpk-stat-sub">Commercial Precursors</div>
            </div>
            <div class="pbpk-stat">
              <div class="pbpk-stat-title">Candidate Routes Found</div>
              <div class="pbpk-stat-val" id="retro-kpi-routes-count" style="color: #38bdf8;">--</div>
              <div class="pbpk-stat-sub">Pareto Top-K Candidates</div>
            </div>
          </div>

          <!-- Candidate Routes Comparison Matrix Table -->
          <div id="retro-comparison-section" style="margin-bottom: 1.5rem;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #93c5fd; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
              <span>📊 Multi-Route Trade-off & Ranking Matrix</span>
              <span style="font-size: 0.75rem; color: var(--text-muted);">Click any route row or tab below to inspect detailed reaction diagram</span>
            </div>
            <div class="table-container">
              <table class="screening-table">
                <thead>
                  <tr>
                    <th>순위 (Rank)</th>
                    <th>상태 (Status)</th>
                    <th>단계 (Depth)</th>
                    <th>누적 수율 (Yield)</th>
                    <th>예상 비용 (Cost)</th>
                    <th>출발 물질 (Building Blocks)</th>
                    <th>주요 반응 (Reaction Rules)</th>
                    <th>Pareto Score</th>
                    <th>선택 (Inspect)</th>
                  </tr>
                </thead>
                <tbody id="retro-comparison-tbody">
                  <tr><td colspan="9" style="text-align: center; color: var(--text-muted);">Click 'Plan Retrosynthesis Routes' to generate candidate pathways.</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Active Route Detail Viewer -->
          <div id="retro-active-route-viewer" style="background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem;">
            <!-- Route Selector Tabs -->
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
              <div id="retro-route-tabs" style="display: flex; gap: 0.5rem; flex-wrap: wrap;"></div>
              <div style="display: flex; gap: 0.4rem;">
                <button class="pill-btn" onclick="toggleRetroViewMode('flowchart')" id="btn-view-flowchart" style="color: #60a5fa; border-color: #3b82f6;">Mermaid Flowchart</button>
                <button class="pill-btn" onclick="toggleRetroViewMode('text')" id="btn-view-text">Text Reaction Tree</button>
                <button class="pill-btn" onclick="copyMermaidCode()">📋 Copy Code</button>
              </div>
            </div>

            <!-- Diagram Display -->
            <div id="retro-flowchart-container" style="background: #111827; border: 1px solid var(--border-color); border-radius: 8px; padding: 1.5rem; overflow-x: auto; min-height: 260px; display: flex; justify-content: center; align-items: center;">
              <div id="retro-mermaid-svg" style="width: 100%; text-align: center;">
                <div style="color: var(--text-muted); font-size: 0.85rem;">Waiting for retrosynthesis planning query...</div>
              </div>
            </div>
            <pre id="retro-text-container" style="display: none; background: #111827; border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #93c5fd; overflow-x: auto; max-height: 380px;"></pre>

            <!-- Step-by-Step Reaction Sequence -->
            <div style="margin-top: 1.25rem;">
              <div style="font-size: 0.85rem; font-weight: 600; color: #93c5fd; margin-bottom: 0.5rem;">🧪 Step-by-Step Reaction Synthesis Protocol</div>
              <div class="table-container">
                <table class="screening-table">
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Reaction Rule</th>
                      <th>Reactants (Precursors)</th>
                      <th>Product</th>
                      <th>Yield (%)</th>
                      <th>Est. Cost ($)</th>
                    </tr>
                  </thead>
                  <tbody id="retro-steps-tbody">
                    <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No active route selected.</td></tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Building Blocks Bill of Materials (BOM) & Supply Chain Disruption Simulation -->
            <div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px dashed var(--border-color);">
              <div style="font-size: 0.85rem; font-weight: 600; color: #34d399; margin-bottom: 0.5rem;">📦 Commercial Starting Materials (Stock Building Blocks) & Supply Chain Testing</div>
              <div id="retro-bom-list" style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="color: var(--text-muted); font-size: 0.8rem;">Stock precursors will be listed here after search.</div>
              </div>
            </div>
          </div>

          <!-- Single Step Disconnection Result Area -->
          <div id="retro-single-step-section" style="margin-top: 1.25rem; display: none; padding-top: 1rem; border-top: 1px solid var(--border-color);">
            <div style="font-size: 0.85rem; font-weight: 600; color: #f59e0b; margin-bottom: 0.5rem;">⚡ Single-Step Retrosynthetic Disconnections (USPTO Policy Top-K)</div>
            <div class="table-container">
              <table class="screening-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Reaction Rule</th>
                    <th>Confidence</th>
                    <th>Precursor Reactants</th>
                  </tr>
                </thead>
                <tbody id="retro-single-step-tbody"></tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- XAI / Optimization Dynamic Panel -->
        <div id="extra-section" class="card" style="margin-top: 1.5rem; display: none;">
          <div class="card-title" id="extra-title">Explainability & Lead Optimization</div>
          <div id="extra-content"></div>
        </div>
      </div>
    </div>
  </main>

  <!-- JSME 2D Molecular Drawing Modal -->
  <div id="jsme-modal" class="modal-overlay">
    <div class="modal-content" style="max-width: 680px;">
      <div class="modal-header">
        <div class="modal-title">✏️ 2D Molecular Structure Editor (JSME)</div>
        <button class="modal-close" onclick="closeModal('jsme-modal')">&times;</button>
      </div>
      <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem;">
        Draw or modify chemical structures interactively. Click "Apply to Input" to transfer the SMILES to the main analysis pipeline.
      </div>
      <div id="jsme_container" style="width: 100%; height: 380px; background: white; border-radius: 8px; overflow: hidden; display: flex; justify-content: center; align-items: center;">
        <div id="jsme-loading" style="color: #4b5563; font-size: 0.9rem;">Initializing 2D Chemical Canvas...</div>
      </div>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
        <div id="jsme-smiles-preview" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #60a5fa; max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"></div>
        <div style="display: flex; gap: 0.5rem;">
          <button class="pill-btn" onclick="clearJsmeCanvas()">Clear</button>
          <button class="btn-primary" style="width: auto; padding: 0.5rem 1.25rem;" onclick="applyJsmeToInput()">Apply to Input &rarr;</button>
        </div>
      </div>
    </div>
  </div>

  <!-- High-Throughput Batch Molecular Screening Modal -->
  <div id="batch-modal" class="modal-overlay">
    <div class="modal-content" style="max-width: 1000px;">
      <div class="modal-header">
        <div class="modal-title">📂 High-Throughput Batch Molecular Screening (CSV / TSV / SDF)</div>
        <button class="modal-close" onclick="closeModal('batch-modal')">&times;</button>
      </div>
      <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
        Upload a chemical library to batch-screen <strong>Lipinski Rule of 5</strong>, <strong>22+ ADMET endpoints</strong>, and <strong>PBPK parameters</strong>. Download the enriched dataset in CSV or Excel format.
      </div>

      <!-- Upload Dropzone -->
      <div style="border: 2px dashed var(--border-color); border-radius: 8px; padding: 1.5rem; text-align: center; background: var(--bg-card); margin-bottom: 1rem; cursor: pointer;" onclick="document.getElementById('batch-file-input').click()">
        <input type="file" id="batch-file-input" accept=".csv,.tsv,.sdf,.txt" style="display: none;" onchange="handleBatchFileSelect(event)" />
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
        <div id="file-name-label" style="font-weight: 500; margin-bottom: 0.3rem;">Drag & drop your molecular library file here, or click to browse</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Supports .csv, .tsv (auto-detects 'smiles' column), or .sdf (multi-molecule)</div>
      </div>

      <div style="display: flex; gap: 0.75rem; margin-bottom: 1rem;">
        <button id="btn-batch-preview" class="btn-primary" style="flex: 1;" onclick="runBatchPreview()" disabled>
          <span>🔍 Screen & Preview Library</span>
        </button>
        <button id="btn-batch-download-csv" class="btn-secondary" style="flex: 1; margin-top: 0;" onclick="downloadBatchFile('csv')" disabled>
          <span>📥 Export Enriched CSV</span>
        </button>
        <button id="btn-batch-download-xlsx" class="btn-secondary" style="flex: 1; margin-top: 0;" onclick="downloadBatchFile('xlsx')" disabled>
          <span>📊 Export Excel (.xlsx)</span>
        </button>
      </div>

      <!-- Batch Screening Results Area -->
      <div id="batch-results-area" style="display: none;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1rem;">
          <div class="pbpk-stat"><div class="pbpk-stat-title">Screened Total</div><div class="pbpk-stat-val" id="batch-stat-total">0</div></div>
          <div class="pbpk-stat"><div class="pbpk-stat-title">Ro5 Pass Rate</div><div class="pbpk-stat-val" id="batch-stat-ro5" style="color: #34d399;">0%</div></div>
          <div class="pbpk-stat"><div class="pbpk-stat-title">hERG Safe Rate</div><div class="pbpk-stat-val" id="batch-stat-herg" style="color: #60a5fa;">0%</div></div>
          <div class="pbpk-stat"><div class="pbpk-stat-title">AMES Non-Mutagenic</div><div class="pbpk-stat-val" id="batch-stat-ames" style="color: #f59e0b;">0%</div></div>
        </div>
        <div class="table-container">
          <table class="screening-table">
            <thead>
              <tr id="batch-table-header"></tr>
            </thead>
            <tbody id="batch-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <footer>
    TDC-Studio Bio-MLOps &copy; 2026 | Therapeutics Data Commons (TDC) SOTA Benchmark Engine | FastAPI + PyTorch + RDKit
  </footer>

  <script>
    function setSmiles(s) {
      document.getElementById('smiles-input').value = s;
    }

    function getDecisionTag(decision) {
      const d = String(decision).toLowerCase();
      if (d.includes('safe') || d.includes('high perm') || d.includes('optimal') || d.includes('low card') || d.includes('low hepa') || d.includes('low clin') || d.includes('non-mut') || d.includes('non-inh')) {
        return `<span class="indicator-decision tag-safe">${decision}</span>`;
      }
      if (d.includes('mod') || d.includes('low abs') || d.includes('poor')) {
        return `<span class="indicator-decision tag-warning">${decision}</span>`;
      }
      return `<span class="indicator-decision tag-danger">${decision}</span>`;
    }

    async function runFullPipeline() {
      const smiles = document.getElementById('smiles-input').value.trim();
      if (!smiles) { alert('Please enter a SMILES string.'); return; }

      const btn = document.getElementById('btn-analyze');
      btn.innerHTML = '<span class="loading-spinner"></span> Running SOTA Inference...';
      btn.disabled = true;

      try {
        const resp = await fetch('/predict/admet_full', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: [smiles] })
        });
        if (!resp.ok) {
          const err = await resp.json();
          alert('Error: ' + (err.detail || 'Inference failed'));
          return;
        }
        const data = await resp.json();
        const profile = data.results[0];

        // 1. Summary
        document.getElementById('summary-section').style.display = 'flex';
        document.getElementById('score-val').innerText = profile.drug_likeness_score;
        document.getElementById('summary-meta').innerText = `Canonical: ${profile.canonical_smiles} | Latency: ${profile.elapsed_ms}ms`;

        // 2. C1 Absorption
        let c1Html = '';
        for (const [k, v] of Object.entries(profile.absorption)) {
          const valStr = v.value !== null ? v.value : (v.probability * 100).toFixed(1) + '%';
          c1Html += `<div class="indicator-row">
            <span class="indicator-name">${v.name} (${valStr})</span>
            ${getDecisionTag(v.decision)}
          </div>`;
        }
        document.getElementById('c1-items').innerHTML = c1Html;

        // 3. C2 Distribution
        let c2Html = '';
        for (const [k, v] of Object.entries(profile.distribution)) {
          const valStr = v.value !== null ? v.value : (v.probability * 100).toFixed(1) + '%';
          c2Html += `<div class="indicator-row">
            <span class="indicator-name">${v.name} (${valStr})</span>
            ${getDecisionTag(v.decision)}
          </div>`;
        }
        document.getElementById('c2-items').innerHTML = c2Html;

        // 4. C3 Metabolism
        let c3Html = '';
        for (const [k, v] of Object.entries(profile.metabolism)) {
          const probStr = (v.probability * 100).toFixed(1) + '%';
          c3Html += `<div class="indicator-row">
            <span class="indicator-name">${v.name} (${probStr})</span>
            ${getDecisionTag(v.decision)}
          </div>`;
        }
        document.getElementById('c3-items').innerHTML = c3Html;

        // 5. C4 Excretion
        let c4Html = '';
        for (const [k, v] of Object.entries(profile.excretion)) {
          c4Html += `<div class="indicator-row">
            <span class="indicator-name">${v.name} (${v.value} ${v.unit})</span>
            ${getDecisionTag(v.decision)}
          </div>`;
        }
        document.getElementById('c4-items').innerHTML = c4Html;

        // 6. C5 Safety
        let c5Html = '';
        for (const [k, v] of Object.entries(profile.toxicity)) {
          const valStr = v.value !== null ? v.value : (v.probability * 100).toFixed(1) + '%';
          c5Html += `<div class="indicator-row" style="background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 6px;">
            <span class="indicator-name">${v.name} (${valStr})</span>
            ${getDecisionTag(v.decision)}
          </div>`;
        }
        document.getElementById('c5-items').innerHTML = c5Html;

        // 7. PBPK Simulation
        if (profile.pbpk) {
          document.getElementById('pbpk-section').style.display = 'block';
          document.getElementById('pbpk-vdss').innerText = profile.pbpk.vdss_l_kg + ' L/kg';
          document.getElementById('pbpk-thalf').innerText = profile.pbpk.half_life_hours + ' h';
          document.getElementById('pbpk-fu').innerText = (profile.pbpk.fraction_unbound * 100).toFixed(1) + ' %';
          document.getElementById('pbpk-cltot').innerText = profile.pbpk.cl_total_l_h_kg + ' L/h/kg';
          document.getElementById('pbpk-eh').innerText = profile.pbpk.hepatic_extraction_ratio.toFixed(3);
          document.getElementById('pbpk-fmax').innerText = (profile.pbpk.max_oral_bioavailability * 100).toFixed(0) + ' %';
        }

      } catch (err) {
        alert('Fetch error: ' + err);
      } finally {
        btn.innerHTML = '<span>⚡ Run 22-Task Full ADMET & PBPK</span>';
        btn.disabled = false;
      }
    }

    async function runXAI() {
      const smiles = document.getElementById('smiles-input').value.trim();
      if (!smiles) { alert('Please enter SMILES.'); return; }
      const panel = document.getElementById('extra-section');
      const content = document.getElementById('extra-content');
      panel.style.display = 'block';
      document.getElementById('extra-title').innerText = '🔬 XAI Atom Attribution Heatmap & Bioisostere Recommendations';
      content.innerHTML = '<div style="text-align: center; padding: 1rem;"><span class="loading-spinner"></span> Computing Integrated Gradients...</div>';

      try {
        const resp = await fetch('/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: smiles })
        });
        const data = await resp.json();

        let recHtml = '';
        if (data.bioisostere_recommendations && data.bioisostere_recommendations.length > 0) {
          recHtml = '<div style="margin-top: 1rem;"><div style="font-weight: 600; margin-bottom: 0.5rem;">Recommended Bioisosteric Substitutions:</div>';
          for (const r of data.bioisostere_recommendations) {
            recHtml += `<div class="opt-card">
              <div><strong>${r.replacement}</strong> &rarr; Target Focus: <span style="color: #60a5fa;">${r.liability_focus}</span></div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;">${r.rationale}</div>
            </div>`;
          }
          recHtml += '</div>';
        }

        content.innerHTML = `
          <div class="xai-preview">
            <img src="${data.svg_data_uri}" alt="Molecular Heatmap" />
          </div>
          <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted);">
            Predicted Property Score: <strong>${data.predicted_score.toFixed(3)}</strong> | Total Atoms: <strong>${data.num_atoms}</strong>
          </div>
          ${recHtml}
        `;
      } catch (e) {
        content.innerHTML = `<div style="color: var(--danger);">XAI Error: ${e}</div>`;
      }
    }

    async function runOptimizer() {
      const smiles = document.getElementById('smiles-input').value.trim();
      if (!smiles) { alert('Please enter SMILES.'); return; }
      const panel = document.getElementById('extra-section');
      const content = document.getElementById('extra-content');
      panel.style.display = 'block';
      document.getElementById('extra-title').innerText = '🛠️ Closed-Loop Lead Self-Optimization Report';
      content.innerHTML = '<div style="text-align: center; padding: 1rem;"><span class="loading-spinner"></span> Diagnosing liabilities & generating repaired candidates...</div>';

      try {
        const resp = await fetch('/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: smiles, max_candidates: 3 })
        });
        const data = await resp.json();

        let primaryHtml = '';
        if (data.primary_liability) {
          const l = data.primary_liability;
          primaryHtml = `<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="color: #f87171; font-weight: 600;">Detected Liability: ${l.liability_name} (${l.liability_key})</div>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Current Value: ${l.current_value} (Threshold: ${l.threshold}, Severity: ${l.severity})</div>
          </div>`;
        } else {
          primaryHtml = `<div style="color: #34d399; margin-bottom: 1rem;">No severe liability detected. Structure possesses favorable ADMET balance.</div>`;
        }

        let candHtml = '';
        if (data.top_candidates && data.top_candidates.length > 0) {
          candHtml = '<div style="font-weight: 600;">Synthetically Accessible Top Candidates:</div>';
          for (const c of data.top_candidates) {
            candHtml += `<div class="opt-card">
              <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.85rem;">
                <span style="color: #34d399;">Transformation: ${c.transformation_name}</span>
                <span>Fitness: ${c.fitness_score.toFixed(2)}</span>
              </div>
              <div class="opt-smiles">${c.smiles}</div>
              <div class="opt-tags">
                <span class="indicator-decision tag-safe">Delta: ${c.liability_delta > 0 ? '+' : ''}${c.liability_delta.toFixed(2)}</span>
                <span class="indicator-decision tag-warning">SA Score: ${c.sa_score.toFixed(2)} / 10</span>
                <span class="indicator-decision ${c.scaffold_preserved ? 'tag-safe' : 'tag-warning'}">${c.scaffold_preserved ? 'Scaffold Preserved' : 'Scaffold Altered'}</span>
                <button class="pill-btn" style="color: #34d399; border-color: rgba(16, 185, 129, 0.4); padding: 0.15rem 0.45rem; font-size: 0.72rem;" onclick="planRouteForSmiles('${c.smiles}')">🧭 Plan Route</button>
              </div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.4rem;">${c.rationale}</div>
            </div>`;
          }
        }

        content.innerHTML = `
          ${primaryHtml}
          <div>Generated: <strong>${data.candidates_generated}</strong> | Passing SA Filter: <strong>${data.candidates_passing_sa_filter}</strong></div>
          ${candHtml}
        `;
      } catch (e) {
        content.innerHTML = `<div style="color: var(--danger);">Optimization Error: ${e}</div>`;
      }
    }

    async function runDTI() {
      const smiles = document.getElementById('smiles-input').value.trim();
      const seq = document.getElementById('target-seq').value.trim();
      if (!smiles || !seq) { alert('Please enter both SMILES and Target Sequence.'); return; }

      const panel = document.getElementById('dti-result-panel');
      panel.style.display = 'block';
      panel.innerHTML = '<span class="loading-spinner"></span> Computing Multi-Affinity (Kd, Ki, IC50)...';

      try {
        const resp = await fetch('/predict/dti/multi', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: [smiles], target_sequences: [seq], return_nm: true })
        });
        if (!resp.ok) {
          panel.innerHTML = `<div style="color: var(--warning); font-size: 0.8rem;">DTI Multi model is initializing or not loaded in current runtime.</div>`;
          return;
        }
        const data = await resp.json();
        const pkd = data.predictions_pkd[0];
        const pki = data.predictions_pki[0];
        const pic50 = data.predictions_pic50[0];
        const kd_nm = data.kd_nm ? data.kd_nm[0].toFixed(1) + ' nM' : 'N/A';
        const tier = data.consistency_tiers ? data.consistency_tiers[0] : 'High';

        panel.innerHTML = `
          <div class="opt-card" style="margin-top: 0;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 0.85rem;">Binding Affinity Profile</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.5rem 0; text-align: center;">
              <div style="background: #111827; padding: 0.4rem; border-radius: 4px;">
                <div style="font-size: 0.7rem; color: var(--text-muted);">pK_d</div>
                <div style="font-weight: 700; color: #60a5fa;">${pkd.toFixed(2)}</div>
              </div>
              <div style="background: #111827; padding: 0.4rem; border-radius: 4px;">
                <div style="font-size: 0.7rem; color: var(--text-muted);">pK_i</div>
                <div style="font-weight: 700; color: #34d399;">${pki.toFixed(2)}</div>
              </div>
              <div style="background: #111827; padding: 0.4rem; border-radius: 4px;">
                <div style="font-size: 0.7rem; color: var(--text-muted);">pIC_50</div>
                <div style="font-weight: 700; color: #f59e0b;">${pic50.toFixed(2)}</div>
              </div>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">
              Kd Dissociation Constant: <strong>${kd_nm}</strong> | Consistency: <strong>${tier}</strong>
            </div>
          </div>
        `;
      } catch (e) {
        panel.innerHTML = `<div style="color: var(--danger); font-size: 0.8rem;">DTI Error: ${e}</div>`;
      }
    }

    // Modal Control
    function openModal(id) {
      document.getElementById(id).classList.add('active');
      if (id === 'jsme-modal') {
        initJsmeIfNeeded();
      }
    }
    function closeModal(id) {
      document.getElementById(id).classList.remove('active');
    }
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
      }
    });

    // JSME 2D Editor
    let jsmeApplet = null;
    function initJsmeIfNeeded() {
      const container = document.getElementById('jsme_container');
      if (!jsmeApplet && typeof JSApplet !== 'undefined' && JSApplet.JSME) {
        container.innerHTML = '';
        jsmeApplet = new JSApplet.JSME("jsme_container", "100%", "380px");
        const currSmiles = document.getElementById('smiles-input').value.trim();
        if (currSmiles) {
          try { jsmeApplet.readGenericMolecularInput(currSmiles); } catch (e) {}
        }
      } else if (!jsmeApplet) {
        setTimeout(initJsmeIfNeeded, 400);
      }
    }

    function openJsmeModal() {
      openModal('jsme-modal');
    }

    function applyJsmeToInput() {
      if (jsmeApplet) {
        const smiles = jsmeApplet.smiles();
        if (smiles) {
          setSmiles(smiles);
          closeModal('jsme-modal');
        } else {
          alert('Please draw a valid molecule on canvas.');
        }
      } else {
        alert('JSME molecular editor is initializing. Please wait a moment.');
      }
    }

    function clearJsmeCanvas() {
      if (jsmeApplet) {
        jsmeApplet.reset();
        document.getElementById('jsme-smiles-preview').innerText = '';
      }
    }

    // Batch Molecular Screening
    let selectedBatchFile = null;

    function handleBatchFileSelect(event) {
      const files = event.target.files;
      if (!files || files.length === 0) return;
      selectedBatchFile = files[0];
      document.getElementById('file-name-label').innerHTML = `Selected: <strong>${selectedBatchFile.name}</strong> (${(selectedBatchFile.size / 1024).toFixed(1)} KB)`;
      document.getElementById('btn-batch-preview').disabled = false;
      document.getElementById('btn-batch-download-csv').disabled = false;
      document.getElementById('btn-batch-download-xlsx').disabled = false;
    }

    async function runBatchPreview() {
      if (!selectedBatchFile) { alert('Please select a file first.'); return; }
      const btn = document.getElementById('btn-batch-preview');
      btn.innerHTML = '<span class="loading-spinner"></span> Screening Library...';
      btn.disabled = true;

      const formData = new FormData();
      formData.append('file', selectedBatchFile);

      try {
        const resp = await fetch('/predict/batch_preview', {
          method: 'POST',
          body: formData
        });
        if (!resp.ok) {
          const err = await resp.json();
          alert('Batch screening error: ' + (err.detail || 'Failed'));
          return;
        }
        const data = await resp.json();
        const summary = data.summary;
        document.getElementById('batch-results-area').style.display = 'block';
        document.getElementById('batch-stat-total').innerText = summary.total_molecules;
        document.getElementById('batch-stat-ro5').innerText = summary.ro5_pass_rate_pct.toFixed(1) + '%';
        document.getElementById('batch-stat-herg').innerText = summary.herg_safe_rate_pct.toFixed(1) + '%';
        document.getElementById('batch-stat-ames').innerText = summary.ames_non_mutagenic_rate_pct.toFixed(1) + '%';

        const preview = data.preview_rows;
        if (preview && preview.length > 0) {
          const cols = Object.keys(preview[0]);
          const priorityCols = ['mol_id', 'smiles', 'MW', 'LogP', 'Ro5_Pass', 'C5_hERG_cardiotox_proba', 'C5_AMES_mutagenic_proba', 'PBPK_CL_total_L_h_kg', 'PBPK_t_half_h'];
          const displayCols = priorityCols.filter(c => cols.includes(c)).concat(cols.filter(c => !priorityCols.includes(c))).slice(0, 10);

          let headerHtml = '';
          for (const c of displayCols) {
            headerHtml += `<th>${c}</th>`;
          }
          document.getElementById('batch-table-header').innerHTML = headerHtml;

          let bodyHtml = '';
          for (const row of preview) {
            bodyHtml += '<tr>';
            for (const c of displayCols) {
              let val = row[c];
              if (typeof val === 'number') val = Number.isInteger(val) ? val : val.toFixed(3);
              if (val === true) val = '<span style="color:#34d399;font-weight:600;">PASS</span>';
              if (val === false) val = '<span style="color:#ef4444;font-weight:600;">FAIL</span>';
              bodyHtml += `<td>${val !== null && val !== undefined ? val : '-'}</td>`;
            }
            bodyHtml += '</tr>';
          }
          document.getElementById('batch-table-body').innerHTML = bodyHtml;
        }
      } catch (e) {
        alert('Preview failed: ' + e);
      } finally {
        btn.innerHTML = '<span>🔍 Screen & Preview Library</span>';
        btn.disabled = false;
      }
    }

    async function downloadBatchFile(format) {
      if (!selectedBatchFile) { alert('Please select a file first.'); return; }
      const btn = document.getElementById(format === 'csv' ? 'btn-batch-download-csv' : 'btn-batch-download-xlsx');
      const originalText = btn.innerHTML;
      btn.innerHTML = '<span class="loading-spinner"></span> Generating...';
      btn.disabled = true;

      const formData = new FormData();
      formData.append('file', selectedBatchFile);

      try {
        const resp = await fetch(`/predict/batch_file?export_format=${format}`, {
          method: 'POST',
          body: formData
        });
        if (!resp.ok) {
          const err = await resp.json();
          alert('Download error: ' + (err.detail || 'Failed'));
          return;
        }
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const stem = selectedBatchFile.name.replace(/\.[^/.]+$/, "");
        a.download = `${stem}_admet_screened.${format}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } catch (e) {
        alert('Export failed: ' + e);
      } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
      }
    }

    // Virtual Population Simulation
    async function runVirtualPopulation() {
      const smiles = document.getElementById('smiles-input').value.trim();
      if (!smiles) { alert('Please enter SMILES.'); return; }

      const subgroup = document.getElementById('vpop-subgroup').value;
      const dose = parseFloat(document.getElementById('vpop-dose').value) || 100.0;
      const btn = document.getElementById('btn-vpop-run');
      btn.innerHTML = '<span class="loading-spinner"></span> Simulating...';
      btn.disabled = true;

      try {
        const resp = await fetch('/pbpk/virtual_population', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            smiles: smiles,
            subgroup: subgroup,
            n_subjects: 500,
            dose_mg: dose,
            t_max_sim_hours: 48.0
          })
        });
        if (!resp.ok) {
          const err = await resp.json();
          alert('Virtual population error: ' + (err.detail || 'Failed'));
          return;
        }
        const data = await resp.json();
        const m = data.metrics;
        document.getElementById('vpop-results-area').style.display = 'block';

        document.getElementById('vpop-cmax').innerText = `${m.cmax_ug_ml.median.toFixed(2)} [${m.cmax_ug_ml.p5.toFixed(2)}-${m.cmax_ug_ml.p95.toFixed(2)}]`;
        document.getElementById('vpop-tmax').innerText = `${m.tmax_hours.median.toFixed(2)}`;
        document.getElementById('vpop-auc').innerText = `${m.auc_inf_ug_h_ml.median.toFixed(1)} [${m.auc_inf_ug_h_ml.p5.toFixed(1)}-${m.auc_inf_ug_h_ml.p95.toFixed(1)}]`;
        document.getElementById('vpop-thalf').innerText = `${m.half_life_hours.median.toFixed(2)} [${m.half_life_hours.p5.toFixed(2)}-${m.half_life_hours.p95.toFixed(2)}]`;

        renderVpopSvgChart(data.trajectory);
      } catch (e) {
        alert('Simulation failed: ' + e);
      } finally {
        btn.innerHTML = 'Simulate (N=500)';
        btn.disabled = false;
      }
    }

    function renderVpopSvgChart(traj) {
      const svg = document.getElementById('vpop-svg-chart');
      const times = traj.time_hours;
      const p5 = traj.p5_ug_ml;
      const med = traj.median_ug_ml;
      const p95 = traj.p95_ug_ml;

      const maxT = Math.max(...times) || 48.0;
      const maxC = Math.max(...p95) * 1.15 || 1.0;

      const W = 600, H = 180, padL = 45, padR = 20, padT = 15, padB = 25;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const x = (t) => padL + (t / maxT) * plotW;
      const y = (c) => padT + plotH - (c / maxC) * plotH;

      let polyPts = [];
      for (let i = 0; i < times.length; i++) {
        polyPts.push(`${x(times[i]).toFixed(1)},${y(p95[i]).toFixed(1)}`);
      }
      for (let i = times.length - 1; i >= 0; i--) {
        polyPts.push(`${x(times[i]).toFixed(1)},${y(p5[i]).toFixed(1)}`);
      }

      let medPath = `M ${x(times[0]).toFixed(1)} ${y(med[0]).toFixed(1)}`;
      for (let i = 1; i < times.length; i++) {
        medPath += ` L ${x(times[i]).toFixed(1)} ${y(med[i]).toFixed(1)}`;
      }

      let gridLines = '';
      for (let i = 0; i <= 4; i++) {
        const yVal = padT + (plotH / 4) * i;
        const concVal = (maxC * (1 - i / 4)).toFixed(1);
        gridLines += `<line x1="${padL}" y1="${yVal}" x2="${W - padR}" y2="${yVal}" stroke="#374151" stroke-dasharray="3 3" opacity="0.6"/>`;
        gridLines += `<text x="${padL - 6}" y="${yVal + 3}" fill="#9ca3af" font-size="9" text-anchor="end">${concVal}</text>`;
      }
      for (let i = 0; i <= 4; i++) {
        const xVal = padL + (plotW / 4) * i;
        const timeVal = ((maxT / 4) * i).toFixed(0);
        gridLines += `<line x1="${xVal}" y1="${padT}" x2="${xVal}" y2="${H - padB}" stroke="#374151" stroke-dasharray="3 3" opacity="0.4"/>`;
        gridLines += `<text x="${xVal}" y="${H - 8}" fill="#9ca3af" font-size="9" text-anchor="middle">${timeVal}h</text>`;
      }

      svg.innerHTML = `
        ${gridLines}
        <polygon points="${polyPts.join(' ')}" fill="rgba(59, 130, 246, 0.25)" stroke="none" />
        <path d="${medPath}" fill="none" stroke="#60a5fa" stroke-width="2.5" />
      `;
    }

    // =========================================================================
    // Retrosynthesis & Multi-Route Pathway Planner
    // =========================================================================
    if (window.mermaid) {
      try {
        mermaid.initialize({
          startOnLoad: false,
          theme: 'dark',
          themeVariables: {
            darkMode: true,
            background: '#111827',
            primaryColor: '#1e3a8a',
            primaryTextColor: '#f3f4f6',
            primaryBorderColor: '#3b82f6',
            lineColor: '#60a5fa',
            secondaryColor: '#1f2937',
            tertiaryColor: '#111827',
            fontFamily: 'Inter, sans-serif'
          }
        });
      } catch (e) {
        console.warn('Mermaid initialization warning:', e);
      }
    }

    let currentRetroData = null;
    let activeRetroRank = 1;
    let currentRetroViewMode = 'flowchart';

    function scrollToSection(id) {
      const el = document.getElementById(id);
      if (el) {
        if (el.style.display === 'none') {
          el.style.display = 'block';
        }
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }

    function clearBannedBuildingBlocks() {
      document.getElementById('retro-banned').value = '';
      if (currentRetroData) {
        runRetroPlanner();
      }
    }

    function banBuildingBlock(smiles) {
      const input = document.getElementById('retro-banned');
      const current = input.value.trim();
      const set = new Set(current ? current.split(',').map(s => s.trim()).filter(Boolean) : []);
      set.add(smiles);
      input.value = Array.from(set).join(', ');
      runRetroPlanner();
    }

    function planRouteForSmiles(smiles) {
      setSmiles(smiles);
      runRetroPlanner();
    }

    function planRouteForCurrentMolecule() {
      runRetroPlanner();
    }

    async function runRetroPlanner() {
      const smiles = document.getElementById('smiles-input').value.trim();
      if (!smiles) { alert('Please enter a target SMILES string.'); return; }

      const section = document.getElementById('retro-section');
      section.style.display = 'block';
      scrollToSection('retro-section');

      const topK = parseInt(document.getElementById('retro-top-k').value, 10) || 3;
      const maxDepth = parseInt(document.getElementById('retro-max-depth').value, 10) || 5;
      const minDiversity = parseFloat(document.getElementById('retro-diversity').value) || 0.25;
      const bannedRaw = document.getElementById('retro-banned').value.trim();
      const bannedSmiles = bannedRaw ? bannedRaw.split(',').map(s => s.trim()).filter(Boolean) : null;

      const statusBadge = document.getElementById('retro-badge-status');
      statusBadge.className = 'indicator-decision tag-warning';
      statusBadge.innerText = 'Searching (Retro*)...';

      const mermaidSvg = document.getElementById('retro-mermaid-svg');
      mermaidSvg.innerHTML = '<div style="text-align: center; padding: 2rem;"><span class="loading-spinner"></span><div style="margin-top: 0.5rem; font-size: 0.85rem; color: #93c5fd;">Running Neural A* (Retro*) Search & Pareto Optimization...</div></div>';

      try {
        const resp = await fetch('/retrosynthesis/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            smiles: smiles,
            top_k: topK,
            min_diversity: minDiversity,
            banned_smiles: bannedSmiles,
            max_depth: maxDepth,
            timeout_sec: 10.0,
            render_mermaid: true
          })
        });

        if (!resp.ok) {
          const err = await resp.json();
          alert('Retrosynthesis error: ' + (err.detail || 'Search failed'));
          statusBadge.className = 'indicator-decision tag-danger';
          statusBadge.innerText = 'Search Failed';
          mermaidSvg.innerHTML = `<div style="color: #ef4444; font-size: 0.85rem;">Search failed: ${err.detail || 'Unknown error'}</div>`;
          return;
        }

        const data = await resp.json();
        currentRetroData = data;
        renderRetroResults(data);

      } catch (err) {
        alert('Retrosynthesis request error: ' + err);
        statusBadge.className = 'indicator-decision tag-danger';
        statusBadge.innerText = 'Error';
        mermaidSvg.innerHTML = `<div style="color: #ef4444; font-size: 0.85rem;">Request error: ${err}</div>`;
      }
    }

    function renderRetroResults(data) {
      const statusBadge = document.getElementById('retro-badge-status');
      const isSolved = data.solved;
      statusBadge.className = isSolved ? 'indicator-decision tag-safe' : 'indicator-decision tag-warning';
      statusBadge.innerText = isSolved ? 'Pathway Solved' : 'Partial Route';

      // KPI Banner
      document.getElementById('retro-kpi-status').innerText = isSolved ? '✅ Stock Available' : '⚠️ Incomplete';
      document.getElementById('retro-kpi-status').style.color = isSolved ? '#34d399' : '#f59e0b';
      document.getElementById('retro-kpi-depth-sub').innerText = `Max Depth: ${data.total_depth} steps`;
      document.getElementById('retro-kpi-yield').innerText = `${data.cumulative_yield.toFixed(1)}%`;
      document.getElementById('retro-kpi-cost').innerText = `$${data.total_cost.toFixed(2)}`;
      document.getElementById('retro-kpi-bbs').innerText = `${data.starting_materials.length} BBs`;
      document.getElementById('retro-kpi-routes-count').innerText = `${data.routes.length} / ${document.getElementById('retro-top-k').value}`;

      // Comparison Table
      renderComparisonTable(data);

      // Tabs for Routes
      renderRouteTabs(data);

      // Default to Rank 1
      activeRetroRank = 1;
      renderActiveRoute(1);
    }

    function renderComparisonTable(data) {
      const tbody = document.getElementById('retro-comparison-tbody');
      const routes = data.routes || [];
      if (routes.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No candidate pathways discovered.</td></tr>';
        return;
      }

      const rankLabels = {
        1: '🥇 1위 (최적)',
        2: '🥈 2위 (대안 A)',
        3: '🥉 3위 (대안 B)',
        4: '4위 (대안 C)',
        5: '5위 (대안 D)'
      };

      let rowsHtml = '';
      routes.forEach(r => {
        const badgeClass = r.rank === 1 ? 'retro-badge-champion' : 'retro-badge-alt';
        const statusTag = r.solved ? '<span class="indicator-decision tag-safe">✅ Solved</span>' : '<span class="indicator-decision tag-warning">⚠️ Partial</span>';
        const rules = Array.from(new Set(r.steps.map(s => s.rule_name))).join(', ') || 'Direct Stock';
        const bbCount = `${r.starting_materials.length}종 (${r.starting_materials.slice(0, 2).join(', ')}${r.starting_materials.length > 2 ? '...' : ''})`;
        const isSelected = r.rank === activeRetroRank;

        rowsHtml += `
          <tr style="cursor: pointer; ${isSelected ? 'background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6;' : ''}" onclick="selectRetroRoute(${r.rank})">
            <td><span class="${badgeClass}">${rankLabels[r.rank] || (r.rank + '위')}</span></td>
            <td>${statusTag}</td>
            <td>${r.total_depth}단계</td>
            <td style="color: #60a5fa; font-weight: 600;">${r.cumulative_yield.toFixed(1)}%</td>
            <td style="color: #fbbf24; font-weight: 600;">$${r.total_cost.toFixed(2)}</td>
            <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">${bbCount}</td>
            <td style="font-size: 0.75rem; color: var(--text-muted);">${rules}</td>
            <td style="font-weight: 600; color: #a78bfa;">${r.rank_score.toFixed(3)}</td>
            <td>
              <button class="pill-btn" style="color: #93c5fd; border-color: rgba(59, 130, 246, 0.4);" onclick="event.stopPropagation(); selectRetroRoute(${r.rank})">
                Inspect
              </button>
            </td>
          </tr>
        `;
      });
      tbody.innerHTML = rowsHtml;
    }

    function renderRouteTabs(data) {
      const container = document.getElementById('retro-route-tabs');
      const routes = data.routes || [];
      const rankLabels = {
        1: '🥇 1위 (최적 경로)',
        2: '🥈 2위 (대안 A)',
        3: '🥉 3위 (대안 B)',
        4: '4위 (대안 C)',
        5: '5위 (대안 D)'
      };

      let tabsHtml = '';
      routes.forEach(r => {
        const activeClass = r.rank === activeRetroRank ? 'active' : '';
        tabsHtml += `
          <button class="retro-tab-btn ${activeClass}" id="retro-tab-${r.rank}" onclick="selectRetroRoute(${r.rank})">
            ${rankLabels[r.rank] || (r.rank + '위')} [수율: ${r.cumulative_yield.toFixed(1)}%, $${r.total_cost.toFixed(1)}]
          </button>
        `;
      });
      container.innerHTML = tabsHtml;
    }

    function selectRetroRoute(rank) {
      activeRetroRank = rank;
      if (!currentRetroData || !currentRetroData.routes) return;

      // Update tabs
      currentRetroData.routes.forEach(r => {
        const tab = document.getElementById(`retro-tab-${r.rank}`);
        if (tab) {
          if (r.rank === rank) {
            tab.classList.add('active');
          } else {
            tab.classList.remove('active');
          }
        }
      });

      // Re-highlight comparison table
      renderComparisonTable(currentRetroData);

      // Render detail for active route
      renderActiveRoute(rank);
    }

    async function renderActiveRoute(rank) {
      if (!currentRetroData || !currentRetroData.routes) return;
      const route = currentRetroData.routes.find(r => r.rank === rank) || currentRetroData.routes[0];
      if (!route) return;

      // 1. Text Tree content
      const textTree = generateTextTree(route);
      document.getElementById('retro-text-container').innerText = textTree;

      // 2. Mermaid flowchart
      const mermaidSvg = document.getElementById('retro-mermaid-svg');
      if (route.mermaid_diagram) {
        mermaidSvg.innerHTML = '<span class="loading-spinner"></span> Rendering SVG flowchart...';
        await renderMermaidSvg(route.mermaid_diagram, mermaidSvg);
      } else {
        mermaidSvg.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No diagram generated for this route.</div>';
      }

      // 3. Step-by-Step Table
      const stepTbody = document.getElementById('retro-steps-tbody');
      if (!route.steps || route.steps.length === 0) {
        stepTbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Direct stock molecule (0 synthetic steps required).</td></tr>';
      } else {
        let stepsHtml = '';
        route.steps.forEach(s => {
          const reactantsStr = s.reactants.map(r => `<span style="font-family: 'JetBrains Mono', monospace; background: #111827; padding: 0.2rem 0.4rem; border-radius: 4px; display: inline-block; margin: 0.1rem 0;">${r}</span>`).join(' + ');
          const prodStr = `<span style="font-family: 'JetBrains Mono', monospace; background: #111827; padding: 0.2rem 0.4rem; border-radius: 4px; display: inline-block;">${s.product}</span>`;
          stepsHtml += `
            <tr>
              <td><span style="font-weight: 600; color: #93c5fd;">#${s.step_number}</span></td>
              <td><span style="font-weight: 600; color: #fbbf24;">${s.rule_name}</span></td>
              <td>${reactantsStr}</td>
              <td>${prodStr}</td>
              <td style="color: #60a5fa; font-weight: 600;">${s.yield_pct.toFixed(1)}%</td>
              <td style="color: #34d399; font-weight: 600;">$${s.cost.toFixed(2)}</td>
            </tr>
          `;
        });
        stepTbody.innerHTML = stepsHtml;
      }

      // 4. Commercial Stock Building Blocks BOM
      const bomContainer = document.getElementById('retro-bom-list');
      if (!route.starting_materials || route.starting_materials.length === 0) {
        bomContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem;">No starting materials cataloged.</div>';
      } else {
        let bomHtml = '';
        route.starting_materials.forEach(bb => {
          bomHtml += `
            <div class="opt-card" style="display: flex; justify-content: space-between; align-items: center; margin-top: 0; padding: 0.6rem 0.85rem;">
              <div style="display: flex; align-items: center; gap: 0.6rem; overflow: hidden;">
                <span style="color: #34d399; font-size: 1rem;">📦</span>
                <div style="overflow: hidden;">
                  <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #f3f4f6; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${bb}</div>
                  <div style="font-size: 0.7rem; color: var(--text-muted);">Commercial Building Block &bull; In Stock</div>
                </div>
              </div>
              <div style="display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0;">
                <button class="pill-btn" style="color: #f87171; border-color: rgba(239, 68, 68, 0.4); font-size: 0.7rem;" onclick="banBuildingBlock('${bb}')" title="Simulate reagent shortage/supply chain disruption">
                  🚫 Ban Reagent
                </button>
                <button class="pill-btn" style="font-size: 0.7rem;" onclick="navigator.clipboard.writeText('${bb}')" title="Copy SMILES">
                  📋
                </button>
              </div>
            </div>
          `;
        });
        bomContainer.innerHTML = bomHtml;
      }
    }

    function generateTextTree(route) {
      let lines = [
        `🎯 Target Molecule: ${route.target_smiles}`,
        `📊 Status: ${route.solved ? 'Solved (All Stock Available)' : 'Incomplete'} | Depth: ${route.total_depth} steps | Cumulative Yield: ${route.cumulative_yield.toFixed(1)}% | Est. Cost: $${route.total_cost.toFixed(2)}`,
        `======================================================================`
      ];
      const stockSet = new Set(route.starting_materials);
      for (let i = route.steps.length - 1; i >= 0; i--) {
        const s = route.steps[i];
        lines.push(`Step ${s.step_number}: [${s.rule_name}] -> Yield: ${s.yield_pct.toFixed(1)}%, Cost: $${s.cost.toFixed(2)}`);
        lines.push(`  └── Product: ${s.product}`);
        lines.push(`  └── Precursors:`);
        s.reactants.forEach(r => {
          const tag = stockSet.has(r) ? ' (📦 Stock Reagent)' : ' (🔄 Intermediate)';
          lines.push(`      ├── ${r}${tag}`);
        });
        lines.push(`--------------------------------------------------`);
      }
      return lines.join('\n');
    }

    async function renderMermaidSvg(code, containerEl) {
      let cleaned = code.replace(/```mermaid/gi, '').replace(/```/g, '').trim();
      if (!window.mermaid) {
        containerEl.innerHTML = `<pre style="font-family: monospace; font-size: 0.8rem; color: #93c5fd; text-align: left; overflow-x: auto;">${cleaned}</pre>`;
        return;
      }
      try {
        const renderId = 'retro_svg_' + Math.floor(Math.random() * 1000000);
        const { svg } = await mermaid.render(renderId, cleaned);
        containerEl.innerHTML = svg;
      } catch (err) {
        console.error('Mermaid render error:', err);
        containerEl.innerHTML = `<div style="color: #ef4444; font-size: 0.8rem; margin-bottom: 0.5rem;">Diagram rendering notice: ${err.message || err}</div><pre style="font-family: monospace; font-size: 0.75rem; color: #9ca3af; text-align: left; overflow-x: auto;">${cleaned}</pre>`;
      }
    }

    function toggleRetroViewMode(mode) {
      currentRetroViewMode = mode;
      const flowContainer = document.getElementById('retro-flowchart-container');
      const textContainer = document.getElementById('retro-text-container');
      const btnFlow = document.getElementById('btn-view-flowchart');
      const btnText = document.getElementById('btn-view-text');

      if (mode === 'flowchart') {
        flowContainer.style.display = 'flex';
        textContainer.style.display = 'none';
        btnFlow.style.color = '#60a5fa';
        btnFlow.style.borderColor = '#3b82f6';
        btnText.style.color = 'var(--text-muted)';
        btnText.style.borderColor = 'var(--border-color)';
      } else {
        flowContainer.style.display = 'none';
        textContainer.style.display = 'block';
        btnText.style.color = '#60a5fa';
        btnText.style.borderColor = '#3b82f6';
        btnFlow.style.color = 'var(--text-muted)';
        btnFlow.style.borderColor = 'var(--border-color)';
      }
    }

    function copyMermaidCode() {
      if (!currentRetroData || !currentRetroData.routes) {
        alert('No route data available.');
        return;
      }
      const route = currentRetroData.routes.find(r => r.rank === activeRetroRank) || currentRetroData.routes[0];
      if (route && route.mermaid_diagram) {
        const clean = route.mermaid_diagram.replace(/```mermaid/gi, '').replace(/```/g, '').trim();
        navigator.clipboard.writeText(clean);
        alert('Copied Mermaid syntax to clipboard!');
      } else {
        alert('No diagram syntax found for this route.');
      }
    }

    async function runSingleStepDisconnection() {
      const smiles = document.getElementById('smiles-input').value.trim();
      if (!smiles) { alert('Please enter target SMILES.'); return; }

      const section = document.getElementById('retro-section');
      section.style.display = 'block';
      scrollToSection('retro-section');

      const subSection = document.getElementById('retro-single-step-section');
      const tbody = document.getElementById('retro-single-step-tbody');
      subSection.style.display = 'block';
      tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;"><span class="loading-spinner"></span> Predicting candidate disconnections...</td></tr>';

      try {
        const resp = await fetch('/retrosynthesis/single-step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: smiles, top_k: 5 })
        });
        if (!resp.ok) {
          const err = await resp.json();
          tbody.innerHTML = `<tr><td colspan="4" style="color: #ef4444;">Error: ${err.detail || 'Disconnection failed'}</td></tr>`;
          return;
        }
        const data = await resp.json();
        if (!data.predictions || data.predictions.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No candidate disconnections found for this structure.</td></tr>';
          return;
        }

        let rowsHtml = '';
        data.predictions.forEach(p => {
          const reactStr = p.reactants.map(r => `<span style="font-family: 'JetBrains Mono', monospace; background: #111827; padding: 0.2rem 0.4rem; border-radius: 4px; display: inline-block; margin: 0.1rem 0;">${r}</span>`).join(' + ');
          rowsHtml += `
            <tr>
              <td><span style="font-weight: 600; color: #93c5fd;">Rank ${p.rank}</span></td>
              <td><span style="font-weight: 600; color: #fbbf24;">${p.rule_name}</span></td>
              <td style="color: #34d399; font-weight: 600;">${(p.confidence * 100).toFixed(1)}%</td>
              <td>${reactStr}</td>
            </tr>
          `;
        });
        tbody.innerHTML = rowsHtml;
      } catch (err) {
        tbody.innerHTML = `<tr><td colspan="4" style="color: #ef4444;">Request error: ${err}</td></tr>`;
      }
    }
  </script>
</body>
</html>
"""
