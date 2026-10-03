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
  </style>
</head>
<body>
  <header>
    <div class="logo-container">
      <div class="logo-badge">TDC</div>
      <div>
        <div class="logo-title">TDC-Studio</div>
        <div class="logo-subtitle">Next-Gen ADMET & DTI Foundation Platform</div>
      </div>
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
        </div>

        <div class="card">
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
      </div>

      <!-- Results Main Panel -->
      <div>
        <!-- Drug-likeness & Summary -->
        <div id="summary-section" class="score-banner" style="display: none;">
          <div>
            <div style="font-size: 0.85rem; color: #93c5fd; font-weight: 600;">COMPOSITE DRUG-LIKENESS SCORE</div>
            <div id="summary-meta" style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;">Canonical SMILES: -</div>
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
  </script>
</body>
</html>
"""
