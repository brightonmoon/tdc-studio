"""Embedded Interactive Biomedical Web Dashboard for TDC-Studio Serving Service."""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TDC-Studio | Bio-MLOps ADMET & DTI Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
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
          <div class="card-title">🧪 Molecule Specification</div>
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
        </div>

        <!-- XAI / Optimization Dynamic Panel -->
        <div id="extra-section" class="card" style="margin-top: 1.5rem; display: none;">
          <div class="card-title" id="extra-title">Explainability & Lead Optimization</div>
          <div id="extra-content"></div>
        </div>
      </div>
    </div>
  </main>

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
  </script>
</body>
</html>
"""
