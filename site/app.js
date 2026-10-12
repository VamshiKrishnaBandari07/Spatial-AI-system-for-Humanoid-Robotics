'use strict';
const urls = new Map();
for (const role of ['before', 'after']) {
  document.getElementById(role).addEventListener('change', event => {
    const video = document.getElementById(role + 'Preview');
    if (urls.has(role)) URL.revokeObjectURL(urls.get(role));
    video.removeAttribute('src');
    const file = event.target.files[0];
    if (file) {const url = URL.createObjectURL(file); urls.set(role, url); video.src = url;}
    video.load();
  });
  document.getElementById(role + 'Preview').addEventListener('error', () => {
    document.getElementById('videoStatus').textContent = 'This browser could not decode the selected preview. The analysis workspace may support a different codec.';
  });
}
document.getElementById('report').addEventListener('change', async event => {
  const status = document.getElementById('reportStatus');
  const results = document.getElementById('results'); results.replaceChildren();
  const file = event.target.files[0]; if (!file) {status.textContent = 'No report loaded.'; return;}
  try {
    if (file.size > 32 * 1024 * 1024) throw new Error('Report exceeds the 32 MB browser limit.');
    const data = JSON.parse(await file.text());
    const visual = data.status === 'visual_proposals';
    const rows = visual ? data.changes : data.events;
    if (!Array.isArray(rows)) throw new Error('Choose changes.json or visual_changes.json.');
    const allowed = new Set(['ADDED','REMOVED','MOVED','UNCHANGED','UNKNOWN','INCONCLUSIVE','ROTATED','DEFORMED']);
    if (rows.some(row => !row || !allowed.has(row.change))) throw new Error('Report contains an unsupported change record.');
    status.textContent = `${rows.length} records loaded. ${visual ? 'Visual model proposals — not verified 3D changes.' : 'Pipeline decisions — inspect supporting evidence before relying on them.'}`;
    for (const row of rows.slice(0, 1000)) {
      const card = document.createElement('article'); card.className = 'result';
      const heading = document.createElement('h3'); heading.textContent = `${row.semantic_label || row.label || 'Unnamed region'} — ${row.change}`;
      const detail = document.createElement('p'); detail.textContent = row.explanation || row.observation || 'No explanation supplied.';
      card.append(heading, detail);
      if (visual) {const uncertainty = document.createElement('p'); uncertainty.textContent = `Uncertainty: ${row.uncertainty || 'Not supplied'}`; card.append(uncertainty);}
      results.append(card);
    }
    if (rows.length > 1000) status.textContent += ' Showing the first 1,000 records.';
  } catch (error) {status.textContent = `Unable to load report: ${error.message}`;}
});
