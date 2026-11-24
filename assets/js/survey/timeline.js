const threshold = document.getElementById('timeline-relevance');
const frame = document.getElementById('histogram-frame');
const histogram = document.getElementById('histogram');
const months = JSON.parse(document.getElementById('timeline-data').textContent);
const countLabel = document.getElementById('timeline-count');
const emptyLabel = document.getElementById('timeline-empty');
function drawHistogram() {
  const bins = new Map();
  const startScore = threshold.value ? Number(threshold.value) + 1 : 0;
  let total = 0;
  let earliestYear = Infinity, latestYear = -Infinity, peak = 0;
  for (const [date, counts] of Object.entries(months)) {
    const [year, month] = date.split('-').map(Number);
    earliestYear = Math.min(earliestYear, year);
    latestYear = Math.max(latestYear, year);
    peak = Math.max(peak, counts.reduce((sum, count) => sum + count, 0));
    const hits = counts.slice(startScore).reduce((sum, count) => sum + count, 0);
    if (!hits) continue;
    bins.set(12 * year + month - 1, hits);
    total += hits;
  }
  countLabel.textContent = total + (total === 1 ? ' paper shown' : ' papers shown');
  countLabel.hidden = !total;
  emptyLabel.hidden = !!total;
  frame.hidden = !bins.size;
  if (!bins.size) return;
  const first = 12 * earliestYear, last = 12 * latestYear + 11;
  const width = Math.max(320, frame.clientWidth);
  const height = frame.clientHeight;
  const left = 42, right = width - 16, top = 24, bottom = height - 46;
  const count = last - first + 1;
  const binWidth = (right - left) / count;
  const target = Math.max(20, peak * 1.25);
  const step = Math.ceil(target / 25) * 5;
  const ceiling = Math.ceil(target / step) * step;
  const y = hits => bottom - (bottom - top) * hits / ceiling;
  let chart = '<text x="4" y="14">Paper count</text>';
  chart += `<line x1="${left}" x2="${left}" y1="${top}" y2="${bottom}"/>`;
  for (let hits = 0; hits <= ceiling; hits += step) {
    chart += `<line class="grid" x1="${left}" x2="${right}" y1="${y(hits)}" y2="${y(hits)}"/>`;
    chart += `<text x="${left - 6}" y="${y(hits) + 4}" text-anchor="end">${hits}</text>`;
  }
  const labelEvery = Math.max(1, Math.ceil(count / Math.max(2, Math.floor((right - left) / 70))));
  for (let offset = 0; offset < count; offset++) {
    const index = first + offset;
    const month = Math.floor(index / 12) + '-' + String(index % 12 + 1).padStart(2, '0');
    const hits = bins.get(index) || 0;
    const x = left + binWidth * offset;
    if (hits) {
      const href = 'papers.html?month=' + month + (threshold.value ? '&amp;relevance=above-' + threshold.value : '');
      chart += `<a href="${href}" aria-label="${month}, ${hits} papers"><rect x="${x + .15 * binWidth}"`;
      chart += ` y="${y(hits)}" width="${.7 * binWidth}" height="${bottom - y(hits)}">`;
      chart += `<title>${month} · ${hits} papers</title></rect></a>`;
    }
    if (offset % labelEvery === 0) {
      chart += `<text x="${x + binWidth / 2}" y="${bottom + 19}" text-anchor="middle">${month}</text>`;
    }
  }
  chart += `<line x1="${left}" x2="${right}" y1="${bottom}" y2="${bottom}"/>`;
  chart += `<text x="${(left + right) / 2}" y="${height - 8}" text-anchor="middle">Submission month</text>`;
  histogram.setAttribute('viewBox', `0 0 ${width} ${height}`);
  histogram.innerHTML = chart;
}
threshold.addEventListener('input', drawHistogram);
window.addEventListener('resize', drawHistogram);
document.getElementById('filters').hidden = false;
drawHistogram();
