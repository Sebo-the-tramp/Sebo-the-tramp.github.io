const minimum = document.getElementById('dataset-relevance');
const search = document.getElementById('dataset-filter');
const body = document.querySelector('tbody');
const rows = Array.from(body.rows, row => ({
  row, name: row.dataset.names.toLowerCase(),
  title: row.cells[0].firstElementChild.textContent,
  count: row.querySelector('[data-paper-count]'), scores: Array.from(row.dataset.scores, Number),
  size: row.dataset.sizeGb === '' ? null : Number(row.dataset.sizeGb)
}));
const total = document.getElementById('dataset-size-total');
const sizes = new Intl.NumberFormat('en', {useGrouping: false, minimumFractionDigits: 1, maximumFractionDigits: 3});
const titles = new Intl.Collator('en', {sensitivity: 'base'});
function filter() {
  const term = search.value.trim().toLowerCase();
  const threshold = minimum.value === '' ? -1 : Number(minimum.value);
  let visible = 0;
  let gb = 0, known = 0;
  for (const item of rows) {
    const {row, name, count, scores, size} = item;
    const matching = scores.filter(score => score > threshold).length;
    item.matching = matching;
    count.textContent = matching;
    row.hidden = !name.includes(term) || (threshold >= 0 && matching === 0);
    if (!row.hidden) {
      visible++;
      if (size !== null) { gb += size; known++; }
    }
  }
  rows.sort((a, b) => b.matching - a.matching || titles.compare(a.title, b.title));
  const reordered = document.createDocumentFragment();
  for (const {row} of rows) reordered.append(row);
  body.replaceChildren(reordered);
  total.textContent = known ? sizes.format(gb) : '—';
  total.title = known + ' of ' + visible + ' visible datasets with a listed size';
  document.getElementById('no-datasets').hidden = visible > 0;
}
minimum.addEventListener('input', filter);
search.addEventListener('input', filter);
document.getElementById('filters').hidden = false;
filter();
