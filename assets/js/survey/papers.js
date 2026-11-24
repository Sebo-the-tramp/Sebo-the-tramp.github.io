const body = document.querySelector('tbody');
const rows = Array.from(body.rows, row => {
  const [paper, submitted, relevance, citations] = row.cells;
  const month = submitted.dataset.month;
  return {row, paper: paper.textContent, month, submitted: submitted.dataset.date,
    searchText: (paper.textContent + ' ' + row.dataset.arxivId).toLowerCase(),
    year: month.slice(0, 4), relevance: relevance.dataset.score, citations: citations.dataset.count};
});
const headers = Array.from(document.querySelectorAll('[data-sort]'));
const search = document.getElementById('title-filter');
const year = document.getElementById('year-filter');
const month = document.getElementById('month-filter');
const relevance = document.getElementById('relevance-filter');
const count = document.getElementById('match-count');
const noMatches = document.getElementById('no-matches');
const titles = new Intl.Collator('en', {sensitivity: 'base'});
let sortKey = 'relevance';
let ascending = false;

function compare(a, b) {
  const left = a[sortKey], right = b[sortKey];
  if (left !== right) {
    if (!left || !right) return Number(!left) - Number(!right);
    const numeric = sortKey === 'relevance' || sortKey === 'citations';
    const difference = numeric ? Number(left) - Number(right) : titles.compare(left, right);
    if (difference) return ascending ? difference : -difference;
  }
  return Number(b.relevance) - Number(a.relevance)
    || titles.compare(b.submitted, a.submitted) || titles.compare(a.paper, b.paper);
}

function render() {
  const words = search.value.toLowerCase().trim()
    .replace(/(?:https?:\/\/(?:www\.|export\.)?arxiv\.org\/(?:abs|pdf|html)\/|arxiv:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?=\s|$)/g, '$1')
    .split(/\s+/);
  const above = relevance.value.startsWith('above-') ? Number(relevance.value.slice(6)) : null;
  const matches = rows.filter(item => words.every(word => item.searchText.includes(word))
    && (!year?.value || item.year === year.value)
    && (!month?.value || item.month === month.value)
    && (!relevance.value || (above === null ? (item.relevance || 'pending') === relevance.value
      : Number(item.relevance) > above)));
  matches.sort(compare);
  const visible = document.createDocumentFragment();
  for (const item of matches) visible.append(item.row);
  body.replaceChildren(visible);
  for (const header of headers) {
    if (header.dataset.sort === sortKey) header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
    else header.removeAttribute('aria-sort');
  }
  count.textContent = matches.length + ' of ' + rows.length + ' papers';
  count.hidden = matches.length === rows.length;
  noMatches.hidden = matches.length > 0;
}

for (const header of headers) {
  header.firstElementChild.addEventListener('click', () => {
    ascending = sortKey === header.dataset.sort ? !ascending : header.dataset.sort === 'paper';
    sortKey = header.dataset.sort;
    render();
  });
}
const query = new URLSearchParams(location.search);
for (const [key, control] of [['month', month], ['relevance', relevance]]) {
  const value = query.get(key);
  if (control && Array.from(control.options).some(option => option.value === value)) control.value = value;
}
for (const control of [search, year, month, relevance]) {
  control?.addEventListener('input', render);
}
document.getElementById('filters').hidden = false;
render();
