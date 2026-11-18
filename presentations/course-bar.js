(() => {
  const mount = document.querySelector('[data-course-bar]');
  if (!mount) return;
  const page = mount.dataset.courseBar;
  const lesson = Number(mount.dataset.lesson);
  const base = mount.dataset.base || '';
  const github = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>';
  const sun = '<svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M11 1h2v5h-2zM11 18h2v5h-2zM18 11h5v2h-5zM1 11h5v2H1zM18.8 3.8l1.4 1.4-3.5 3.5-1.4-1.4zM7.3 15.3l1.4 1.4-3.5 3.5-1.4-1.4zM18.8 20.2l1.4-1.4-3.5-3.5-1.4 1.4zM7.3 8.7l1.4-1.4-3.5-3.5-1.4 1.4z"/></svg>';
  const moon = '<svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.6A8.6 8.6 0 0 1 9.4 3.5a8.6 8.6 0 1 0 11.1 11.1z"/></svg>';
  const lessons = Array.from({ length: 7 }, (_, index) => {
    const number = index + 1;
    return `<a href="${base}leerdoelen/les${number}.html"${page === 'goals' && number === lesson ? ' aria-current="page"' : ''}>Les ${number}</a>`;
  }).join('');
  const topics = [
    ['harness', '1', 'Agentic AI'], ['statistiek', '3', 'Statistiek'],
    ['tijd', '4', 'De residual'], ['vectorruimtes', '5', 'Vectorruimtes'],
    ['mcp', '6', 'MCP en skills'],
  ].map(([id, number, title]) => `<a href="#${id}"><span class="no">${number}</span> ${title}</a>`).join('');
  const primary = page === 'recap'
    ? `<nav class="primary-links" id="toc" aria-label="Onderwerpen">${topics}</nav>`
    : `<nav class="primary-links" aria-label="Visuele leerdoelen per les">${lessons}</nav>`;
  const slides = { 1: 'lesson1-tensors.html', 2: 'lesson2-kernels_states_gates.html' };
  const crossLinks = [
    ['start', 'index.html', 'Start'],
    ...(slides[lesson] ? [['slides', slides[lesson], `Slides les ${lesson}`]] : []),
    ...(page === 'recap' ? [['goals', 'index.html#lessons', 'Leerdoelen']] : []),
    ['exercise', 'your-study.html', 'Methodology'],
    ['example', 'lesson1-class-rehearsal.html', 'Class example'],
    ['notation', 'notatie.html', 'Notatie'],
    ['recap', 'dav-recap.html', 'DAV-recap'],
  ];
  const current = id => id === page && id !== 'goals' ? ' aria-current="page"' : '';
  const cross = crossLinks.map(([id, href, label]) => `<a href="${base}${href}"${current(id)}>${label}</a>`).join('');
  const brandLabel = { recap: 'DAV-recap', notation: 'notatie', start: 'start', exercise: 'methodology', goals: `les ${lesson}` }[page] || '';
  mount.outerHTML = `<header class="course-bar"><a class="skip" href="#top">Sla navigatie over</a><div class="inner">
    <div class="row"><span class="tools"><a class="repo" href="https://github.com/raoulg/MADS-MachineLearning-course" aria-label="MADS-ML repository op GitHub">${github}</a>
    <button class="theme-toggle" type="button" aria-label="Zet donker thema aan" data-mode="light">${sun}${moon}<span>Donker</span></button></span>
    <a class="brand" href="${base}index.html">MADS-ML${brandLabel ? ` <span>· ${brandLabel}</span>` : ''}</a>
    ${page === 'recap' ? '<a class="review" id="cnt" href="#reviewlijst"><b id="n">0</b><span>reviewlijst</span></a>' : ''}
    <nav class="cross-links" aria-label="Andere cursuspagina’s">${cross}</nav>
</div>
    ${primary}
  </div></header>`;
  const bar = document.querySelector('.course-bar');
  const inner = bar.querySelector('.inner');
  const primaryNav = bar.querySelector('.primary-links');
  // The page's own links take a row of their own when they do not fit beside the
  // cross links; the bar's height then goes to --course-bar-height for scroll offsets.
  const phone = matchMedia('(max-width: 760px)');
  const fitBar = () => {
    inner.classList.remove('stacked');
    if (phone.matches || primaryNav.scrollWidth > primaryNav.clientWidth + 1) inner.classList.add('stacked');
    document.documentElement.style.setProperty('--course-bar-height', `${bar.offsetHeight}px`);
  };
  addEventListener('resize', fitBar);
  addEventListener('load', fitBar);
  document.fonts?.ready.then(fitBar);
  fitBar();
  const root = document.documentElement;
  const button = document.querySelector('.course-bar .theme-toggle');
  const key = 'mads-course-theme';
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(key) ?? localStorage.getItem('dav-recap-theme')); } catch {}
  if (saved === 'dark' || saved === 'light') root.dataset.theme = saved;
  const media = matchMedia('(prefers-color-scheme: dark)');
  const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : media.matches;
  const paint = () => {
    const dark = isDark();
    button.dataset.mode = dark ? 'dark' : 'light';
    button.setAttribute('aria-label', dark ? 'Zet licht thema aan' : 'Zet donker thema aan');
    button.querySelector('span').textContent = dark ? 'Licht' : 'Donker';
  };
  button.addEventListener('click', () => {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    try { localStorage.setItem(key, JSON.stringify(root.dataset.theme)); } catch {}
    paint();
  });
  media.addEventListener?.('change', paint);
  addEventListener('storage', event => {
    if (event.key !== key) return;
    try {
      const value = JSON.parse(event.newValue);
      if (value === 'dark' || value === 'light') root.dataset.theme = value;
      else delete root.dataset.theme;
      paint();
    } catch {}
  });
  if (page === 'recap') {
    document.addEventListener('DOMContentLoaded', () => {
      const toc = document.getElementById('toc');
      const links = [...toc.querySelectorAll('a')];
      const sections = links.map(link => document.getElementById(link.hash.slice(1)));
      const markSection = () => {
        const edge = bar.offsetHeight + 40;
        let current = -1;
        sections.forEach((section, index) => { if (section?.getBoundingClientRect().top <= edge) current = index; });
        if (sections.at(-1)?.getBoundingClientRect().bottom < edge) current = -1;
        links.forEach((link, index) => {
          if (index === current) {
            if (link.getAttribute('aria-current') !== 'true') {
              link.setAttribute('aria-current', 'true');
              const left = link.offsetLeft - toc.offsetLeft;
              if (left < toc.scrollLeft || left + link.offsetWidth > toc.scrollLeft + toc.clientWidth) toc.scrollLeft = left - 8;
            }
          } else link.removeAttribute('aria-current');
        });
      };
      addEventListener('scroll', markSection, { passive: true });
      addEventListener('resize', markSection);
      addEventListener('load', markSection);
      addEventListener('hashchange', () => setTimeout(markSection, 0));
      markSection();
    });
  }
  paint();
})();
