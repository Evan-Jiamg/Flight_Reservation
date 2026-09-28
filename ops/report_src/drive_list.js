// List the files of public Google Drive folders (no login): name, id, type; recurses into subfolders.
const { chromium } = require('C:/Users/User/node_modules/playwright');
const roots = { 'TREC-reports': '1wcPFN-5giFqDkuUE-SB_y1G-4ujGVuQt', 'internal-meetings': '1VzgeVqtecsmCY2zkoN5NQJY5xS-Nk-Qu' };
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const out = [];
  async function list(label, id, depth) {
    await page.goto('https://drive.google.com/drive/folders/' + id, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(3000);
    for (let i = 0; i < 8; i++) { await page.mouse.wheel(0, 3000); await page.waitForTimeout(700); }
    const items = await page.evaluate(() => {
      const r = [];
      document.querySelectorAll('[data-id]').forEach(el => {
        const id = el.getAttribute('data-id');
        const txt = (el.innerText || '').split('\n').map(s => s.trim()).filter(Boolean);
        const tip = el.querySelector('[data-tooltip]');
        const aria = el.getAttribute('aria-label') || '';
        r.push({ id, name: txt[0] || aria, aria, tip: tip ? tip.getAttribute('data-tooltip') : '' });
      });
      return r;
    });
    const seen = new Set();
    for (const it of items) {
      if (!it.id || it.id.length < 20 || seen.has(it.id) || it.id === id) continue;
      seen.add(it.id);
      const isFolder = /folder|資料夾/i.test(it.aria + ' ' + it.tip);
      out.push({ root: label, depth, id: it.id, name: it.name, folder: isFolder, aria: it.aria.slice(0, 120) });
      if (isFolder && depth < 2) await list(label + '/' + it.name, it.id, depth + 1);
    }
  }
  for (const [label, id] of Object.entries(roots)) await list(label, id, 0);
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error('ERR', e.message); process.exit(1); });
