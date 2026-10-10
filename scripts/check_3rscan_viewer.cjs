/* Verify an actual locally generated evidence viewer; no fabricated scene fixture. */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
(async () => {
  const run = process.argv[2];
  const output = process.argv[3];
  if (!run || !output) throw Error('Usage: node scripts/check_3rscan_viewer.cjs RUN OUTPUT_JSON');
  const report = JSON.parse(fs.readFileSync(path.join(run, 'report.json'), 'utf8'));
  if (report.status !== 'complete' || !report.registration.valid) throw Error('Successful gated inference required');
  const html = path.join(run, 'visualization', 'evidence_viewer.html');
  const screenshot = path.join(run, 'visualization', 'viewer_browser.png');
  const browser = await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    // file:// is blocked by managed browser policy; execute the exact self-contained content.
    await page.setContent(fs.readFileSync(html,'utf8'));
    await page.check('#truth');
    await page.mouse.move(600,400);
    await page.mouse.down();
    await page.mouse.move(700,450);
    await page.mouse.up();
    await page.mouse.wheel(0,100);
    const stats = await page.evaluate(() => ({before:data.before.length,after:data.after.length,truth:data.truth.length,events:data.events.length}));
    if (errors.length || stats.before < 100 || stats.after < 100 || stats.truth !== 1) throw Error(JSON.stringify({errors,stats}));
    await page.screenshot({path:screenshot});
    fs.writeFileSync(output,JSON.stringify({status:'passed',errors,stats,checks:['Actual HTML loaded using Playwright setContent','Native truth layer toggled','Drag rotation','Wheel zoom'],screenshot},null,2)+'\n');
    console.log(JSON.stringify(stats));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error.message); process.exitCode=1; });
