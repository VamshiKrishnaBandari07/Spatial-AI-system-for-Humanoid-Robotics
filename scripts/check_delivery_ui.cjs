/* Replay actual completed inference. Does not run models or create annotations. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('fs');
(async () => {
  const origin = process.env.SPATIALMEMORY_URL;
  const id = process.env.SPATIALMEMORY_EXISTING_JOB;
  const out = process.env.SPATIALMEMORY_UI_CHECK_OUTPUT;
  if (!origin || !id || !out) throw Error('Set URL, EXISTING_JOB and UI_CHECK_OUTPUT');
  const browser = await chromium.launch({executablePath:'/usr/bin/chromium', headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin);
    await page.waitForFunction(()=>document.getElementById('serviceStatus').dataset.state==='ready');
    assert.equal(await page.locator('[name=geometry_backend]').inputValue(),'vggt');
    await page.evaluate(id=>loadRun(id),id);
    assert.equal(await page.locator('#artifactLinks a').count(),5);
    const response=await page.request.get(`${origin}/jobs/${id}/artifacts/aligned`);
    assert.equal(response.status(),200);
    assert.match((await response.body()).toString(),/^ply\n/);
    await page.locator('#bv').dispatchEvent('error');
    assert.match(await page.locator('#beforePreview').innerText(),/preview is unavailable/);
    await page.locator('#bv').dispatchEvent('loadedmetadata');
    assert.equal(await page.locator('#beforePreview').isVisible(),false);
    await page.locator('.event').first().click();
    assert.match(await page.locator('#explanation').innerText(),/More evidence/);
    await page.screenshot({path:out+'.desktop.png',fullPage:true});
    await page.setViewportSize({width:375,height:812});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    await page.screenshot({path:out+'.mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(out+'.json',JSON.stringify({status:'passed',source_job:id,checks:['Neural profile default','Actual completed report and scene replay','Allowlisted PLY download','Video decoder error and recovery','Unknown decision wording','375px mobile layout','No page errors'],model_inference_started:false},null,2)+'\n');
    console.log('Delivery UI passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
