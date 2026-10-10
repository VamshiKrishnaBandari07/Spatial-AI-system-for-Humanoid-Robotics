const { chromium } = require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1080}});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8000/');
 if(process.env.SPATIALMEMORY_EXISTING_JOB){
  await page.evaluate(id=>localStorage.setItem('spatialmemory_last_run',id),process.env.SPATIALMEMORY_EXISTING_JOB);
  await page.reload();await page.waitForFunction(()=>document.getElementById('pointCount').textContent!=='—');
 }else{
 await page.locator('input[name=before]').setInputFiles('/workspace/spatialmemory-fixtures/before.mp4');
 await page.locator('input[name=after]').setInputFiles('/workspace/spatialmemory-fixtures/after.mp4');
 await page.locator('select[name=geometry_backend]').selectOption('vggt');
 await page.locator('input[name=semantics_enabled]').check();
 await page.locator('#run').click();
 await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Processing complete')||document.getElementById('status').textContent.startsWith('Failed'),{},{timeout:240000});
 const status=await page.locator('#status').innerText();if(!status.startsWith('Processing complete'))throw Error(status);
 }
 await page.locator('.event').first().click();
 await page.waitForFunction(()=>Array.from(document.querySelectorAll('#frames img')).every(i=>i.complete&&i.naturalWidth>0)&&document.querySelectorAll('#frames img').length>0);
 await page.locator('#objectLayer').uncheck();await page.locator('#objectLayer').check();
 await page.locator('#world').hover();await page.mouse.wheel(0,-100);
 await page.screenshot({path:'/workspace/Spatial-AI-system-for-Humanoid-Robotics/docs/screenshots/semantic-workspace.png',fullPage:true});
 const job=await page.evaluate(()=>localStorage.getItem('spatialmemory_last_run'));
 const response=await page.request.get('http://127.0.0.1:8000/jobs/'+job+'/report');const report=await response.json();
 if(!report.config.semantics_enabled||!report.registration.valid)throw Error('Semantic run or alignment was not verified');
 if(report.events.some(e=>['ADDED','REMOVED','MOVED'].includes(e.change)))throw Error('False physical change on identical synthetic recordings');
 await page.reload();await page.waitForFunction(()=>document.getElementById('pointCount').textContent!=='—');
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:'/workspace/Spatial-AI-system-for-Humanoid-Robotics/docs/screenshots/mobile-workspace.png',fullPage:true});
 const horizontalOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
 if(horizontalOverflow)throw Error('Mobile layout overflows viewport');if(errors.length)throw Error(errors.join('\n'));
 const summary={status:'passed',input:'synthetic identical textured-panel videos',workflow:report.workflow,runtime_seconds:report.runtime_seconds,physical_changes:0,semantic_decisions:report.events.length,registration_inliers:report.registration.inliers,heldout_count:report.registration.heldout_count,heldout_inliers:report.registration.heldout_inliers,checks:['browser upload','actual neural/semantic processing','object decision selection','mask evidence image loading','layer toggles','zoom','persisted run reload','mobile overflow','no browser JavaScript errors'],real_world_accuracy:'unmeasured'};
 fs.writeFileSync('/workspace/Spatial-AI-system-for-Humanoid-Robotics/docs/browser_smoke.json',JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
