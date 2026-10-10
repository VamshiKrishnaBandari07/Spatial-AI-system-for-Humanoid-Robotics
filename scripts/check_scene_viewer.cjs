const fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const run=process.argv[2],out=process.argv[3];
 if(!run||!out)throw Error('Usage: node scripts/check_scene_viewer.cjs RUN OUTPUT_JSON');
 const report=JSON.parse(fs.readFileSync(path.join(run,'report.json'),'utf8'));
 if(report.status!=='complete'||!report.registration.valid)throw Error('Completed gated inference required');
 const browser=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox'],headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:900}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setContent(fs.readFileSync(path.join(run,'visualization/index.html'),'utf8'));
  const expected=report.events.filter(e=>e.change==='UNKNOWN').length;
  await page.selectOption('#filter','UNKNOWN');
  if(await page.locator('#decisions button').count()!==expected)throw Error('Decision count mismatch');
  if(expected){await page.locator('#decisions button').first().click();if(!(await page.locator('#detail').innerText()).includes('UNKNOWN'))throw Error('Missing evidence detail');}
  await page.uncheck('#before');await page.check('#before');await page.mouse.move(400,350);await page.mouse.down();await page.mouse.move(520,420);await page.mouse.up();await page.mouse.wheel(0,-120);
  if(errors.length)throw Error(JSON.stringify(errors));
  const screenshot=path.join(run,'visualization/scene_viewer_browser.png');await page.screenshot({path:screenshot});
  fs.writeFileSync(out,JSON.stringify({status:'passed',errors,unknown_decisions:expected,checks:['Actual saved scene loaded via setContent','Unknown filter matches report','Evidence detail selection','Before layer toggled','Drag rotation','Wheel zoom'],screenshot},null,2)+'\n');console.log('Scene viewer passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
