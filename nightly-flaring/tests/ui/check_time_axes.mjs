/** D66 acceptance runner. NODE_PATH=/path/to/node_modules node tests/ui/check_time_axes.mjs
 * Starts and closes its own local server; uses desktop system Chrome and local
 * vendor copies for the page's two unchanged CDN script URLs. No live APIs.
 */
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {checkRound2} from './round2_assertions.mjs';
import {checkRound3} from './round3_assertions.mjs';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const requested=process.argv.slice(2),builds=requested.length?requested:['site','site_long_1500','site_long_5000'];
const out=path.join(root,'outputs/ui_checks');
const server=createServer(async(req,res)=>{
  try{
    const rel=decodeURIComponent(new URL(req.url,'http://localhost').pathname),file=path.resolve(root,'outputs','.'+rel);
    if(!file.startsWith(path.join(root,'outputs')+path.sep)){res.writeHead(403).end();return;}
    const body=await fs.readFile(rel.endsWith('/')?path.join(file,'index.html'):file);
    res.setHeader('Content-Type',file.endsWith('.json')?'application/json':file.endsWith('.js')?'text/javascript':file.endsWith('.png')?'image/png':'text/html');res.end(body);
  }catch{res.writeHead(404).end('Not found');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
let browser;
const reports={};
async function dataReady(page){await page.waitForFunction(()=>document.documentElement.dataset.ready==='true'&&document.documentElement.dataset.clearpasses==='true',{timeout:30000});}
async function snapshot(page){return page.evaluate(()=>({tiles:[...document.querySelectorAll('#tiles .num')].map(e=>e.textContent),contexts:[...document.querySelectorAll('#tiles .ctx')].map(e=>e.textContent),top:[...document.querySelectorAll('#rank li')].slice(0,25).map(e=>{const copy=e.cloneNode(true);copy.querySelectorAll('svg').forEach(s=>s.remove());return copy.textContent.replace(/\s+/g,' ').trim();}),regions:[...document.querySelectorAll('#regions tbody tr')].map(e=>[...e.children].slice(0,4).map(c=>c.textContent))}));}
async function mark(report,name,fn){try{report.parity[name]={pass:true,value:await fn()};}catch(error){report.parity[name]={pass:false,error:error.message};}}
try {
  browser=await chromium.launch({channel:'chrome',headless:true});
  for(const build of builds){
    const folder=path.join(out,build);await fs.mkdir(folder,{recursive:true});
    const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'no-preference'});
    await context.route('https://**/*',async route=>{
      const url=route.request().url();
      if(url.includes('d3.min.js'))return route.fulfill({path:path.join(root,'outputs/vendor/d3.min.js'),contentType:'text/javascript'});
      if(url.includes('topojson-client.min.js'))return route.fulfill({path:path.join(root,'outputs/vendor/topojson-client.min.js'),contentType:'text/javascript'});
      return route.abort();
    });
    const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    const report={budgets:{},parity:{},correctness:{},screenshots:[],errors};reports[build]=report;
    await page.goto(`${origin}/${build}/?debug`,{waitUntil:'domcontentloaded'});
    try{await dataReady(page);}catch(e){report.loadError=e.message;await page.screenshot({path:path.join(folder,'load-error.png'),fullPage:true});continue;}
    report.record=await page.evaluate(()=>({nights:__timeAxes.state.range[1]+1,sites:__timeAxes.sites.length}));
    const cdp=await context.newCDPSession(page);await cdp.send('HeapProfiler.collectGarbage');const heap=await cdp.send('Runtime.getHeapUsage');
    report.budgets=await page.evaluate(async()=>{
      const a=__timeAxes,stats=[],refresh=[];
      for(let i=0;i<7;i++){let t=performance.now();a.computeRange();stats.push(performance.now()-t);t=performance.now();a.setRange(0,a.sites[0].st.length-1);await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));refresh.push(performance.now()-t);}
      const median=x=>x.sort((a,b)=>a-b)[Math.floor(x.length/2)];
      return {first_paint_ms:a.metrics.firstPaint,interactive_ms:a.metrics.interactive,range_statistics_ms:median(stats),range_refresh_ms:median(refresh),heap_mb:performance.memory?.usedJSHeapSize/1e6};
    });
    report.budgets.heap_mb=(heap.usedSize+(heap.backingStorageSize||0))/1e6;
    report.budgets.heap_details=heap;
    // All-range tiles, top 25 names/figures and regional figures use displayed text.
    const realSnapshot=await snapshot(page);
    if(build==='site'){
      const before=await context.newPage();await before.goto(`${origin}/site_before/`,{waitUntil:'load'});
      const baseline=await snapshot(before);
      // A longer record (D68) prepends nights: compare the baseline's own date range.
      const nightsOf=p=>p.evaluate(()=>JSON.parse(document.querySelector('#flaring-data').textContent).nights);
      const oldNights=await nightsOf(before);await before.close();
      const newNights=await nightsOf(page),offset=newNights.indexOf(oldNights[0]);
      let compared=realSnapshot;
      if(offset>0&&newNights.at(-1)===oldNights.at(-1)){
        await page.evaluate(([a,b])=>__timeAxes.setRange(a,b),[offset,newNights.length-1]);await page.waitForTimeout(400);
        compared=await snapshot(page);await page.evaluate(n=>__timeAxes.setRange(0,n-1),newNights.length);await page.waitForTimeout(400);
      }
      report.correctness.baseline_offset=offset;report.correctness.before=baseline;report.correctness.after=compared;
      report.correctness.all_range_tiles=JSON.stringify(baseline.tiles)===JSON.stringify(compared.tiles);
      report.correctness.all_range_contexts=JSON.stringify(baseline.contexts)===JSON.stringify(compared.contexts);
      report.correctness.all_range_top25=JSON.stringify(baseline.top)===JSON.stringify(compared.top);
      report.correctness.all_range_regions=JSON.stringify(baseline.regions)===JSON.stringify(compared.regions);
    }
    report.correctness.block_sums=await page.evaluate(()=>{
      const a=__timeAxes,ids=[a.sites.find(s=>s.rh),a.sites.find(s=>!s.rh),...a.sites.filter(s=>s.cp).slice(0,8)];
      let maxMeanError=0,maxCountError=0;
      for(const s of ids)for(const [lo,hi] of [[0,s.st.length-1],[1,30],[31,65],[33,96],[s.st.length-29,s.st.length-1]]){
        const expected=[0,0,0,0,0,0];
        for(let i=lo;i<=hi;i++){const c=s.st[i],v=c==='C'?0:s.rh?.[i];if(Number.isFinite(v)){expected[0]+=v;expected[1]++;}expected[2]+=c==='F';expected[3]+=c!=='N';expected[4]+=c==='D';expected[5]+=s.cp?.[i]||0;}
        const q=a.rangeStats(s,lo,hi);maxMeanError=Math.max(maxMeanError,Math.abs((q.mean||0)-(expected[1]?expected[0]/expected[1]:0)));
        maxCountError=Math.max(maxCountError,...[q.clear,q.lit,q.obs,q.cloud,q.passes].map((v,i)=>Math.abs(v-expected[i+1])));
      }
      return {max_mean_error:maxMeanError,max_count_error:maxCountError,nonflaring_sites_allocate_no_arrays:a.sites.filter(s=>!s.everLit).every(s=>s.rh===null)};
    });
    const month=await page.evaluate(()=>{const a=__timeAxes;return a.calendarBins('month').find(d=>!d.partial&&d.a>0)||a.calendarBins('month')[0];});
    await page.evaluate(d=>__timeAxes.setRange(d.a,d.b),month);
    const monthMean=(await snapshot(page)).tiles[1];
    await page.evaluate(d=>__timeAxes.showStep('month',d.a),month);
    const monthStep=(await snapshot(page)).tiles[1];
    const monthMW=await page.evaluate(d=>__timeAxes.networkBin(d).mw,month);
    const overview=await page.evaluate(()=>{const a=__timeAxes,bar=document.querySelector('#record-overview .time-bar');return a.calendarBins(bar?document.querySelector('#record-overview').getAttribute('aria-label').match(/1 bar = 1 (\w+)/)[1]:'month').find(d=>!d.partial)||a.calendarBins('month')[0];});
    const box=await page.locator('#record-overview').boundingBox();
    await page.mouse.move(box.x+(overview.a+overview.b+1)/2/report.record.nights*box.width,box.y+10);
    const tooltipMW=+(await page.locator('#tl-tip').getAttribute('data-mw')),tooltipText=await page.locator('#tl-tip').innerText();
    const overviewExpected=await page.evaluate(d=>__timeAxes.networkBin(d).mw,overview);
    assert.equal(tooltipMW,overviewExpected);
    const displayed=Math.round(monthMW)+'MW';
    report.correctness.month={month,overview_mw:monthMW,overview_tooltip_mw:tooltipMW,overview_tooltip_text:tooltipText,range_tile:monthMean,step_tile:monthStep,equal:monthMean===monthStep&&monthMean===displayed&&tooltipMW===overviewExpected};
    await mark(report,'overview brush, pan and year controls',async()=>{
      await page.getByRole('button',{name:'All',exact:true}).click();const box=await page.locator('#record-overview').boundingBox();
      await page.mouse.move(box.x+box.width*.3,box.y+10);await page.mouse.down();await page.mouse.move(box.x+box.width*.6,box.y+10,{steps:5});await page.mouse.up();
      const selected=await page.evaluate(()=>__timeAxes.state.range);assert.notDeepEqual(selected,[0,report.record.nights-1]);const selection=await page.locator('#record-overview .selection').boundingBox();
      await page.mouse.move(selection.x+selection.width/2,selection.y+10);await page.mouse.down();await page.mouse.move(selection.x+selection.width/2+20,selection.y+10,{steps:4});await page.mouse.up();
      const moved=await page.evaluate(()=>__timeAxes.state.range);assert.equal(moved[1]-moved[0],selected[1]-selected[0]);
      await page.locator('#year-select').selectOption({index:1});const range=await page.evaluate(()=>__timeAxes.state.range);assert.ok(range[1]-range[0]<366);
      await page.locator('#year-axis button').first().click();
    });
    await mark(report,'click a night',async()=>{
      await page.getByRole('button',{name:'Last 30',exact:true}).click();const box=await page.locator('#strip').boundingBox();await page.mouse.click(box.x+box.width/3,box.y+20);
      assert.equal(await page.evaluate(()=>__timeAxes.state.unit),'night');assert.ok(await page.evaluate(()=>__timeAxes.state.night>=0));
    });
    await mark(report,'drag a range',async()=>{
      const box=await page.locator('#strip').boundingBox();await page.mouse.move(box.x+5,box.y+20);await page.mouse.down();await page.mouse.move(box.x+box.width/2,box.y+20,{steps:6});await page.mouse.up();
      assert.ok(await page.evaluate(()=>__timeAxes.state.range[1]-__timeAxes.state.range[0]<29));
    });
    await mark(report,'presets',async()=>{for(const name of ['Last 7','Last 30','Last 90',...(report.record.nights>365?['12 months']:[]),'All']){await page.getByRole('button',{name,exact:true}).click();assert.equal(await page.getByRole('button',{name,exact:true}).getAttribute('aria-pressed'),'true');assert.equal(await page.locator('#presets button[aria-pressed="true"]').count(),1);}});
    await mark(report,'arrow Home End Esc keys',async()=>{
      await page.locator('#strip').focus();await page.keyboard.press('ArrowRight');await page.keyboard.press('End');assert.equal(await page.evaluate(()=>__timeAxes.state.night),report.record.nights-1);
      await page.keyboard.press('Home');assert.equal(await page.evaluate(()=>__timeAxes.state.night),0);await page.keyboard.press('Escape');assert.equal(await page.evaluate(()=>__timeAxes.state.night),-1);
    });
    await mark(report,'Play/Pause and fixed heat filter',async()=>{
      await page.locator('#pct').fill('50');await page.locator('#pct').dispatchEvent('input');
      const ids=()=>page.evaluate(()=>__timeAxes.sites.filter(s=>s.rFlare>0&&s.rMean>=__timeAxes.state.thr).map(s=>s.id));const initial=await ids();
      await page.getByRole('button',{name:'▶ Play',exact:true}).click();await page.waitForTimeout(700);await page.getByRole('button',{name:'❚❚ Pause',exact:true}).click();assert.deepEqual(await ids(),initial);
      await page.evaluate(()=>__timeAxes.showStep('week',__timeAxes.state.range[0]));await page.keyboard.press('ArrowRight');assert.deepEqual(await ids(),initial);
      await page.locator('#pct').fill('0');await page.locator('#pct').dispatchEvent('input');
    });
    await mark(report,'Flaring only',async()=>{await page.getByRole('button',{name:'Flaring only',exact:true}).click();assert.equal(await page.evaluate(()=>__timeAxes.state.flaringOnly),true);await page.getByRole('button',{name:'All sites',exact:true}).click();});
    await page.getByRole('button',{name:'All',exact:true}).click();
    await mark(report,'list orders and Show more',async()=>{for(const name of ['Most consistent','Most clear views','Strongest'])await page.getByRole('button',{name,exact:true}).click();const n=await page.locator('#rank li').count();await page.locator('#rank-more').click();assert.ok(await page.locator('#rank li').count()>n);});
    await mark(report,'list hover highlights map',async()=>{await page.locator('#rank button').first().hover();assert.ok(await page.evaluate(()=>__timeAxes.state.hl?.id));await page.mouse.move(0,0);});
    await mark(report,'open site from list; pass loading and window arrows',async()=>{
      await page.locator('#rank button').first().click();await page.waitForFunction(()=>document.querySelector('#detail .passbox svg'));assert.ok(await page.evaluate(()=>__timeAxes.state.selected?.id));
      const old=await page.evaluate(()=>__timeAxes.state.win[0]);await page.getByRole('button',{name:'Earlier nights',exact:true}).click();assert.ok(await page.evaluate(()=>__timeAxes.state.win[0])<old);await page.getByRole('button',{name:'Later nights',exact:true}).click();
    });
    await mark(report,'calendar and record bar navigation',async()=>{const old=await page.evaluate(()=>__timeAxes.state.win[0]);await page.locator('.calendar rect[data-night]').nth(10).click();assert.notEqual(await page.evaluate(()=>__timeAxes.state.win[0]),old);await page.locator('#detail .overview').click({position:{x:80,y:30}});});
    await page.getByRole('button',{name:'Close site detail'}).click();
    await mark(report,'regions zoom',async()=>{const before=await page.evaluate(()=>document.querySelector('#map').__zoom.k);await page.locator('#regions tbody tr').first().click();await page.waitForTimeout(750);assert.notEqual(await page.evaluate(()=>document.querySelector('#map').__zoom.k),before);});
    await mark(report,'open site from map',async()=>{
      await page.getByRole('button',{name:'World',exact:true}).click();await page.waitForTimeout(750);const hit=await page.evaluate(()=>{const h=__timeAxes.hits.at(-1);return h?[h[0],h[1]]:null;});assert.ok(hit);
      const box=await page.locator('#map').boundingBox();await page.mouse.click(box.x+hit[0],box.y+hit[1]);assert.ok(await page.evaluate(()=>__timeAxes.state.selected));await page.getByRole('button',{name:'Close site detail'}).click();
    });
    await mark(report,'Flare-outs selection, map, card hash',async()=>{
      await page.getByRole('button',{name:'Flare-outs',exact:true}).click();await page.locator('#fo-spells .fo-row').last().click();assert.ok((await page.evaluate(()=>location.hash)).startsWith('#flare-outs/card/'));
      await page.getByRole('button',{name:'Show on map',exact:true}).click();await page.waitForFunction(()=>__timeAxes.state.selected);await page.getByRole('button',{name:'Close site detail'}).click();
    });
    await mark(report,'dark-night filters, neighbour and hash links',async()=>{
      await page.getByRole('button',{name:'Flare-outs',exact:true}).click();await page.getByRole('button',{name:'Single dark nights',exact:true}).click();await page.waitForSelector('.dn-row');
      await page.getByLabel('With an EMIT scene',{exact:true}).uncheck();await page.getByLabel('Minimum recent heat',{exact:true}).selectOption('1');await page.getByLabel('Region',{exact:true}).selectOption({index:1});await page.getByLabel('Region',{exact:true}).selectOption('');await page.getByLabel('Minimum recent heat',{exact:true}).selectOption('any');
      await page.locator('.dn-row').first().click();assert.ok((await page.evaluate(()=>location.hash)).startsWith('#flare-outs/dark-nights/'));
      const linked=await page.evaluate(async()=>{const d=await(await fetch('darknights.json')).json();return d.runs.find(r=>r.neighbours.length)?.id;});assert.ok(linked);
      await page.evaluate(id=>location.hash='#flare-outs/dark-nights/'+encodeURIComponent(id),linked);
      const neighbour=page.locator('.dn-detail a[href^="#flare-outs/dark-nights/"]').first();await neighbour.waitFor();const oldHash=await page.evaluate(()=>location.hash);await neighbour.click();assert.notEqual(await page.evaluate(()=>location.hash),oldHash);
      await page.locator('.dn-histogram').scrollIntoViewIfNeeded();const hist=await page.locator('.dn-histogram').boundingBox();await page.mouse.move(hist.x+hist.width*.7,hist.y+10);await page.mouse.down();await page.mouse.move(hist.x+hist.width*.9,hist.y+10,{steps:5});await page.mouse.up();assert.match(await page.locator('#dn-content p[role="status"]').textContent(),/Run starts/);await page.getByRole('button',{name:'All dates',exact:true}).click();
    });
    await mark(report,'How it works rail',async()=>{await page.getByRole('button',{name:'How it works',exact:true}).click();await page.locator('#ex-toc a').last().click();assert.ok((await page.evaluate(()=>location.hash)).startsWith('#how-it-works/'));});
    await page.getByRole('button',{name:'Map',exact:true}).click();await page.getByRole('button',{name:'All',exact:true}).click();
    await mark(report,'shareable range, step and site state',async()=>{
      await page.getByRole('button',{name:'Last 90',exact:true}).click();await page.locator('#mode-week').click();await page.locator('#rank button').first().click();
      const expected=await page.evaluate(()=>({range:__timeAxes.state.range,night:__timeAxes.state.night,unit:__timeAxes.state.unit,site:__timeAxes.state.selected.id}));
      await page.reload({waitUntil:'domcontentloaded'});await dataReady(page);
      assert.deepEqual(await page.evaluate(()=>({range:__timeAxes.state.range,night:__timeAxes.state.night,unit:__timeAxes.state.unit,site:__timeAxes.state.selected.id})),expected);
      await page.getByRole('button',{name:'Close site detail'}).click();await page.getByRole('button',{name:'All',exact:true}).click();
    });
    await mark(report,'loading and HTTP error state',async()=>{
      const failure=await context.newPage();await failure.route('**/data/nights-*.json',async route=>{await new Promise(r=>setTimeout(r,400));return route.abort();});
      await failure.goto(`${origin}/${build}/?debug`,{waitUntil:'domcontentloaded'});
      await failure.locator('#record-overview rect').first().waitFor();
      await failure.getByRole('alert').waitFor();assert.match(await failure.getByRole('alert').textContent(),/data\/nights-/);assert.doesNotMatch(await failure.getByRole('alert').textContent(),/Serve this page over HTTP/);await failure.getByRole('button',{name:'Retry',exact:true}).waitFor();await failure.close();
      const mobile=await context.newPage({viewport:{width:390,height:1000}});await mobile.setViewportSize({width:390,height:1000});await mobile.goto(`${origin}/${build}/?debug`);await dataReady(mobile);
      assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth),390);
      const bars=await mobile.locator('#strip > rect:not([stroke])').count();assert.ok(bars>1);await mobile.close();
    });
    await page.evaluate(()=>{__timeAxes.showStep('month',__timeAxes.state.range[0]);__timeAxes.metrics.frames=[];__timeAxes.play();});await page.waitForTimeout(2000);
    report.budgets.playback_frame_median_ms=await page.evaluate(()=>{__timeAxes.pause();const f=__timeAxes.metrics.frames.sort((a,b)=>a-b);return f[Math.floor(f.length/2)];});
    await checkRound2({page,context,origin,build,report,snapshot});
    await checkRound3({page,context,origin,build,report,snapshot});
    if(!process.env.SKIP_SCREENSHOTS)for(const width of [1440,1024,390]){
      await page.setViewportSize({width,height:1000});
      for(const mode of ['range','night','week','month','site','flareouts','darknights']){
        await page.getByRole('button',{name:'Map',exact:true}).click();await page.evaluate(()=>{__timeAxes.select(null,false);__timeAxes.setRange(0,__timeAxes.sites[0].st.length-1);});
        if(['night','week','month'].includes(mode))await page.evaluate(unit=>__timeAxes.showStep(unit,__timeAxes.state.range[1]),mode);
        if(mode==='site'){await page.locator('#rank button').first().click();await page.waitForTimeout(100);}
        if(['flareouts','darknights'].includes(mode)){await page.getByRole('button',{name:'Flare-outs',exact:true}).click();await page.getByRole('button',{name:'Dark spells',exact:true}).click();}
        if(mode==='flareouts')await page.locator('#fo-spells .fo-row').first().click();
        if(mode==='darknights'){await page.getByRole('button',{name:'Single dark nights',exact:true}).click();await page.waitForSelector('.dn-row');}
        await page.evaluate(()=>{scrollTo(0,0);document.querySelector('#flareouts').scrollTop=0;document.querySelector('.panel').scrollTop=0;});
        const file=path.join(folder,`${width}-${mode}.png`);await page.screenshot({path:file,fullPage:true});report.screenshots.push(file);
        if(mode==='site'||mode==='flareouts'){
          const target=mode==='site'?'#detail .passbox':'#fo-selected';await page.locator(target).scrollIntoViewIfNeeded();
          const extra=path.join(folder,`${width}-${mode}-detail.png`);await page.screenshot({path:extra,fullPage:true});report.screenshots.push(extra);
        }
        if(width===390&&mode==='range')report.parity['phone layout at 390 px']={pass:await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),value:await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}))};
      }
    }
    await context.close();await fs.writeFile(path.join(folder,'report.json'),JSON.stringify(report,null,2));
  }
  // Raw HTML growth between the two long-record builds, per 1,000 nights.
  if(reports.site_long_1500&&reports.site_long_5000){const sizes=await Promise.all(['site_long_1500','site_long_5000'].map(async b=>(await fs.stat(path.join(root,'outputs',b,'index.html'))).size));reports.site_long_5000.budgets.index_growth_kb_per_1000=(sizes[1]-sizes[0])/3.5/1000;}
  for(const [build,report] of Object.entries(reports)){
    if(report.loadError)continue;
    const limits=build==='site_long_5000'?{index_growth_kb_per_1000:150,first_paint_ms:1000,interactive_ms:4000,range_statistics_ms:30,range_refresh_ms:150,playback_frame_median_ms:42,heap_mb:350}:build==='site_long_1500'?{interactive_ms:2000}:{};
    report.budget_checks=Object.fromEntries(Object.entries(limits).map(([key,limit])=>[key,{measured:report.budgets[key],budget:limit,pass:report.budgets[key]<=limit}]));
    await fs.writeFile(path.join(out,build,'report.json'),JSON.stringify(report,null,2));
    await fs.rm(path.join(out,build,'load-error.png'),{force:true});
  }
} finally {
  await browser?.close();await new Promise(resolve=>server.close(resolve));
  await fs.mkdir(out,{recursive:true});await fs.writeFile(path.join(out,'time_axes_report.json'),JSON.stringify(reports,null,2));
}
console.log(JSON.stringify(Object.fromEntries(Object.entries(reports).map(([b,r])=>[b,{budgets:r.budgets,parity:r.parity,correctness:{...r.correctness,before:undefined,after:undefined},errors:r.errors,loadError:r.loadError,screenshots:r.screenshots.length}])),null,2));
if(Object.values(reports).some(r=>r.loadError||r.errors.length||Object.values(r.parity).some(p=>!p.pass)||Object.values(r.budget_checks||{}).some(p=>!p.pass)||r.correctness.block_sums?.max_count_error>0||r.correctness.block_sums?.max_mean_error>1e-6||r.correctness.month?.equal===false||r.correctness.all_range_contexts===false||r.correctness.all_range_tiles===false||r.correctness.all_range_top25===false||r.correctness.all_range_regions===false))process.exitCode=1;
