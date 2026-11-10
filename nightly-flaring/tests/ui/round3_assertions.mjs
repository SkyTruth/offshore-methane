/** Round 3 regressions: exercise the review's failing paths, not just defaults. */
import assert from 'node:assert/strict';

export async function checkRound3({page,context,origin,build,report,snapshot}) {
  const mark=async(name,fn)=>{try{report.parity[name]={pass:true,value:await fn()};}catch(e){report.parity[name]={pass:false,error:e.stack};}console.log(build,name,report.parity[name].pass?'PASS':'FAIL');};
  const reset=()=>page.evaluate(()=>{const a=__timeAxes;a.pause();a.select(null,false);a.state.sort='strong';a.state.pct=0;a.state.flaringOnly=false;a.setRange(0,a.sites[0].st.length-1);});
  await reset();
  await mark('R3 O2-1 M2-2 uniform month ticks name every tick year in every layout/range',async()=>{
    const checks=[];
    for(const width of [1440,1024,390]) {
      await page.setViewportSize({width,height:1000});
      // Wait for ResizeObserver to finish before testing the rendered axis.
      await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
      checks.push(...await page.evaluate(()=>{
        const a=__timeAxes,nights=JSON.parse(document.querySelector('#flaring-data').textContent).nights,n=nights.length;
        const ranges=[['All',0,n-1],['12 months',Math.max(0,n-365),n-1],['Last 90',n-90,n-1],...Array.from(new Set(nights.map(d=>d.slice(0,4))),y=>['Year '+y,nights.findIndex(d=>d.startsWith(y)),nights.findLastIndex(d=>d.startsWith(y))])];
        return ranges.map(([name,lo,hi])=>{
          a.setRange(lo,hi);
          const modes=[...document.querySelectorAll('.modebar .seg button')].filter(b=>!b.hidden);if(new Set(modes.map(b=>b.getBoundingClientRect().top)).size!==1||modes.some(b=>b.scrollWidth>b.clientWidth))throw Error('Mode buttons wrap or text overflows');
          const root=document.querySelector('#strip'),unit=root.getAttribute('aria-label').match(/1 bar = 1 (\w+)/)[1];
          const ticks=[...root.querySelectorAll('text[data-date]')],years=new Map();
          ticks.forEach(t=>{const y=t.dataset.date.slice(0,4);if(!years.has(y))years.set(y,[]);years.get(y).push(t.textContent);});
          for(const [year,labels] of years)if(!labels[0].includes(year))throw Error(name+' first kept tick omits '+year+': '+labels);
          const months=ticks.map(t=>{const d=new Date(t.dataset.date);return d.getUTCFullYear()*12+d.getUTCMonth();});
          const gaps=months.slice(1).map((v,i)=>v-months[i]);
          if(unit!=='night'&&(new Set(gaps).size>1||gaps.some(k=>![1,2,3,6,12,24,60,120].includes(k))))throw Error(name+' irregular month spacing '+gaps);
          let right=-Infinity;for(const t of ticks){const r=t.getBoundingClientRect();if(r.left<right+2)throw Error(name+' labels closer than 2 px');right=r.right;}
          const caption=root.querySelector('.bin-caption').getBoundingClientRect();if(ticks.some(t=>t.getBoundingClientRect().bottom>caption.top))throw Error('caption shares tick row');
          return {width:innerWidth,range:name,unit,ticks:ticks.map(t=>[t.dataset.date,t.textContent]),month_stride:gaps[0]||null};
        });
      }));
    }
    await page.setViewportSize({width:1440,height:1000});return checks;
  });
  await reset();
  await mark('R3 S2-1 O2-2 O2-9 O2-16 old-range calendar, rows, native expansion keys and Escape',async()=>{
    const fresh=await context.newPage();
    try {
      const dates=await page.evaluate(()=>JSON.parse(document.querySelector('#flaring-data').textContent).nights);
      const end=build==='site'?dates.length-1:60,from=dates[Math.max(0,end-30)],to=dates[end];
      await fresh.goto(`${origin}/${build}/?debug#map?from=${from}&to=${to}&step=week&at=${to}&site=7`);
      await fresh.waitForFunction(()=>document.documentElement.dataset.ready==='true');
      const verify=()=>fresh.evaluate(()=>{
        const a=__timeAxes,root=document.querySelector('.calendar'),stops=root.querySelectorAll('[tabindex="0"]');
        if(stops.length!==1)throw Error('Expected one roving stop, got '+stops.length);
        if(root.querySelectorAll('[role="row"]').length<7)throw Error('Calendar has no grid rows');
        const d=a.state.steps[a.state.stepIndex];
        for(const i of [a.state.range[1],d.a,d.b,...a.state.win])if(!root.querySelector(`[data-night="${i}"]`))throw Error('Selected night hidden: '+i);
        return {cells:root.querySelectorAll('[data-night]').length,cursor:+stops[0].dataset.night};
      });
      const initial=await verify();assert.equal(initial.cursor,await fresh.evaluate(()=>__timeAxes.state.win[1]));
      if(build!=='site') {
        await fresh.getByRole('button',{name:'Show earlier years',exact:true}).focus();await fresh.keyboard.press('Enter');
        assert.equal(await fresh.locator('.calendar [data-night]').count(),dates.length);
        // A separate initial panel checks Space's native button activation too.
        await fresh.reload();await fresh.waitForFunction(()=>document.documentElement.dataset.ready==='true');
        await fresh.getByRole('button',{name:'Show earlier years',exact:true}).focus();await fresh.keyboard.press('Space');
        assert.equal(await fresh.locator('.calendar [data-night]').count(),dates.length);
      }
      await fresh.locator('.calendar [tabindex="0"]').focus();await fresh.keyboard.press('ArrowLeft');
      assert.equal(await fresh.locator('.calendar [tabindex="0"]').count(),1);
      await fresh.keyboard.press('Escape');assert.equal(await fresh.evaluate(()=>__timeAxes.state.selected),null);
      // Cached panel must also render an old selection after initially opening All.
      await fresh.evaluate(()=>{const a=__timeAxes;a.setRange(0,a.sites[0].st.length-1);a.select(a.sites.find(s=>s.id===7),false);a.setRange(15,45);a.showStep('week',20);});
      await verify();return initial;
    } finally {await fresh.close();}
  });
  await reset();
  await mark('R3 S2-2 exactly one never-flaring pass empty state through hash, steps, ranges and Play',async()=>{
    await page.evaluate(()=>location.hash='#map?site=0');await page.waitForFunction(()=>__timeAxes.state.selected?.id===0);
    const verify=async()=>assert.equal(await page.locator('#detail .passbox > p').count(),1);
    await verify();await page.evaluate(()=>{const a=__timeAxes;a.showStep('night',a.state.range[1]);for(let i=0;i<8;i++)a.stepNight(-1);});await verify();
    await page.getByRole('button',{name:'Last 30',exact:true}).click();await verify();
    await page.locator('#play').click();await page.waitForTimeout(1700);await page.locator('#play').click();await verify();
    return {empty_states:await page.locator('#detail .passbox > p').count()};
  });
  await reset();
  await mark('R3 O2-4 deferred clearpasses never delay data-ready and update pending site facts',async()=>{
    const loading=await context.newPage();let release;const held=new Promise(r=>release=r),requests=[];
    await loading.route('**/data/clearpasses-*.json',async route=>{requests.push(await loading.evaluate(()=>({ready:document.documentElement.dataset.ready,time:performance.now()})));await held;await route.continue();});
    try {
      await loading.goto(`${origin}/${build}/?debug#map?site=7`);await loading.waitForFunction(()=>document.documentElement.dataset.ready==='true');
      await loading.waitForFunction(()=>document.querySelector('[data-sort="clear"]').textContent==='Loading views…');
      assert.equal(await loading.locator('[data-sort="clear"]').isDisabled(),true);
      assert.equal(await loading.locator('#detail .facts .num').nth(3).textContent(),'…');
      await loading.getByRole('button',{name:'Last 90',exact:true}).click();assert.equal(await loading.locator('#play').isDisabled(),false);
      release();await loading.waitForFunction(()=>document.documentElement.dataset.clearpasses==='true');
      assert.equal(await loading.locator('[data-sort="clear"]').isDisabled(),false);assert.notEqual(await loading.locator('#detail .facts .num').nth(3).textContent(),'…');
      assert.ok(requests.every(r=>r.ready==='true'));return {requests};
    } finally {release();await loading.close();}
  });
  await reset();
  await mark('R3 O2-8 both guarded 404 reloads reset; both Retry controls use fresh index file names',async()=>{
    const response=await context.request.get(`${origin}/${build}/`),html=await response.text(),results=[];
    for(const kind of ['nights','clearpasses']) {
      const recovery=await context.newPage();
      const first=await page.evaluate(kind=>{const m=JSON.parse(document.querySelector('#flaring-data').textContent).meta;return (kind==='nights'?m.chunks:m.clearpasses)[0];},kind);
      const stale=html.replace(first,`data/${kind}-2026.deadbeef.json`),key=(kind==='nights'?'night-chunk-reload:':'clearpass-chunk-reload:')+`/${build}/`;
      let fresh=false,refreshes=0;const cacheModes=[];
      await recovery.exposeFunction('captureIndexCache',cache=>cacheModes.push(cache));
      await recovery.addInitScript(()=>{const original=window.fetch;window.fetch=function(url,options){if(url===location.href.split('#')[0])window.captureIndexCache(options?.cache);return original.apply(this,arguments);};});
      await recovery.route(url=>url.pathname===`/${build}/`,async route=>{
        if(route.request().resourceType()==='fetch'){fresh=true;refreshes++;}
        return route.fulfill({status:200,contentType:'text/html',body:fresh?html:stale});
      });
      try {
        for(let round=0;round<2;round++) {
          fresh=false;await recovery.goto(`${origin}/${build}/?debug`);
          await recovery.waitForFunction(()=>document.documentElement.dataset.ready==='true'&&document.documentElement.dataset.clearpasses==='true');
          assert.equal(await recovery.evaluate(key=>sessionStorage.getItem(key),key),null);
        }
        assert.equal(refreshes,2);
        await recovery.evaluate(key=>sessionStorage.setItem(key,'1'),key);
        fresh=false;await recovery.goto(`${origin}/${build}/?debug`);
        const retry=recovery.getByRole('button',{name:kind==='nights'?'Retry':'Retry views',exact:true});await retry.waitFor();
        if(kind==='nights')assert.match(await recovery.getByRole('alert').textContent(),/deadbeef.*HTTP 404/);
        else assert.match(await retry.getAttribute('title'),/deadbeef.*HTTP 404/);
        await retry.click();
        await recovery.waitForFunction(()=>document.documentElement.dataset.ready==='true'&&document.documentElement.dataset.clearpasses==='true');
        assert.equal(refreshes,3);assert.deepEqual(cacheModes,['reload','reload','reload']);
        assert.equal(await recovery.evaluate(key=>sessionStorage.getItem(key),key),null);
        results.push({kind,refreshes,cacheModes});
      } finally {await recovery.close();}
    }
    return results;
  });
  await reset();
  await mark('R3 O2-5 clear tile and bin tooltip share F/C definition; six night counts match baseline',async()=>{
    const result=await page.evaluate(()=>{
      const a=__timeAxes,nights=JSON.parse(document.querySelector('#flaring-data').textContent).nights,results=[];
      for(const [unit,start,end] of [['week','2026-10-19','2026-10-25'],['quarter','2026-01-01','2026-03-31']]) {
        const lo=nights.indexOf(start),hi=nights.indexOf(end);a.setRange(lo,hi);a.showStep(unit,lo);
        const d=a.state.steps[0],expected=a.sites.filter(s=>/F|C/.test(s.st.slice(lo,hi+1))).length,network=a.networkBin(d).ns;
        const tile=+document.querySelectorAll('#tiles .num')[3].textContent.replaceAll(',','');
        if(expected!==tile||network!==tile)throw Error(`${unit}: expected ${expected}, tile ${tile}, bin ${network}`);results.push({unit,expected,tile,bin:network});
      }
      return results;
    });
    if(build==='site')assert.deepEqual(result.map(r=>r.tile),[7254,7619]);return result;
  });
  await reset();
  await mark('R3 O2-6 all stored single-night values and six displayed nights equal site_before',async()=>{
    if(build!=='site')return 'Real comparison only';
    const baseline=await context.newPage();await baseline.goto(`${origin}/site_before/`);
    try {
      const old=await baseline.evaluate(()=>JSON.parse(document.querySelector('#flaring-data').textContent).sites.map(r=>[r[0],r[5]]));
      const count=await page.evaluate(old=>{
        const a=__timeAxes,byId=new Map(old);let compared=0,mismatches=0;
        const fmt=v=>v==null?'–':v===0?'0':v>=100?v.toFixed(0):v>=10?v.toFixed(1):v>=1?v.toFixed(2):v.toFixed(3);
        for(const s of a.sites){let j=0;for(let i=0;i<s.st.length;i++)if('FD'.includes(s.st[i])){const value=byId.get(s.id)[j++];if(value!=null){compared++;if(fmt(a.rangeStats(s,i,i).mean)!==fmt(value))mismatches++;}}}
        return {compared,mismatches};
      },old);
      assert.equal(count.compared,40817);assert.equal(count.mismatches,0);
      const nights=[0,30,90,123,150,307],displays=[];
      await page.getByRole('button',{name:'All',exact:true}).click();
      for(const i of nights) {
        await page.evaluate(i=>__timeAxes.showStep('night',i),i);
        const box=await baseline.locator('#strip').boundingBox(),bar=await baseline.locator('#strip > rect').nth(i).boundingBox();await baseline.mouse.click(bar.x+bar.width/2,box.y+20);
        const now=await snapshot(page),before=await snapshot(baseline);
        assert.deepEqual(now.tiles,before.tiles);assert.deepEqual(now.contexts,before.contexts);assert.deepEqual(now.regions,before.regions);
        // Single-night row counts intentionally use step counts; compare names and values.
        const values=p=>p.locator('#rank li').evaluateAll(rows=>rows.slice(0,25).map(row=>[row.querySelector('.n b').textContent,row.querySelector('.val').textContent]));
        assert.deepEqual(await values(page),await values(baseline));displays.push(i);
      }
      return {...count,displayed_nights:displays};
    } finally {await baseline.close();}
  });
  await reset();
  await mark('R3 O2-10 year button focus survives Enter and next Tab',async()=>{
    const button=page.locator('#year-axis button').first(),year=await button.textContent();await button.focus();await page.keyboard.press('Enter');
    assert.equal(await page.evaluate(()=>document.activeElement.textContent),year);await page.keyboard.press('Tab');
    assert.notEqual(await page.evaluate(()=>document.activeElement.tagName),'BODY');return {year};
  });
  await reset();
  await mark('R3 O2-13 M2-1 one deliberate mode row; O2-14 calendar shifts; O2-15 markers; O2-16 labels',async()=>{
    const value=await page.evaluate(()=>{
      const a=__timeAxes,nights=JSON.parse(document.querySelector('#flaring-data').textContent).nights;
      const buttons=[...document.querySelectorAll('.modebar .seg button')].filter(b=>!b.hidden),tops=buttons.map(b=>b.getBoundingClientRect().top);
      if(new Set(tops).size!==1)throw Error('Mode buttons wrap: '+tops);
      const shift=(from,to,nextFrom,nextTo)=>{
        if(!nights.includes(from)||!nights.includes(nextTo))return null;
        a.setRange(nights.indexOf(from),nights.indexOf(to));document.querySelector('#range-next').click();
        const dates=a.state.range.map(i=>nights[i]);if(JSON.stringify(dates)!==JSON.stringify([nextFrom,nextTo]))throw Error('Calendar shift '+dates);return dates;
      };
      const shifts=[shift('2024-01-01','2024-12-31','2025-01-01','2025-12-31'),shift('2023-02-01','2023-02-28','2023-03-01','2023-03-31'),shift('2026-01-01','2026-03-31','2026-04-01','2026-06-30'),shift('2026-10-12','2026-10-18','2026-10-19','2026-10-25')].filter(Boolean);
      a.setRange(0,nights.length-1);a.select(a.sites.find(s=>s.everLit),false);
      for(const unit of ['night','week']) {a.showStep(unit,Math.floor(nights.length/2));for(const r of document.querySelectorAll('.step-marker,.range-shade'))if(getComputedStyle(r).display!=='none'&&+r.getAttribute('width')<2)throw Error('Marker below 2px');}
      a.showStep('week',nights.length-1);
      const clock=document.querySelector('#clock .d').textContent;if(clock!=='Week of Nov 2, 2026 · 3 of 7 nights (partial week)')throw Error('Partial label '+clock);
      if(!document.querySelector('#strip').getAttribute('aria-label').includes('Arrow keys'))throw Error('Missing arrow hint');
      const clear=a.sites.find(s=>s.stepStats.clear>0&&!s.stepStats.lit&&!s.stepStats.cloud);a.showTip(clear,10,10);
      if(document.querySelector('#tip .v').textContent!=='No flare')throw Error('Tooltip repeats the period in its status');
      const tipPeriod=document.querySelector('#tip .u').textContent;document.querySelector('#tip').hidden=true;
      return {buttons:buttons.map(b=>b.textContent),shifts,clock,tooltip_period:tipPeriod};
    });return value;
  });
  await reset();
  await mark('R3 S2-3 O2-11 dark date inputs, brush and count stay synchronized and clamp',async()=>{
    await page.getByRole('button',{name:'Flare-outs',exact:true}).click();await page.getByRole('button',{name:'Single dark nights',exact:true}).click();await page.waitForSelector('.dn-row');
    const from=page.getByLabel('From run start date',{exact:true}),to=page.getByLabel('To run start date',{exact:true});
    await page.getByRole('button',{name:'All dates',exact:true}).click();
    await from.fill('2026-07-14');await from.dispatchEvent('change');await to.fill('2026-07-14');await to.dispatchEvent('change');
    await page.getByLabel('With an EMIT scene',{exact:true}).uncheck();
    await page.evaluate(()=>location.hash='#flare-outs/dark-nights/4482_2026-07-14');await page.waitForFunction(()=>document.querySelector('.dn-detail')?.textContent.includes('4482'));
    await page.locator('.dn-detail a[href="#flare-outs/dark-nights/4481_2026-07-13"]').click();
    assert.equal(await from.inputValue(),'');assert.equal(await to.inputValue(),'');
    assert.match(await page.locator('#dn-content p[role="status"]').textContent(),/All dates/);assert.equal(await page.locator('.dn-histogram .selection').isVisible(),false);
    await from.fill('1900-01-01');await from.dispatchEvent('change');await to.fill('2099-03-01');await to.dispatchEvent('change');
    const record=await page.evaluate(()=>{const n=JSON.parse(document.querySelector('#flaring-data').textContent).nights;return [n[0],n.at(-1)];});
    assert.deepEqual([await from.inputValue(),await to.inputValue()],record);
    assert.ok(+(await page.locator('.dn-histogram .selection').getAttribute('width'))>=0);
    await page.getByRole('button',{name:'All dates',exact:true}).click();await page.getByRole('button',{name:'Map',exact:true}).click();return {clamped:record};
  });
  await reset();
  await mark('R3 O2-3 O2-12 touch scrolling over spell and histogram; distinguishable phone lanes',async()=>{
    const touch=await page.context().browser().newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    await touch.route('https://**/*',route=>{
      const url=route.request().url();const file=url.includes('d3.min.js')?'d3.min.js':url.includes('topojson-client.min.js')?'topojson-client.min.js':null;
      return file?route.fulfill({path:new URL('../../outputs/vendor/'+file,import.meta.url).pathname,contentType:'text/javascript'}):route.abort();
    });
    const phone=await touch.newPage(),results=[];
    try {
      await phone.goto(`${origin}/${build}/?debug`);await phone.waitForFunction(()=>document.documentElement.dataset.ready==='true');
      await phone.getByRole('button',{name:'Flare-outs',exact:true}).click();const cdp=await touch.newCDPSession(phone);
      for(const selector of ['.spell-timeline','.dn-histogram']) {
        if(selector.includes('histogram')){await phone.getByRole('button',{name:'Single dark nights',exact:true}).click();await phone.waitForSelector('.dn-row');}
        await phone.evaluate(selector=>{const r=document.querySelector(selector).getBoundingClientRect();scrollBy(0,r.top-250);},selector);await phone.waitForTimeout(150);
        const box=await phone.locator(selector).boundingBox(),before=await phone.evaluate(()=>scrollY);
        await cdp.send('Input.synthesizeScrollGesture',{x:Math.round(box.x+box.width*.7),y:Math.round(box.y+Math.min(50,box.height/2)),yDistance:-300,gestureSourceType:'touch',speed:800});await phone.waitForTimeout(200);
        const after=await phone.evaluate(()=>scrollY);assert.ok(after-before>100,selector+' did not scroll: '+before+' → '+after);results.push({selector,before,after});
        if(selector.includes('spell')) {
          const labels=await phone.locator('.spell-timeline > text').allTextContents();
          const lanes=labels.filter(t=>/^\d/.test(t));assert.equal(lanes.length,19);assert.ok(new Set(lanes).size>=18);
        }
      }
      return results;
    } finally {await touch.close();}
  });
  await page.setViewportSize({width:1440,height:1000});await page.getByRole('button',{name:'Map',exact:true}).click();await reset();
}
