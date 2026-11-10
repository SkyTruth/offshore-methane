/** Round 2 regression assertions. Uses the explicit ?debug inspection surface. */
import assert from 'node:assert/strict';

export async function checkRound2({page,context,origin,build,report,snapshot}) {
  const mark=async(name,fn)=>{try{report.parity[name]={pass:true,value:await fn()};}catch(e){report.parity[name]={pass:false,error:e.stack};}};
  const reset=()=>page.evaluate(()=>{const a=__timeAxes;a.pause();a.select(null,false);a.state.sort='strong';a.state.pct=0;a.state.flaringOnly=false;a.setRange(0,a.sites[0].st.length-1);});
  await reset();
  await mark('O1 tile context parity for All, Last 7/30/90 and one night',async()=>{
    if(build!=='site')return 'Real reference comparison runs on site';
    const before=await context.newPage();await before.goto(`${origin}/site_before/`);
    try {
      const values=[];
      for(const name of ['All','Last 7','Last 30','Last 90']) {
        await page.getByRole('button',{name,exact:true}).click();await before.getByRole('button',{name,exact:true}).click();
        const now=await snapshot(page),baseline=await snapshot(before),after=now.contexts,old=baseline.contexts;assert.deepEqual(after,old);assert.deepEqual(now.tiles,baseline.tiles);values.push({range:name,contexts:after});
      }
      await page.getByRole('button',{name:'All',exact:true}).click();await before.getByRole('button',{name:'All',exact:true}).click();
      const i=await page.evaluate(()=>{const a=__timeAxes,i=150;a.showStep('night',i);return i;});
      const box=await before.locator('#strip').boundingBox(),bar=await before.locator('#strip > rect').nth(i).boundingBox();await before.mouse.click(bar.x+bar.width/2,box.y+20);
      assert.deepEqual((await snapshot(page)).contexts,(await snapshot(before)).contexts);assert.deepEqual((await snapshot(page)).tiles,(await snapshot(before)).tiles);return values;
    } finally {await before.close();}
  });
  await reset();
  await mark('S1 O4 one night scale for every bar and step unit',async()=>{
    return page.evaluate(()=>{
      const a=__timeAxes,n=a.sites[0].st.length,results=[];
      a.select(a.sites.find(s=>s.rFlare>0),false);
      for(const unit of ['night','week','month','quarter','year']) {
        const bins=a.calendarBins(unit);
        for(const d of [bins[0],bins.at(-1)]) {
          a.showStep(unit,d.a);a.showTip(a.state.selected,10,10);
          const roots=[document.querySelector('#strip'),document.querySelector('#record-overview'),...document.querySelectorAll('#rank .spark svg,#regions tbody svg,#detail .overview,#tip svg')];
          let checked=0,maxError=0;
          for(const root of roots){
            const marker=root.querySelector('.step-marker');if(!marker||getComputedStyle(marker).display==='none')continue;
            const width=root.viewBox.baseVal.width,pad=root.closest('.spark,#tip')?1:root.closest('#detail')?2:0;
            const extent=(d.b-d.a+1)/n*(width-2*pad),minimum=Math.max(2,extent);
            const expected=[pad+(d.a+(d.b-d.a+1)/2)/n*(width-2*pad)-minimum/2,minimum];
            const actual=[+marker.getAttribute('x'),+marker.getAttribute('width')];
            const error=Math.max(...actual.map((v,i)=>Math.abs(v-expected[i])));maxError=Math.max(maxError,error);
            if(error>.5)throw Error(`${unit} ${d.a} ${root.outerHTML.slice(0,100)} marker error ${error}`);
            const bar=root.querySelector(`.time-bar[data-a="${d.a}"][data-b="${d.b}"]`);
            if(bar&&extent>=2)for(const attr of ['x','width'])if(Math.abs(+bar.getAttribute(attr)-+marker.getAttribute(attr))>.5)throw Error('bar/marker extent mismatch');
            // Every bar's width and start independently agree with its nights.
            for(const bar of root.querySelectorAll('.time-bar')) {
              const lo=+bar.dataset.a,hi=+bar.dataset.b;
              if(Math.abs(+bar.getAttribute('x')-(pad+lo/n*(width-2*pad)))>.5||Math.abs(+bar.getAttribute('width')-(hi-lo+1)/n*(width-2*pad))>.5)throw Error('bar is not on linear night scale');
            }
            checked++;
          }
          results.push({unit,edge:d.a===bins[0].a?'first':'last',charts:checked,max_error_px:maxError});
        }
      }
      return results;
    });
  });
  await reset();
  await mark('S2 O21 O23 fragment navigation, missing site and clamped dates',async()=>{
    const expected=await page.evaluate(()=>{const a=__timeAxes,lo=a.sites[0].st.length-90,site=a.sites.find(s=>s.rFlare>0).id;return {lo,site};});
    await page.evaluate(({lo,site})=>{const n=__timeAxes.sites[0].st.length;location.hash=`#map?from=${document.querySelector('#flaring-data')?JSON.parse(document.querySelector('#flaring-data').textContent).nights[lo]:''}&to=2099-12-31&step=month&at=2099-12-31&site=${site}`;},expected);
    await page.waitForFunction(site=>__timeAxes.state.selected?.id===site,expected.site);
    assert.equal(await page.evaluate(()=>__timeAxes.state.range[0]),expected.lo);
    assert.equal(await page.evaluate(()=>__timeAxes.state.range[1]),report.record.nights-1);
    await page.evaluate(()=>location.hash='#map?from=1900-01-01&to=2099-12-31&step=week&at=1900-01-01');
    await page.waitForFunction(()=>__timeAxes.state.selected===null&&__timeAxes.state.unit==='week');
    assert.deepEqual(await page.evaluate(()=>__timeAxes.state.range),[0,report.record.nights-1]);
    assert.equal(await page.evaluate(()=>__timeAxes.state.night),0);
  });
  await reset();
  await mark('O24 JS calendar bins equal independent date grouping including leap years',async()=>{
    return page.evaluate(()=>{
      const a=__timeAxes,dates=JSON.parse(document.querySelector('#flaring-data').textContent).nights,results={};
      for(const unit of ['night','week','month','quarter','year']) {
        const keys=dates.map(iso=>{const d=new Date(iso+'T00:00:00Z');return unit==='night'?iso:unit==='year'?iso.slice(0,4):unit==='month'?iso.slice(0,7):unit==='quarter'?iso.slice(0,4)+'Q'+Math.floor(d.getUTCMonth()/3):Math.floor((+d+3*864e5)/(7*864e5));});
        const expected=[];keys.forEach((key,i)=>{if(i===0||key!==keys[i-1])expected.push({a:i,b:i});else expected.at(-1).b=i;});
        const actual=a.calendarBins(unit).map(({a,b})=>({a,b}));if(JSON.stringify(actual)!==JSON.stringify(expected))throw Error(unit+' differs from independent date groups');
        if(a.totals[unit].network.length!==expected.length)throw Error('embedded total count differs');results[unit]=expected.length;
      }
      results.leap_days=dates.filter(d=>d.endsWith('02-29')).length;return results;
    });
  });
  await mark('S3 O3 M2 one unit control and complete step rows',async()=>{
    assert.equal(await page.locator('#step-unit').count(),0);
    return page.evaluate(()=>{
      const a=__timeAxes,units=['night','week','month','quarter','year'],results=[];
      const mw=v=>v==null?'–':v===0?'0':v>=100?v.toFixed(0):v>=10?v.toFixed(1):v>=1?v.toFixed(2):v.toFixed(3);
      const fmt=n=>n.toLocaleString('en-US'),nights=n=>fmt(n)+(n===1?' night':' nights');
      const compare={strong:(x,y)=>y.stepStats.mean-x.stepStats.mean,consistent:(x,y)=>y.rLo-x.rLo||y.rShare-x.rShare||y.rClear-x.rClear||y.rMean-x.rMean,clear:(x,y)=>y.rPasses-x.rPasses||y.rClear-x.rClear||y.rMean-x.rMean};
      for(const unit of units){
        const bin=a.calendarBins(unit).find(d=>!d.partial&&d.a>0)||a.calendarBins(unit)[0];a.showStep(unit,bin.a);
        if(document.querySelector(`#mode-${unit}`).getAttribute('aria-pressed')!=='true')throw Error('unit button not selected '+unit);
        if(!document.querySelector('#legend-keys .lt').textContent.toLowerCase().includes(unit==='night'?'on':'mean for'))throw Error('legend period missing');
        for(const sort of ['strong','consistent','clear']) {
          document.querySelector(`[data-sort="${sort}"]`).click();
          const expected=a.sites.filter(s=>s.stepStats.lit>0).sort(compare[sort]);
          const rows=[...document.querySelectorAll('#rank button')];
          rows.forEach((row,i)=>{
            const s=expected[i],sub=row.querySelector('.n > span').textContent,val=row.querySelector('.val').textContent;
            if(!sub.startsWith('site '+s.id+' · '))throw Error(`${unit} ${sort} wrong site/order ${i}`);
            if(sort==='strong'&&(!val.startsWith(mw(s.stepStats.mean))||!sub.endsWith(`${fmt(s.stepStats.lit)}/${nights(s.stepStats.clear)}`)))throw Error('strong row does not use step');
            if(sort==='consistent'&&(val!==`${Math.round(100*s.rShare)}%${fmt(s.rFlare)} of ${nights(s.rClear)}`||!sub.endsWith(`mean ${mw(s.rMean)} MW`)))throw Error('consistent row does not use range');
            if(sort==='clear'&&(val!==`${fmt(s.rPasses)}${nights(s.rClear)}`||!sub.endsWith(`flaring ${fmt(s.rFlare)}/${fmt(s.rClear)}`)))throw Error('clear row does not use range');
          });
          const hint=document.querySelector('#rank-hint').textContent;if(sort!=='strong'&&!hint.includes(new Date(JSON.parse(document.querySelector('#flaring-data').textContent).nights[0]).getUTCFullYear()))throw Error('hint missing range period');
          results.push({unit,sort,rows:rows.length,hint});
        }
      }
      return results;
    });
  });
  await reset();
  await mark('O5 pass windows and complete step outlines',async()=>{
    await page.locator('#rank button').first().click();await page.waitForSelector('#detail .passbox svg');
    return page.evaluate(()=>{
      const a=__timeAxes,n=a.sites[0].st.length,results=[];
      for(const unit of ['night','week','month','quarter','year']) {
        const d=a.calendarBins(unit).at(-1);a.showStep(unit,d.a);
        const win=a.state.win,len=d.b-d.a+1,expected=unit==='night'?14:len<=21?len:14;
        if(win[1]-win[0]+1!==expected)throw Error(`${unit} wrong window ${win}`);
        const outlines=[...document.querySelectorAll('.pass-step')].map(e=>+e.dataset.night),overlap=[];
        for(let i=Math.max(d.a,win[0]);i<=Math.min(d.b,win[1]);i++)overlap.push(i);
        if(JSON.stringify(outlines)!==JSON.stringify(overlap))throw Error(`${unit} missing outlined nights`);
        results.push({unit,win,outlines:outlines.length});
      }
      return results;
    });
  });
  await mark('S4 O8 O18 calendar roving cursor, month alignment and stable focus',async()=>{
    const cursor=page.locator('.calendar [tabindex="0"]');assert.equal(await cursor.count(),1);await cursor.focus();
    let i=await cursor.getAttribute('data-night');const mapNight=await page.evaluate(()=>__timeAxes.state.night);
    const box0=await cursor.boundingBox();await page.keyboard.press('ArrowLeft');assert.equal(+(await cursor.getAttribute('data-night')),+i-7);const box1=await page.locator('.calendar [role=gridcell][tabindex="0"]').boundingBox();assert.ok(box1.x<box0.x-1&&Math.abs(box1.y-box0.y)<1,'ArrowLeft moves one column left');
    await page.keyboard.press('ArrowUp');assert.equal(+(await cursor.getAttribute('data-night')),+i-8);const box2=await page.locator('.calendar [role=gridcell][tabindex="0"]').boundingBox();assert.ok(box2.y<box1.y-1&&Math.abs(box2.x-box1.x)<1,'ArrowUp moves one row up');
    await page.keyboard.press('PageUp');i=+(await cursor.getAttribute('data-night'));
    await page.keyboard.press('Enter');assert.equal(await page.evaluate(()=>document.activeElement.dataset.night),String(i));
    assert.equal(await page.evaluate(()=>__timeAxes.state.night),mapNight);
    await page.keyboard.press('Space');await page.keyboard.press('PageDown');await page.keyboard.press('ArrowDown');await page.keyboard.press('ArrowRight');
    assert.equal(await cursor.count(),1);
    await page.evaluate(()=>__timeAxes.refresh());assert.equal(await page.evaluate(()=>document.activeElement.getAttribute('role')),'gridcell');
    return page.evaluate(()=>[...document.querySelectorAll('.calendar text[data-month]')].every(t=>+t.getAttribute('x')>=0));
  });
  await reset();
  await mark('O2 click inside step selection, first drag and live focus selection O6',async()=>{
    const box=await page.locator('#record-overview').boundingBox();
    await page.mouse.move(box.x+box.width*.2,box.y+10);await page.mouse.down();await page.mouse.move(box.x+box.width*.5,box.y+10,{steps:5});await page.mouse.up();
    const range=await page.evaluate(()=>__timeAxes.state.range);assert.notDeepEqual(range,[0,report.record.nights-1]);
    await page.evaluate(()=>__timeAxes.showStep('week',__timeAxes.state.range[0]));
    const selection=await page.locator('#record-overview .selection').boundingBox();await page.mouse.click(selection.x+selection.width/2,selection.y+10);
    assert.deepEqual(await page.evaluate(()=>__timeAxes.state.range),range);assert.ok(await page.evaluate(()=>__timeAxes.state.night>=0));
    const focus=await page.locator('#strip').boundingBox();await page.mouse.move(focus.x+focus.width*.2,focus.y+20);await page.mouse.down();await page.mouse.move(focus.x+focus.width*.5,focus.y+20,{steps:4});
    assert.equal(await page.locator('.focus-drag').getAttribute('visibility'),'visible');assert.equal(await page.evaluate(()=>getSelection().toString()),'');await page.mouse.up();
  });
  await reset();
  await mark('O7 O11 playback with/without open site and all years',async()=>{
    const result={};
    for(const open of [false,true]) {
      await reset();if(open){await page.locator('#rank button').first().click();await page.waitForSelector('#detail .passbox svg');const more=page.getByRole('button',{name:/Show.*earlier years/});if(await more.count())await more.click();}
      result[open?'site_open':'site_closed']=await page.evaluate(async open=>{
        const a=__timeAxes,range=[],paint=[];
        for(const count of [30,90,a.sites[0].st.length,7,a.sites[0].st.length]){const t=performance.now();a.setRange(a.sites[0].st.length-count,a.sites[0].st.length-1);range.push(performance.now()-t);await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));paint.push(performance.now()-t);}
        a.showStep('month',a.state.range[0]);a.metrics.frames=[];a.metrics.frameWork=[];
        const calendar=document.querySelector('.calendar'),record=document.querySelector('#detail .overview'),pass=document.querySelector('#detail .passbox svg');
        const old=history.replaceState;let writes=0;history.replaceState=function(...args){writes++;return old.apply(this,args);};a.play();writes=0;
        await new Promise(resolve=>setTimeout(resolve,2500));
        const writesWhilePlaying=writes,stable=!open||(calendar===document.querySelector('.calendar')&&record===document.querySelector('#detail .overview')&&pass===document.querySelector('#detail .passbox svg'));
        a.pause();const pauseWrites=writes-writesWhilePlaying;history.replaceState=old;
        const percentile=(values,p)=>[...values].sort((x,y)=>x-y)[Math.floor((values.length-1)*p)];
        return {range_ms:range,range_max_ms:Math.max(...range),range_through_paint_ms:paint,range_through_paint_max_ms:Math.max(...paint),median_ms:percentile(a.metrics.frames,.5),p95_ms:percentile(a.metrics.frames,.95),frame_work_p95_ms:percentile(a.metrics.frameWork,.95),writesWhilePlaying,pauseWrites,stable,calendar_cells:calendar?.querySelectorAll('[data-night]').length||0};
      },open);
      const value=result[open?'site_open':'site_closed'];if(open)assert.equal(value.calendar_cells,report.record.nights);assert.equal(value.writesWhilePlaying,0);assert.ok(value.pauseWrites>0);assert.equal(value.stable,true);
      if(build==='site_long_5000'){assert.ok(value.range_through_paint_max_ms<=150,`range through paint ${value.range_through_paint_max_ms}`);assert.ok(value.median_ms<=42);assert.ok(value.p95_ms<=67,`p95 ${value.p95_ms}`);}
    }
    report.budgets.round2_playback=result;return result;
  });
  await reset();
  await mark('O17 Esc closes one layer per press',async()=>{
    await reset();await page.evaluate(()=>__timeAxes.showStep('week',__timeAxes.state.range[1]));await page.locator('#rank button').first().click();
    await page.getByRole('button',{name:'About this order',exact:true}).click();assert.equal(await page.locator('#infotip').isVisible(),true);await page.keyboard.press('Escape');
    assert.equal(await page.locator('#infotip').isHidden(),true);assert.ok(await page.evaluate(()=>__timeAxes.state.selected));assert.ok(await page.evaluate(()=>__timeAxes.state.night>=0));
    await page.locator('#strip').focus();await page.keyboard.press('Escape');assert.equal(await page.evaluate(()=>__timeAxes.state.selected),null);assert.ok(await page.evaluate(()=>__timeAxes.state.night>=0));
    await page.keyboard.press('Escape');assert.equal(await page.evaluate(()=>__timeAxes.state.night),-1);
  });
  await mark('O10 O22 O23 loading placeholders, disabled controls, 404 reload and Retry',async()=>{
    const failure=await context.newPage();let requests=0,retries=0;
    await failure.route('**/data/nights-*.json',async route=>{if(!route.request().url().includes((await page.evaluate(()=>JSON.parse(document.querySelector('#flaring-data').textContent).meta.chunks[0])).split('/').at(-1)))return route.continue();requests++;await new Promise(r=>setTimeout(r,350));if(retries)return route.continue();return route.fulfill({status:404,body:'missing'});});
    try {
      await failure.goto(`${origin}/${build}/?debug`);assert.match(await failure.locator('#tiles').textContent(),/Loading/);assert.equal(await failure.locator('#play').isDisabled(),true);assert.equal(await failure.locator('#year-select').isDisabled(),true);assert.equal(await failure.locator('#year-axis button:not(:disabled)').count(),0);
      await failure.getByRole('alert').waitFor();const text=await failure.getByRole('alert').textContent();assert.match(text,/data\/nights-.*HTTP 404/);assert.doesNotMatch(text,/Serve this page over HTTP/);assert.equal(requests,2);
      retries++;await failure.getByRole('button',{name:'Retry',exact:true}).click();await failure.waitForFunction(()=>document.documentElement.dataset.ready==='true');assert.equal(await failure.evaluate(()=>sessionStorage.getItem('night-chunk-reload:'+location.pathname)),null);assert.equal(await failure.locator('#play').isDisabled(),false);return {requests,error:text};
    } finally {await failure.close();}
  });
  await mark('O22 production debug surface absent',async()=>{const prod=await context.newPage();try{await prod.goto(`${origin}/${build}/`);await prod.waitForFunction(()=>document.documentElement.dataset.ready==='true');assert.equal(await prod.evaluate(()=>typeof window.__timeAxes),'undefined');}finally{await prod.close();}});
  await mark('S5 O12 O15 O16 M1 M3 M4 M8 M9 readable responsive charts',async()=>{
    const results=[];
    for(const width of [1440,1024,390]) {
      await page.setViewportSize({width,height:1000});await reset();
      const map=await page.evaluate(()=>{
        const checkText=root=>{const box=root.getBoundingClientRect();return [...root.querySelectorAll('text')].map(t=>{const r=t.getBoundingClientRect(),size=+t.getAttribute('font-size')*box.width/root.viewBox.baseVal.width;if(size<9.99)throw Error('text below 10 CSS px '+size);if(r.left<box.left-.5||r.right>box.right+.5)throw Error('text outside chart '+t.textContent);return size;});};
        checkText(document.querySelector('#strip'));
        const strip=document.querySelector('#strip'),unit=strip.getAttribute('aria-label').match(/1 bar = 1 (\w+)/)[1];
        const ticks=[...strip.querySelectorAll('text[data-date]')],tickYears=new Set();let right=-Infinity;
        for(const t of ticks){const date=new Date(t.dataset.date),r=t.getBoundingClientRect();if(r.left<right)throw Error('axis labels collide');right=r.right;if(unit==='week'&&date.getUTCDate()!==1)throw Error('weekly ticks not on month boundaries');if(['month','quarter','year'].includes(unit)&&date.getUTCDate()!==1)throw Error('coarse ticks not on calendar boundaries');const year=date.getUTCFullYear();if(!tickYears.has(year)&&!t.textContent.includes(year))throw Error('First kept tick without year');tickYears.add(year);}

        const yearButtons=[...document.querySelectorAll('#year-axis button')],data=JSON.parse(document.querySelector('#flaring-data').textContent);if(yearButtons.at(-1).textContent!==data.nights.at(-1).slice(0,4))throw Error('newest year missing');const labelled=yearButtons.map(b=>+b.textContent),steps=new Set(labelled.slice(1).map((y,i)=>y-labelled[i]));if(steps.size>1)throw Error('uneven year stride '+[...steps]);const menu=new Set([...document.querySelectorAll('#timeline select option')].map(o=>o.value));for(const y of new Set(data.nights.map(n=>n.slice(0,4))))if(!menu.has(y))throw Error('year '+y+' not in the Year menu');const years=yearButtons.map(b=>b.getBoundingClientRect());for(let i=0;i<years.length;i++){if(years[i].width<24||years[i].height<24)throw Error('small year button');if(i&&years[i].left<years[i-1].right)throw Error('overlapping year buttons');}
        const over=document.querySelector('#record-overview').getAttribute('aria-label').match(/1 bar = 1 (\w+)/)[1],focus=document.querySelector('#strip').getAttribute('aria-label').match(/1 bar = 1 (\w+)/)[1],ladder=['night','week','month','quarter','year'];if(ladder.indexOf(focus)>ladder.indexOf(over))throw Error('focus coarser than overview');
        const full=[...document.querySelectorAll('#record-overview .time-bar')].find(b=>+b.dataset.b-+b.dataset.a>100);if(full&&+full.getAttribute('width')<3)throw Error('overview unit too fine');
        return {overview:over,focus,years:years.length};
      });
      await page.getByRole('button',{name:'Flare-outs',exact:true}).click();await page.getByRole('button',{name:'Dark spells',exact:true}).click();
      await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
      const spell=await page.evaluate(()=>{for(const chart of document.querySelectorAll('#fo-spells .fo-plot svg')){const box=chart.getBoundingClientRect();for(const t of chart.querySelectorAll('text'))if(+t.getAttribute('font-size')*box.width/chart.viewBox.baseVal.width<9.99)throw Error('evidence chart font too small');}const svg=document.querySelector('.spell-timeline'),box=svg.getBoundingClientRect(),scale=box.width/svg.viewBox.baseVal.width;for(const t of svg.querySelectorAll('text')){if(+t.getAttribute('font-size')*scale<9.99)throw Error('spell font too small');const r=t.getBoundingClientRect();if(r.left<box.left-.5||r.right>box.right+.5)throw Error('spell text out of bounds');}const bars=[...svg.querySelectorAll('rect')];if(bars.some(b=>+b.getAttribute('height')*scale>14))throw Error('spell lane too tall');return {width:box.width,viewbox:svg.viewBox.baseVal.width,height:box.height};});
      await page.getByRole('button',{name:'Single dark nights',exact:true}).click();await page.waitForSelector('.dn-histogram');
      await page.getByLabel('From run start date',{exact:true}).fill('2026-03-01');await page.getByLabel('From run start date',{exact:true}).dispatchEvent('change');await page.getByLabel('To run start date',{exact:true}).fill('2026-03-31');await page.getByLabel('To run start date',{exact:true}).dispatchEvent('change');
      assert.match(await page.locator('#dn-content p[role="status"]').textContent(),/Run starts/);
      const hist=await page.evaluate(()=>{const svg=document.querySelector('.dn-histogram'),box=svg.getBoundingClientRect();for(const t of svg.querySelectorAll('text')){if(+t.getAttribute('font-size')*box.width/svg.viewBox.baseVal.width<9.99)throw Error('hist font too small');const r=t.getBoundingClientRect();if(r.left<box.left-.5||r.right>box.right+.5)throw Error('hist text out of bounds');}if(!svg.querySelector('.count-scale')||![...svg.querySelectorAll('.hist-bar title')].every(t=>/runs/.test(t.textContent)))throw Error('hist counts missing');return {width:box.width,viewbox:svg.viewBox.baseVal.width};});
      results.push({width,map,spell,hist});await page.getByRole('button',{name:'Map',exact:true}).click();
    }
    return results;
  });
  await page.setViewportSize({width:1440,height:1000});await reset();
}
