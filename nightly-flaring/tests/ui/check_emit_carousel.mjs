// D67 checks: EMIT carousel, ↑ ↓ / ← → keys and separate list/detail panes in
// both Flare-outs modes. Not collected by pytest. Run from nightly-flaring/:
//   NODE_PATH=~/git/skinner/node_modules node tests/ui/check_emit_carousel.mjs [build]
// The build defaults to outputs/site; CDN scripts are routed to outputs/vendor.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const outputs = path.resolve("outputs"), siteDir = path.join(outputs, process.argv[2] || "site");
const vendor = path.join(outputs, "vendor");
const types = { ".html": "text/html", ".json": "application/json", ".png": "image/png" };
const server = http.createServer((req, res) => {
  const p = path.join(siteDir, decodeURIComponent(req.url.split("?")[0]).replace(/\/$/, "/index.html"));
  fs.readFile(p, (err, buf) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "content-type": types[path.extname(p)] || "application/octet-stream" }); res.end(buf);
  });
}).listen(0);
const browser = await chromium.launch({ channel: "chrome" });
const settle = page => page.waitForTimeout(700);

async function open(width, height, hash) {
  const page = await browser.newPage({ viewport: { width, height } });
  await page.route(/cdnjs\.cloudflare\.com\/.*d3\.min\.js/, r => r.fulfill({ path: path.join(vendor, "d3.min.js") }));
  await page.route(/jsdelivr\.net\/npm\/topojson-client/, r => r.fulfill({ path: path.join(vendor, "topojson-client.min.js") }));
  page.errors = []; page.on("pageerror", e => page.errors.push(String(e)));
  await page.goto(`http://localhost:${server.address().port}/index.html${hash}`);
  await page.waitForTimeout(2500);
  return page;
}
const state = (page, root) => page.evaluate(root => {
  const box = document.querySelector(root), car = box.querySelector(".emit-carousel"), ws = box.querySelector(".fo-workspace");
  const a = document.getElementById("flareouts"), emit = box.querySelector(".emit-section");
  return {
    hash: location.hash,
    selected: box.querySelector('.fo-row[aria-current="true"]')?.dataset.id,
    counter: car?.querySelector(".emit-nav > span")?.textContent,
    images: car ? [...car.querySelectorAll("img")].map(i => i.complete && i.naturalWidth > 0) : [],
    nulls: car ? /\bnull\b/.test(car.textContent) : false,
    article: Math.round(a.scrollTop), list: Math.round(ws.children[0].scrollTop), detail: Math.round(ws.children[1].scrollTop),
    wsTop: Math.round(ws.getBoundingClientRect().top - a.getBoundingClientRect().top),
    emitTop: emit ? Math.round(emit.getBoundingClientRect().top - ws.children[1].getBoundingClientRect().top) : null,
    rows: box.querySelectorAll(".fo-row").length,
  };
}, root);
const toEmit = (page, root) => page.locator(root + " .fo-workspace > :nth-child(2)").evaluate(p => {
  const e = p.querySelector(".emit-section"); p.scrollTop += e.getBoundingClientRect().top - p.getBoundingClientRect().top - 40;
});
const results = {};

for (const [mode, hash, root] of [["spells", "#flare-outs", "#fo-spells"], ["darknights", "#flare-outs/dark-nights", "#dn-content"]]) {
  const page = await open(1440, 900, hash), r = {};
  // Panes: the tab scrolls first, then each pane on its own.
  await page.evaluate(root => { const a = document.getElementById("flareouts"), ws = document.querySelector(root + " .fo-workspace"); a.scrollTop += ws.getBoundingClientRect().top - a.getBoundingClientRect().top - 450; }, root);
  await settle(page);
  const list = await page.locator(root + " .fo-workspace > :first-child").boundingBox();
  const s0 = await state(page, root);
  await page.mouse.move(list.x + 60, Math.min(850, list.y + 150)); await page.mouse.wheel(0, 200); await settle(page);
  const s1 = await state(page, root);
  assert.equal(s1.article - s0.article, 200, mode + ": the tab scrolls before the workspace fills it");
  assert.equal(s1.list, 0, mode + ": the list waits until the workspace fills the tab");
  for (let i = 0; i < 4; i++) { await page.mouse.wheel(0, 200); await page.waitForTimeout(400); }
  await settle(page);
  const s2 = await state(page, root);
  assert.ok(s2.wsTop >= 10 && s2.wsTop <= 16, mode + ": the workspace fills the tab with its top in view (" + s2.wsTop + ")");
  assert.ok(s2.list > 0 && s2.detail === 0, mode + ": the list scrolls without the detail");
  const det = await page.locator(root + " .fo-workspace > :nth-child(2)").boundingBox();
  await page.mouse.move(det.x + 200, det.y + 300); await page.mouse.wheel(0, 300); await settle(page);
  const s3 = await state(page, root);
  assert.equal(s3.list, s2.list, mode + ": the detail scrolls without the list");
  assert.ok(s3.detail > 0, mode + ": the detail scrolls");
  // One upward gesture (events < 250 ms apart) that starts mid-list stops at
  // the list's top; a new gesture from the top returns the tab to the overview.
  await page.mouse.move(list.x + 60, 500);
  for (let i = 0; i < 30; i++) { await page.mouse.wheel(0, -400); await page.waitForTimeout(50); }
  await settle(page);
  const s4 = await state(page, root);
  assert.ok(s4.list === 0 && s4.article === s3.article && s4.detail === s3.detail, mode + ": a gesture that scrolled the list stops at its top");
  for (let i = 0; i < 10; i++) { await page.mouse.wheel(0, -400); await page.waitForTimeout(50); }
  await settle(page);
  const s5 = await state(page, root);
  assert.ok(s5.article === 0 && s5.detail === s3.detail, mode + ": a new upward gesture from the list's top returns to the overview");
  r.panes = { s0, s1, s2, s3, s4, s5 };

  // Carousel and keys. Dark spells: card 4080 has three scenes with images.
  if (mode === "spells") await page.locator(root + " .fo-row", { hasText: "4080" }).first().click();
  await settle(page);
  const c0 = await state(page, root);
  assert.ok(c0.counter, mode + ": carousel shown");
  assert.ok(!c0.nulls, mode + ": no stray null text");
  if (c0.counter.includes(" of ")) {
    const atEnd = await page.locator(root + ' .emit-carousel button[aria-label="Next EMIT scene"]').isDisabled();
    const [go, back] = atEnd ? ["ArrowLeft", "ArrowRight"] : ["ArrowRight", "ArrowLeft"];
    await page.keyboard.press(go); await settle(page);
    const c1 = await state(page, root);
    assert.notEqual(c1.counter, c0.counter, mode + ": " + go + " moves the carousel");
    await page.keyboard.press(back); await settle(page);
    assert.equal((await state(page, root)).counter, c0.counter, mode + ": " + back + " moves back");
  }
  if (c0.images.length) assert.ok(c0.images.every(Boolean), mode + ": the image pair loads");
  // ↓ keeps the EMIT section where it was in the detail pane.
  await toEmit(page, root); await settle(page);
  const k0 = await state(page, root);
  await page.keyboard.press("ArrowDown"); await settle(page);
  const k1 = await state(page, root);
  assert.notEqual(k1.selected, k0.selected, mode + ": ↓ selects the next row");
  assert.notEqual(k1.hash, k0.hash, mode + ": ↓ updates the address");
  assert.ok(Math.abs(k1.emitTop - k0.emitTop) <= 3, mode + ": EMIT stays in place (" + k0.emitTop + " → " + k1.emitTop + ")");
  await page.keyboard.press("ArrowUp"); await settle(page);
  assert.equal((await state(page, root)).selected, k0.selected, mode + ": ↑ returns");
  r.keys = { c0, k0, k1 };
  if (mode === "darknights") {
    const sel = page.locator(root + " select").first();
    await sel.focus(); const before = (await state(page, root)).selected;
    await page.keyboard.press("ArrowDown"); await settle(page);
    assert.equal((await state(page, root)).selected, before, "a focused select keeps its arrow keys");
    await page.locator(root + " .fo-row").first().focus();
    for (let i = 0; i < 60; i++) await page.keyboard.press("ArrowDown");
    await settle(page);
    const far = await state(page, root);
    assert.ok(far.rows > 50, "↓ past the 50th row loads more rows (" + far.rows + ")");
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute("aria-current")), "true", "focus follows the selected row");
    r.far = far;
  }
  assert.deepEqual(page.errors, [], mode + ": no browser errors");
  results[mode] = r;
  await page.close();
}
// Dark spells: 4080 opens on its scene during the spell; the EMIT anchor
// returns to its wanted offset after a short detail (4102 has no scene).
{
  const page = await open(1440, 900, "#flare-outs"), root = "#fo-spells";
  await page.locator(root + " .fo-row", { hasText: "4080" }).first().click(); await settle(page);
  assert.match(await page.locator(root + " .emit-carousel h4").textContent(), /^2026-08-24/, "4080 opens on its scene during the spell");
  assert.ok((await page.locator(root + " .emit-strip button").allTextContents()).every(s => / · (before|during)$/.test(s)), "strip dates carry their timing");
  const anchorTop = () => page.evaluate(r => { const p = document.querySelector(r + " .fo-workspace > :nth-child(2)"), a = p.querySelector(".emit-carousel") || p.querySelector(".emit-section"); return Math.round(a.getBoundingClientRect().top - p.getBoundingClientRect().top); }, root);
  await page.locator(root + " .fo-workspace > :nth-child(2)").evaluate(p => { const a = p.querySelector(".emit-carousel"); p.scrollTop += a.getBoundingClientRect().top - p.getBoundingClientRect().top - 40; });
  await settle(page); const t0 = await anchorTop();
  await page.keyboard.press("ArrowUp"); await settle(page);   // 4102: no scenes, short detail
  await page.keyboard.press("ArrowDown"); await settle(page); // back to 4080
  assert.ok(Math.abs((await anchorTop()) - t0) <= 3, "the anchor returns to its wanted offset after a short detail");
  assert.deepEqual(page.errors, [], "anchor memory: no browser errors");
  await page.close();
}
// 800 px (page scrolls as one, two columns): the anchor holds after charts redraw.
for (const [hash, root] of [["#flare-outs", "#fo-spells"], ["#flare-outs/dark-nights", "#dn-content"]]) {
  const page = await open(800, 900, hash);
  if (root === "#fo-spells") await page.locator(root + " .fo-row", { hasText: "4080" }).first().click();
  await settle(page);
  await page.evaluate(r => { document.activeElement?.blur(); const a = document.querySelector(r + " .emit-carousel") || document.querySelector(r + " .emit-section"); window.scrollBy(0, a.getBoundingClientRect().top - 60); }, root);
  await settle(page);
  const tops = [];
  for (let i = 0; i < 4; i++) { await page.keyboard.press("ArrowDown"); await page.waitForTimeout(900);
    tops.push(await page.evaluate(r => Math.round((document.querySelector(r + " .emit-carousel") || document.querySelector(r + " .emit-section")).getBoundingClientRect().top), root)); }
  assert.ok(tops.every(v => Math.abs(v - 60) <= 4), root + ": 800 px anchor holds (" + tops + ")");
  // A list-focused ↓ owns the window scroll: the focused row stays in view
  // after the charts redraw (no deferred imagery placement).
  const rows = page.locator(root + " .fo-row");
  const visible = await rows.evaluateAll(rs => rs.findIndex(r => { const b = r.getBoundingClientRect(); return b.top > 0 && b.bottom < innerHeight; }));
  await rows.nth(Math.max(0, visible)).focus();
  const rowBox = () => page.evaluate(() => { const b = document.activeElement.getBoundingClientRect(); return [Math.round(b.top), Math.round(b.bottom), innerHeight, document.activeElement.classList.contains("fo-row")]; });
  for (const [key, n] of [["ArrowDown", 6], ["ArrowUp", 8]]) for (let i = 0; i < n; i++) {
    await page.keyboard.press(key); await page.waitForTimeout(900);
    const box = await rowBox();
    assert.ok(box[3] && box[0] >= -2 && box[1] <= box[2] + 2, root + ": 800 px list-focused " + key + " keeps the whole focused row in view (" + box + ")");
  }
  await page.close();
}
// Click a row (focus stays on it), scroll to the imagery, press ↓: the
// imagery holds, at 390 and 800 px, because the selected row is out of view.
for (const width of [390, 800]) for (const [hash, root] of [["#flare-outs", "#fo-spells"], ["#flare-outs/dark-nights", "#dn-content"]]) {
  const page = await open(width, width === 390 ? 844 : 900, hash);
  await page.locator(root + " .fo-row", root === "#fo-spells" ? { hasText: "4080" } : {}).first().click(); await settle(page);
  assert.ok(await page.evaluate(() => !!document.activeElement?.closest(".fo-list")), "the clicked row keeps focus");
  const at = () => page.evaluate(r => Math.round((document.querySelector(r + " .emit-carousel") || document.querySelector(r + " .emit-section")).getBoundingClientRect().top), root);
  await page.evaluate(r => { const a = document.querySelector(r + " .emit-carousel") || document.querySelector(r + " .emit-section"); window.scrollBy(0, a.getBoundingClientRect().top - 60); }, root);
  await settle(page);
  const tops = [];
  for (let i = 0; i < 3; i++) { await page.keyboard.press("ArrowDown"); await page.waitForTimeout(900); tops.push(await at()); }
  // At 390 px a short detail near the page end can clamp the scroll; the imagery must stay on screen.
  assert.ok(tops.every(v => width === 390 ? v > -50 && v < 400 : Math.abs(v - 60) <= 4), root + " " + width + " px: click, scroll to imagery, ↓ holds it (" + tops + ")");
  assert.deepEqual(page.errors, [], "click-hold: no browser errors");
  await page.close();
}
// Short and zoomed windows: each image stays at least 200 px wide (floor 255 px).
for (const [w, h] of [[1176, 666], [980, 555]]) {
  const page = await open(w, h, "#flare-outs");
  await page.locator("#fo-spells .fo-row", { hasText: "4080" }).first().click(); await settle(page);
  const widths = await page.evaluate(() => [...document.querySelectorAll("#fo-spells .emit-carousel img")].map(i => Math.round(i.getBoundingClientRect().width)));
  assert.ok(widths.length === 2 && widths.every(v => v >= 200), w + "×" + h + ": images at least 200 px wide (" + widths + ")");
  await page.close();
}
const active = page => page.evaluate(() => document.activeElement?.getAttribute("aria-label") || document.activeElement?.textContent?.trim() || null);
for (const [mode, hash, root] of [["spells", "#flare-outs", "#fo-spells"], ["darknights", "#flare-outs/dark-nights", "#dn-content"]]) {
  // Phone (page scrolls as one): ↓ while reading the imagery keeps the new
  // EMIT section where the old one was, instead of scrolling back to the list.
  const phone = await open(390, 844, hash);
  if (mode === "spells") await phone.locator(root + " .fo-row", { hasText: "4080" }).first().click();
  await settle(phone);
  assert.equal(await phone.evaluate(() => document.documentElement.scrollWidth), 390, mode + ": no horizontal scroll at 390 px");
  await phone.evaluate(r => { document.activeElement?.blur(); const e = document.querySelector(r + " .emit-section"); window.scrollBy(0, e.getBoundingClientRect().top - 60); }, root);
  await settle(phone);
  const emitTop = () => phone.evaluate(r => Math.round(document.querySelector(r + " .emit-section").getBoundingClientRect().top), root);
  const before = await emitTop(), id0 = (await state(phone, root)).selected;
  await phone.keyboard.press("ArrowDown"); await settle(phone);
  assert.notEqual((await state(phone, root)).selected, id0, mode + ": phone ↓ selects the next row");
  assert.ok(Math.abs((await emitTop()) - before) <= 3, mode + ": phone ↓ keeps the EMIT section in place");
  assert.deepEqual(phone.errors, [], mode + ": phone: no browser errors");
  await phone.close();
  // Desktop focus: a focused carousel control survives ↓ (rebuilt detail)
  // and ← → (rebuilt slide); a ‹ › that becomes disabled hands focus over.
  const page = await open(1440, 900, hash);
  if (mode === "spells") await page.locator(root + " .fo-row", { hasText: "4080" }).first().click();
  await settle(page);
  const next = page.locator(root + ' .emit-carousel button[aria-label="Next EMIT scene"]');
  const prev = page.locator(root + ' .emit-carousel button[aria-label="Previous EMIT scene"]');
  const link = page.locator(root + " .emit-carousel .emit-slide a").first();
  const nav = (await next.count()) ? ((await next.isDisabled()) ? prev : next) : null;
  const target = nav || link, label = nav ? await nav.getAttribute("aria-label") : null;
  await target.focus();
  await page.keyboard.press("ArrowDown"); await settle(page);
  const afterDown = await active(page);
  // The same button in the new carousel, or the carousel when that button is disabled there.
  if (label) assert.ok(await page.evaluate(([r, l]) => { const a = document.activeElement, c = document.querySelector(r + " .emit-carousel");
    return a === c || (c.contains(a) && a.getAttribute("aria-label") === l); }, [root, label]), mode + ": focus stays on the carousel control after ↓ (" + afterDown + ")");
  else assert.ok(afterDown && afterDown !== "null", mode + ": focus stays in the detail after ↓");
  await page.keyboard.press("ArrowUp"); await settle(page);
  if (await link.count()) {
    await link.focus(); await page.keyboard.press("ArrowRight"); await settle(page);
    assert.ok(await page.evaluate(r => document.querySelector(r + " .emit-carousel").contains(document.activeElement), root), mode + ": focus stays in the carousel after →");
  }
  if (await next.count()) {
    if (await next.isDisabled()) await prev.click();
    await next.focus(); for (let i = 0; i < 6; i++) await page.keyboard.press("Enter");
    assert.equal(await active(page), "Previous EMIT scene", mode + ": a disabled › hands focus to ‹");
  }
  assert.deepEqual(page.errors, [], mode + ": focus checks: no browser errors");
  await page.close();
}
console.log(JSON.stringify(results, null, 1));
console.log("EMIT carousel checks: PASS");
await browser.close(); server.close();
