/* Actual UI upload and fresh neural inference. Explicit inputs; no mocked outputs. */
const { chromium } = require("playwright");
const fs = require("fs");
const assert = require("node:assert/strict");
const origin = process.env.SPATIALMEMORY_URL;
const before = process.env.SPATIALMEMORY_BEFORE;
const after = process.env.SPATIALMEMORY_AFTER;
const output = process.env.SPATIALMEMORY_E2E_OUTPUT;
if (!origin || !before || !after || !output)
  throw Error("Set URL, BEFORE, AFTER and E2E_OUTPUT");
if ([".json", ".job.json", ".zip", ".desktop.png"].some(suffix => fs.existsSync(output + suffix)))
  throw Error("Choose a new E2E output prefix; existing artifacts are preserved");

(async () => {
  const browser = await chromium.launch({
    executablePath: "/usr/bin/chromium", headless: true, args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(origin);
    await page.waitForFunction(() => document.getElementById("serviceStatus").dataset.state === "ready");
    await page.locator("[name=before]").setInputFiles(before);
    await page.locator("[name=after]").setInputFiles(after);
    await page.locator("[name=geometry_backend]").selectOption("vggt");
    await page.locator("[name=semantics_enabled]").check();
    const response = page.waitForResponse(r => r.url().endsWith("/jobs") && r.request().method() === "POST");
    const start = Date.now();
    await page.locator("#run").click();
    const upload = await response;
    assert.equal(upload.status(), 200);
    const { id } = await upload.json();
    fs.writeFileSync(output + ".job.json", JSON.stringify({ id }));
    await page.waitForFunction(
      () => /Processing complete|^Failed:/.test(document.getElementById("status").textContent),
      {}, { timeout: 900000 },
    );
    assert.match(await page.locator("#status").innerText(), /Processing complete/);
    const report = await (await page.request.get(`${origin}/jobs/${id}/report`)).json();
    assert.equal(report.status, "complete");
    assert.equal(report.registration.valid, true);
    for (const state of ["before", "after"])
      assert.equal(report.geometry[state].predictions_origin, "pretrained inference in current run");
    assert.equal(report.config.association_mode, "multiview");
    assert.equal(await page.locator("#artifactLinks a").count(), 7);
    const archive = await page.request.get(`${origin}/jobs/${id}/results`);
    assert.equal(archive.status(), 200);
    fs.writeFileSync(output + ".zip", await archive.body());
    const cloud = await page.request.get(`${origin}/jobs/${id}/artifacts/aligned`);
    assert.equal(cloud.status(), 200);
    assert.match((await cloud.body()).toString(), /^ply/);
    const audit = await (await page.request.get(`${origin}/jobs/${id}/artifacts/association`)).json();
    assert.equal(audit.score_is_probability, false);
    assert(audit.candidates.length > 0);
    if (report.events.length) {
      await page.locator(".event").first().click();
      assert(await page.locator("#frames img").count() > 0);
    }
    assert.deepEqual(errors, []);
    await page.screenshot({ path: output + ".desktop.png", fullPage: true });
    fs.writeFileSync(output + ".json", JSON.stringify({
      status: "passed", job: id,
      experiment: before === after ? "Identical-recording neural UI control; no real change benchmark" : "Neural UI execution; ground-truth accuracy not evaluated",
      elapsed_seconds: (Date.now() - start) / 1000,
      runtime_seconds: report.runtime_seconds, events: report.events.length,
      registration: report.registration, model_geometry: report.geometry,
      page_errors: errors,
    }, null, 2) + "\n");
    console.log("Actual neural application E2E passed", id);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
