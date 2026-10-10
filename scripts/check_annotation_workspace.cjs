/* Browser workflow uses explicit synthetic annotation fixtures, not benchmark labels. */
const { chromium } = require("playwright");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const assert = require("node:assert/strict");
(async () => {
  const browser = await chromium.launch({
    executablePath: "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox"],
  });
  const fixture = fs.mkdtempSync(
    path.join(os.tmpdir(), "spatialmemory-annotation-check-"),
  );
  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 900 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(
      (process.env.SPATIALMEMORY_URL || "http://127.0.0.1:8000") + "/annotate",
    );
    for (const state of ["before", "after"]) {
      execFileSync("ffmpeg", [
        "-v",
        "error",
        "-n",
        "-i",
        `/workspace/spatialmemory-fixtures/${state}.mp4`,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        path.join(fixture, state + ".mp4"),
      ]);
      await page
        .locator(`#${state}File`)
        .setInputFiles(path.join(fixture, state + ".mp4"));
    }
    await page.waitForFunction(
      () =>
        document.getElementById("beforeVideo").readyState >= 2 &&
        document.getElementById("afterVideo").readyState >= 2,
    );
    await page.locator("#sceneId").fill("synthetic_ui_check");
    await page
      .locator("#provenance")
      .fill("Explicit browser synthetic test fixture; not real ground truth");
    await page.locator("#origin").selectOption("SYNTHETIC");
    await page.locator("#objectId").fill("unit_object");
    await page.locator("#label").fill("textured panel");
    await page.locator("#change").selectOption("UNCHANGED");
    await page.locator("#annotator").fill("Test actor A");
    await page
      .locator("#source")
      .fill("Automated UI fixture; no independent human review");
    await page.locator("#save").click();
    assert.match(
      await page.locator("#objects").innerText(),
      /UNCHANGED.*SINGLE_REVIEW/,
    );
    await page.locator("#reviewer").fill("Test actor A");
    await page.locator("#save").click();
    assert.match(
      await page.locator("#status").innerText(),
      /different reviewer/,
    );
    await page.locator("#reviewer").fill("");
    await page.locator("#save").click();
    const downloaded = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloaded;
    const stream = await download.createReadStream();
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    const json = JSON.parse(Buffer.concat(chunks).toString());
    assert.equal(json.data_origin, "SYNTHETIC");
    assert.equal(json.changes.length, 1);
    assert.equal(json.changes[0].annotation_quality, "SINGLE_REVIEW");
    assert.equal(json.changes[0].before_observation.recording_id, "before.mp4");
    await page.locator("#import").setInputFiles({
      name: "synthetic_import.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(json)),
    });
    await page.waitForFunction(() =>
      document.getElementById("status").textContent.startsWith("Imported"),
    );
    await page.locator("#selected").selectOption("unit_object");
    assert.equal(await page.locator("#change").inputValue(), "UNCHANGED");
    await page.locator("#remove").click();
    assert.equal(await page.locator("#objects li").count(), 0);
    await page.locator("#export").click();
    assert.match(
      await page.locator("#status").innerText(),
      /at least one annotation/,
    );
    await page.setViewportSize({ width: 320, height: 740 });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    assert.deepEqual(errors, []);
    fs.mkdirSync("reports", { recursive: true });
    fs.writeFileSync(
      "reports/annotation_browser_check.json",
      JSON.stringify(
        {
          status: "passed",
          input: "explicit synthetic UI fixture; no real annotations created",
          checks: [
            "both video previews decode",
            "object-category save",
            "same-reviewer rejection",
            "schema validation and JSON download",
            "annotation import and selection",
            "deletion",
            "empty export rejection",
            "320px layout",
            "no JavaScript errors",
          ],
        },
        null,
        2,
      ) + "\n",
    );
    process.stdout.write("Annotation browser workflow passed\n");
  } finally {
    await browser.close();
    fs.rmSync(fixture, { recursive: true, force: true });
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
