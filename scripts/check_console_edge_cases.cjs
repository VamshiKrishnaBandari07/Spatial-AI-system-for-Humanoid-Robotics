/* Regression checks with actual completed-run data and explicitly simulated failures.
 * No new inference, robot commands, training, or source report modification. */
const { chromium } = require("playwright");
const fs = require("fs");
const assert = require("node:assert/strict");
(async () => {
  const job = process.env.SPATIALMEMORY_EXISTING_JOB;
  if (!job)
    throw Error("Set SPATIALMEMORY_EXISTING_JOB to a completed local job");
  const origin = process.env.SPATIALMEMORY_URL || "http://127.0.0.1:8000";
  const browser = await chromium.launch({
    executablePath: "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox"],
  });
  const errors = [];
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1080 },
    timezoneId: "Europe/London",
    deviceScaleFactor: 2,
  });
  page.on("pageerror", (e) => errors.push(e.message));
  const report = await (
    await page.request.get(`${origin}/jobs/${job}/report`)
  ).json();
  const world = await (
    await page.request.get(`${origin}/jobs/${job}/world`)
  ).json();
  assert(
    report.events.length >= 2,
    "Replay needs two actual decisions to check draft preservation",
  );
  await page.goto(origin);
  await page.evaluate((id) => {
    localStorage.setItem("spatialmemory_trials", "{}");
    localStorage.setItem("spatialmemory_reviews_" + id, "[]");
    localStorage.setItem("spatialmemory_last_run", id);
  }, job);
  await page.reload();
  await page.waitForFunction(
    () => document.getElementById("pointCount").textContent !== "—",
  );
  assert.equal(await page.locator("input[name=before]").inputValue(), "");
  assert.match(
    await page.locator("#beforeFile").innerText(),
    /Saved recording/,
  );
  await page.locator(".event").nth(0).click();
  await page.locator("#reviewDecision").selectOption("investigate");
  await page
    .locator("#reviewNotes")
    .fill("Unsaved draft: capture another angle.");
  await page.locator(".event").nth(1).click();
  await page.locator(".event").nth(0).click();
  assert.equal(
    await page.locator("#reviewNotes").inputValue(),
    "Unsaved draft: capture another angle.",
  );
  assert.match(await page.locator("#reviewCount").innerText(), /^0 \/ /);
  await page.reload();
  await page.waitForFunction(
    () => document.getElementById("pointCount").textContent !== "—",
  );
  await page.locator(".event").nth(0).click();
  assert.equal(
    await page.locator("#reviewNotes").inputValue(),
    "Unsaved draft: capture another angle.",
  );
  await page.locator("#saveReview").click();
  assert.match(await page.locator("#reviewCount").innerText(), /^1 \/ /);
  const slow = "a".repeat(32),
    latest = "b".repeat(32);
  for (const id of [slow, latest]) {
    await page.route(`**/jobs/${id}/report`, async (route) => {
      if (id === slow) await new Promise((resolve) => setTimeout(resolve, 200));
      await route.fulfill({ json: report });
    });
    await page.route(`**/jobs/${id}/world`, async (route) => {
      if (id === slow) await new Promise((resolve) => setTimeout(resolve, 200));
      await route.fulfill({ json: world });
    });
  }
  const selected = await page.evaluate(
    async ([slow, latest]) => {
      const first = loadRun(slow);
      await new Promise((resolve) => setTimeout(resolve, 30));
      const second = loadRun(latest);
      await Promise.all([first, second]);
      return job;
    },
    [slow, latest],
  );
  assert.equal(selected, latest, "Latest requested trial must remain selected");
  await page.evaluate((id) => loadRun(id), job);
  // An old research run must not enable unavailable profiles in a new upload.
  await page.route("**/health", (route) =>
    route.fulfill({
      json: {
        status: "ok",
        capabilities: { sparse: true, vggt: false, semantics: false },
      },
    }),
  );
  await page.reload();
  await page.waitForFunction(
    () => document.getElementById("pointCount").textContent !== "—",
  );
  assert.equal(
    await page.locator("select[name=geometry_backend]").inputValue(),
    "sparse",
  );
  assert.equal(
    await page.locator("input[name=semantics_enabled]").isChecked(),
    false,
  );
  assert.equal(
    await page.locator("input[name=semantics_enabled]").isDisabled(),
    true,
  );
  assert.match(await page.locator("#loadedTrial").innerText(), /vggt geometry/);
  await page
    .locator("input[name=before]")
    .setInputFiles("/workspace/spatialmemory-fixtures/before.mp4");
  await page.locator("input[name=before]").setInputFiles([]);
  assert.equal(
    await page.locator("#beforeFile").innerText(),
    "No recording selected",
  );
  assert.equal(await page.locator("#bv").isVisible(), false);
  assert.equal(
    await page
      .locator("input[name=before]")
      .evaluate((input) => input.validationMessage),
    "Please select a file.",
  );
  await page
    .locator("input[name=before]")
    .setInputFiles("/workspace/spatialmemory-fixtures/before.mp4");
  await page
    .locator("input[name=after]")
    .setInputFiles("/workspace/spatialmemory-fixtures/after.mp4");
  let uploadProfileCorrect = false;
  await page.route("**/jobs", async (route) => {
    const body = route.request().postDataBuffer().toString();
    uploadProfileCorrect =
      body.includes('name="geometry_backend"\r\n\r\nsparse') &&
      !body.includes('name="semantics_enabled"');
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.fulfill({
      status: 429,
      json: { detail: "A job is already active" },
    });
  });
  await page.locator("#run").click();
  await page.waitForFunction(
    () => document.getElementById("trialName").disabled,
  );
  assert.equal(await page.locator("input[name=before]").isDisabled(), true);
  assert.equal(
    await page.locator("#recentRuns button").first().isDisabled(),
    true,
  );
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.startsWith("Failed"),
  );
  assert(
    uploadProfileCorrect,
    "Upload must preserve form values before disabling inputs",
  );
  assert.equal(await page.locator("#trialName").isDisabled(), false);
  assert.equal(
    await page.locator("input[name=semantics_enabled]").isDisabled(),
    true,
  );
  await page.unroute("**/jobs");
  await page.route("**/evidence/*", (route) =>
    route.fulfill({ status: 404, body: "Missing evidence" }),
  );
  await page.evaluate(() => {
    delete report.events[0].evidence.observations[0].provenance
      .raw_detector_score;
  });
  await page.locator(".event").nth(0).click();
  await page.waitForFunction(() =>
    document
      .getElementById("frames")
      .textContent.includes("Supporting image unavailable"),
  );
  await page
    .locator("#frames details")
    .first()
    .evaluate((details) => (details.open = true));
  assert.match(
    await page.locator("#frames").innerText(),
    /raw detector score unavailable/,
  );
  await page.setViewportSize({ width: 320, height: 740 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    ),
    false,
  );
  const blocked = await browser.newPage();
  blocked.on("pageerror", (e) => errors.push(e.message));
  await blocked.addInitScript(() => {
    Storage.prototype.getItem = () => {
      throw new DOMException("Storage disabled", "SecurityError");
    };
    Storage.prototype.setItem = () => {
      throw new DOMException("Storage disabled", "SecurityError");
    };
  });
  await blocked.goto(origin);
  await blocked.waitForFunction(
    () => document.getElementById("serviceStatus").dataset.state === "ready",
  );
  await blocked.evaluate((id) => loadRun(id), job);
  await blocked.locator(".event").first().click();
  await blocked.locator("#reviewDecision").selectOption("investigate");
  await blocked
    .locator("#reviewNotes")
    .fill("Session-only review with storage blocked.");
  await blocked.locator("#saveReview").click();
  assert.match(
    await blocked.locator("#reviewStatus").innerText(),
    /session only/,
  );
  const downloadPromise = blocked.waitForEvent("download");
  await blocked.locator("#exportReview").click();
  const download = await downloadPromise;
  const reviewExport = JSON.parse(
    fs.readFileSync(await download.path(), "utf8"),
  );
  assert(
    Object.values(reviewExport.reviews).some(
      (r) => r.notes === "Session-only review with storage blocked.",
    ),
  );
  const offline = await browser.newPage();
  offline.on("pageerror", (e) => errors.push(e.message));
  await offline.route("**/health", (route) =>
    route.fulfill({ status: 503, body: "Offline" }),
  );
  await offline.goto(origin);
  await offline.waitForFunction(
    () =>
      document.getElementById("serviceStatus").dataset.state === "unavailable",
  );
  assert.equal(await offline.locator("#run").isDisabled(), true);
  assert.deepEqual(errors, []);
  const summary = {
    status: "passed",
    mode: "UI regressions using an existing synthetic report; failure responses explicitly simulated",
    checks: [
      "wrong-shaped browser storage",
      "unsaved review drafts across selection and reload",
      "stale trial load race",
      "unavailable profile after research-run restore",
      "cleared file preview and validation",
      "upload form snapshot and busy controls",
      "missing detector score and evidence image",
      "320px viewport at 2x pixel ratio",
      "blocked browser storage with session review export",
      "offline service startup",
    ],
    model_outputs_modified: false,
    training_started: false,
    real_world_accuracy: "not evaluated",
  };
  fs.writeFileSync(
    "docs/console_edge_cases.json",
    JSON.stringify(summary, null, 2) + "\n",
  );
  console.log(JSON.stringify(summary, null, 2));
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
