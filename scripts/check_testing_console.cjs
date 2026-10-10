/* UI replay check: uses an existing real report, does not start inference or training. */
const { chromium } = require("playwright");
const fs = require("fs");
const assert = require("node:assert/strict");
(async () => {
  const job = process.env.SPATIALMEMORY_EXISTING_JOB;
  if (!job)
    throw Error("Set SPATIALMEMORY_EXISTING_JOB to a completed local job");
  const browser = await chromium.launch({
    executablePath: "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1080 },
    timezoneId: "Europe/London",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const origin = process.env.SPATIALMEMORY_URL || "http://127.0.0.1:8000";
  await page.goto(origin);
  await page.waitForFunction(
    () => document.getElementById("serviceStatus").dataset.state === "ready",
  );
  assert.equal(await page.locator("h1").innerText(), "SpatialMemory-X Ultra");
  assert.equal(
    await page.evaluate(() => document.documentElement.dataset.theme),
    "dark",
  );
  await page.locator("#themeToggle").click();
  assert.equal(
    await page.evaluate(() => document.documentElement.dataset.theme),
    "light",
  );
  await page.reload();
  await page.waitForFunction(
    () => document.getElementById("serviceStatus").dataset.state === "ready",
  );
  assert.equal(
    await page.evaluate(() => document.documentElement.dataset.theme),
    "light",
  );
  await page.locator("#themeToggle").click();
  assert.equal(
    await page.evaluate(() => document.documentElement.dataset.theme),
    "dark",
  );
  assert.equal(await page.locator("#sceneEmpty").isVisible(), true);
  assert.equal(await page.locator("#exportReport").isDisabled(), true);
  await page.locator("select[name=geometry_backend]").selectOption("vggt");
  assert.equal(await page.locator("#profileNotice").isVisible(), true);
  await page.locator("select[name=geometry_backend]").selectOption("sparse");
  assert.equal(await page.locator("#profileNotice").isVisible(), false);
  await page.evaluate(
    (id) => localStorage.setItem("spatialmemory_last_run", id),
    job,
  );
  await page.reload();
  await page.waitForFunction(
    () => document.getElementById("pointCount").textContent !== "—",
  );
  assert.equal(await page.locator("#sceneEmpty").isVisible(), false);
  assert.equal(
    await page.locator("#alignmentBadge").innerText(),
    "Alignment passed",
  );
  await page.locator(".event").first().click();
  assert.doesNotMatch(
    await page.locator(".event").first().innerText(),
    /model-proposed|proposed electronic device/i,
  );
  assert.match(
    await page.locator("#explanation").innerText(),
    /appears unchanged/,
  );
  const outputBefore = await (
    await page.request.get(origin + "/jobs/" + job + "/report")
  ).json();
  assert.match(outputBefore.events[0].explanation, /Model-proposed/);
  assert.equal(
    await page.evaluate(() =>
      processingMessage({ status: "running", stage: "shared_registration" }),
    ),
    "Aligning your scenes",
  );
  assert.equal(
    await page.evaluate(() =>
      resultSummary({ change: "UNKNOWN" }).includes("More evidence is needed"),
    ),
    true,
  );
  await page.waitForFunction(
    () =>
      document.querySelectorAll("#frames img").length > 0 &&
      Array.from(document.querySelectorAll("#frames img")).every(
        (i) => i.complete && i.naturalWidth > 0,
      ),
  );
  await page.locator("#reviewerName").fill("UI test reviewer");
  await page.locator("#reviewDecision").selectOption("investigate");
  await page
    .locator("#reviewNotes")
    .fill(
      "UI replay check: request a closer viewpoint. Not an accuracy evaluation.",
    );
  await page.locator("#saveReview").click();
  assert.match(await page.locator("#reviewStatus").innerText(), /saved/);
  assert.match(await page.locator("#reviewCount").innerText(), /^1 \/ /);
  for (const [selector, kind] of [
    ["#exportReport", "report"],
    ["#exportReview", "reviews"],
  ]) {
    const downloadPromise = page.waitForEvent("download");
    await page.locator(selector).click();
    const download = await downloadPromise;
    const contents = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    if (kind === "report") assert(contents.registration.valid);
    else {
      assert.equal(contents.job_id, job);
      assert(
        Object.values(contents.reviews).some(
          (r) => r.assessment === "investigate",
        ),
      );
    }
  }
  await page.reload();
  await page.waitForFunction(
    () => document.getElementById("pointCount").textContent !== "—",
  );
  await page.locator(".event").first().click();
  assert.equal(
    await page.locator("#reviewDecision").inputValue(),
    "investigate",
  );
  assert.match(
    await page.locator("#reviewNotes").inputValue(),
    /closer viewpoint/,
  );
  await page.locator("#objectLayer").uncheck();
  await page.locator("#objectLayer").check();
  await page.locator("#world").focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("+");
  await page.keyboard.press("0");
  await page.locator("#recentRuns button").first().click();
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.startsWith("Loaded"),
  );
  // Controlled HTTP failure verifies recovery without launching a new job.
  await page.route("**/jobs", (route) =>
    route.fulfill({
      status: 429,
      contentType: "text/plain",
      body: "A job is already active",
    }),
  );
  await page
    .locator("input[name=before]")
    .setInputFiles("/workspace/spatialmemory-fixtures/before.mp4");
  await page
    .locator("input[name=after]")
    .setInputFiles("/workspace/spatialmemory-fixtures/after.mp4");
  await page.locator("#trialName").fill("UI-only failure recovery");
  await page.locator("#run").click();
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.startsWith("Failed"),
  );
  assert.equal(await page.locator("#run").isDisabled(), false);
  assert.equal(await page.locator("#exportReport").isDisabled(), false);
  await page.unroute("**/jobs");
  await page.locator("#recentRuns button").first().click();
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.startsWith("Loaded"),
  );
  await page.locator(".event").first().click();
  await page.evaluate(() => window.scrollTo(0, 0));
  fs.mkdirSync("docs/screenshots", { recursive: true });
  await page.screenshot({
    path: "docs/screenshots/testing-console-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "docs/screenshots/testing-console-mobile.png",
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  const summary = {
    status: "passed",
    mode: "existing completed synthetic-run replay; no new inference",
    job_id: job,
    checks: [
      "original product branding and direct result copy",
      "dark theme and persisted light theme toggle",
      "friendly labels for actual processing stages",
      "unchanged original report provenance",
      "empty state",
      "service availability",
      "research profile licence notice",
      "real report and geometry loading",
      "evidence image loading",
      "browser review save and reload",
      "report and separate review downloads",
      "keyboard orbit and zoom",
      "layer toggles",
      "recent trial restore",
      "simulated HTTP 429 recovery",
      "mobile overflow",
      "no JavaScript errors",
    ],
    real_world_accuracy: "not evaluated",
  };
  fs.writeFileSync(
    "docs/testing_console_smoke.json",
    JSON.stringify(summary, null, 2) + "\n",
  );
  console.log(JSON.stringify(summary, null, 2));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
