/**
 * Renders the MkDocs home page (light palette) and writes a PNG for local review.
 *
 * Prerequisites (openscvx conda env):
 *   conda install -c conda-forge nodejs
 *   cd docs-dev && npm install
 *
 * Run (after `conda activate openscvx` so this env’s Node/npm are first on PATH):
 *   npm run screenshot-home
 *
 * Repo note: a top-level `material/` stub exists for theme overrides; Python would
 * import it instead of mkdocs-material unless PYTHONSAFEPATH=1 (set automatically below).
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const siteDir = path.join(repoRoot, "site");
const outPng = path.join(__dirname, "home-light.png");

function resolvePython() {
  if (process.env.PYTHON && fs.existsSync(process.env.PYTHON)) return process.env.PYTHON;
  if (process.env.CONDA_PREFIX) {
    const p = path.join(process.env.CONDA_PREFIX, "bin", "python");
    if (fs.existsSync(p)) return p;
  }
  return "python";
}

function runMkdocsBuild() {
  const bin = resolvePython();
  const r = spawnSync(bin, ["-m", "mkdocs", "build", "-q"], {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONSAFEPATH: "1" },
  });
  if (r.status !== 0) {
    console.error(r.stderr || r.stdout);
    throw new Error(`mkdocs build failed (exit ${r.status})`);
  }
}

await runMkdocsBuild();

const indexUrl = path.join(siteDir, "index.html");
if (!fs.existsSync(indexUrl)) {
  throw new Error(`Missing ${indexUrl}; mkdocs build did not produce site/.`);
}

const browser = await puppeteer.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
  await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 2 });
  await page.goto(`file://${indexUrl}`, { waitUntil: "networkidle0", timeout: 60_000 });
  await page.screenshot({ path: outPng, fullPage: true });
  console.log(`Wrote ${outPng}`);
} finally {
  await browser.close();
}
