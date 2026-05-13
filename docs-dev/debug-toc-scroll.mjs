/**
 * Find which element sits above the TOC after scrolling (home page, slate).
 * Run: cd docs-dev && PATH=$CONDA_PREFIX/bin:$PATH node debug-toc-scroll.mjs
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const indexUrl = path.join(repoRoot, "site", "index.html");

function resolvePython() {
  if (process.env.CONDA_PREFIX) {
    const p = path.join(process.env.CONDA_PREFIX, "bin", "python");
    if (fs.existsSync(p)) return p;
  }
  return "python";
}

function runMkdocsBuild() {
  const r = spawnSync(resolvePython(), ["-m", "mkdocs", "build", "-q"], {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONSAFEPATH: "1" },
  });
  if (r.status !== 0) throw new Error((r.stderr || r.stdout).slice(0, 2000));
}

if (!fs.existsSync(indexUrl)) runMkdocsBuild();

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 1 });
await page.goto(`file://${indexUrl}`, { waitUntil: "networkidle0", timeout: 90_000 });

const probe = async (label, scrollY) => {
  await page.evaluate((y) => window.scrollTo(0, y), scrollY);
  await new Promise((r) => setTimeout(r, 150));
  return page.evaluate(() => {
    const toc = document.querySelector(".md-sidebar--secondary");
    if (!toc) return { error: "no .md-sidebar--secondary" };
    const r = toc.getBoundingClientRect();
    const pts = [
      { name: "toc upper", x: r.left + r.width * 0.5, y: r.top + 48 },
      { name: "toc mid", x: r.left + r.width * 0.5, y: r.top + r.height * 0.35 },
      { name: "toc lower", x: r.left + r.width * 0.5, y: r.top + r.height * 0.55 },
    ];
    return pts.map(({ name, x, y }) => {
      const el = document.elementFromPoint(x, y);
      if (!el) return { name, x, y, el: null };
      let cur = el;
      const chain = [];
      for (let i = 0; i < 8 && cur; i++) {
        const cls = cur.className?.toString?.().slice(0, 120) || "";
        chain.push(`${cur.tagName}${cls ? "." + cls.split(" ").filter(Boolean).slice(0, 4).join(".") : ""}`);
        cur = cur.parentElement;
      }
      const cs = getComputedStyle(el);
      return {
        name,
        x,
        y,
        topEl: chain[0],
        chain: chain.join(" < "),
        zIndex: cs.zIndex,
        position: cs.position,
        bg: cs.backgroundColor,
      };
    });
  });
};

for (const y of [0, 400, 900, 1400, 2200]) {
  const rows = await probe(`scrollY=${y}`, y);
  console.log("\n=== scrollY", y, "===");
  console.log(JSON.stringify(rows, null, 2));
}

await browser.close();
