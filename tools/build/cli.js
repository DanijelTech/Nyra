#!/usr/bin/env node
/**
 * nyra-build launcher.
 *
 * The build itself is `tools/build/nyra_build.py` — this file only finds a Python
 * interpreter, forwards argv/stdin/stdout and returns the child's exit code. It
 * deliberately has zero dependencies: an npm package that installs nothing cannot
 * become a supply-chain problem for a robot build.
 *
 * Environment:
 *   NYRA_PYTHON   interpreter to use (default: python3, or `python` on Windows)
 *   NYRA_ROOT     workspace root (default: discovered from the current directory)
 */
"use strict";

const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const here = __dirname;
const script = path.join(here, "nyra_build.py");

if (!fs.existsSync(script)) {
  console.error(`nyra-build: builder not found at ${script}`);
  process.exit(1);
}

function findRoot(start) {
  let dir = path.resolve(start);
  for (;;) {
    if (fs.existsSync(path.join(dir, ".gitmodules"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

function configuredPython() {
  const root = process.env.NYRA_ROOT || findRoot(process.cwd());
  if (!root) return "";
  const file = path.join(root, ".nyra-build.json");
  if (!fs.existsSync(file)) return "";
  try {
    const configured = JSON.parse(fs.readFileSync(file, "utf8")).python || "";
    return configured && fs.existsSync(configured) ? configured : "";
  } catch {
    return "";
  }
}

const python =
  process.env.NYRA_PYTHON ||
  configuredPython() ||
  (process.platform === "win32" ? "python" : "python3");

const args = [script, ...process.argv.slice(2)];
if (process.env.NYRA_ROOT && !args.includes("--root")) {
  args.push("--root", process.env.NYRA_ROOT);
}

const child = spawn(python, args, {
  stdio: "inherit",
  cwd: process.cwd(),
  env: process.env,
  windowsHide: true,
});

child.on("error", (error) => {
  if (error.code === "ENOENT") {
    console.error(
      `nyra-build: cannot start '${python}'. Set NYRA_PYTHON to a Python 3.10+ interpreter.`
    );
    process.exit(127);
  }
  console.error(`nyra-build: ${error.message}`);
  process.exit(1);
});

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => {
    if (!child.killed) child.kill(signal);
  });
}

child.on("exit", (code, signal) => {
  if (signal) {
    // Re-raise so the shell sees the same signal (e.g. Ctrl-C -> 130).
    process.kill(process.pid, signal);
    return;
  }
  process.exit(typeof code === "number" ? code : 1);
});
