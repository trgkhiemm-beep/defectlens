// DefectLens in the browser: shared embedder + per-category scorer, ONNX Runtime Web.
// Mirrors src/edge/runtime.py: resize to 256, ImageNet normalisation, embed, score, threshold.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

const $ = (id) => document.getElementById(id);
const state = { manifest: null, embedderBytes: null, embedder: null, backend: null, ready: "", scorerBytes: new Map(),
                scorers: new Map(), hasImage: false, running: false, pending: false };

const setStatus = (text) => { $("status").textContent = text; };
const mb = (bytes) => (bytes / 2 ** 20).toFixed(1);

async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    setStatus(`Downloading ${label}: ${mb(got)}${total ? ` / ${mb(total)}` : ""} MB (cached by the browser afterwards)`);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function wantedProviders() {
  return $("backend").value === "webgpu" && navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
}

async function createSession(bytes) {
  const providers = wantedProviders();
  try {
    const s = await ort.InferenceSession.create(bytes, { executionProviders: providers, graphOptimizationLevel: "all" });
    return { session: s, backend: providers[0] };
  } catch (err) {
    if (providers[0] !== "webgpu") throw err;
    console.warn("WebGPU session failed, falling back to WASM", err);
    const s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return { session: s, backend: "wasm" };
  }
}

async function loadEmbedder() {
  const t0 = performance.now();
  state.embedderBytes ??= await fetchBytes(state.manifest.embedder.model, "embedder");
  setStatus("Compiling the model…");
  const { session, backend } = await createSession(state.embedderBytes);
  state.embedder = session;
  state.backend = backend;
  state.scorers.clear();
  const threads = ort.env.wasm.numThreads;
  state.ready = `Ready · ${backend === "webgpu" ? "WebGPU" : `WASM (${threads} thread${threads > 1 ? "s" : ""})`} · ` +
                `loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s · everything runs locally`;
  setStatus(state.ready);
}

async function scorerFor(cat) {
  const key = `${state.backend}:${cat}`;
  if (!state.scorers.has(key)) {
    if (!state.scorerBytes.has(cat)) {
      state.scorerBytes.set(cat, await fetchBytes(state.manifest.categories[cat].model, `${cat} memory bank`));
    }
    const { session } = await createSession(state.scorerBytes.get(cat));
    state.scorers.set(key, session);
    setStatus(state.ready);
  }
  return state.scorers.get(key);
}

function toTensor(ctx, size) {
  const { data } = ctx.getImageData(0, 0, size, size);
  const [m0, m1, m2] = state.manifest.mean;
  const [s0, s1, s2] = state.manifest.std;
  const n = size * size;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = (data[4 * i] / 255 - m0) / s0;
    f[n + i] = (data[4 * i + 1] / 255 - m1) / s1;
    f[2 * n + i] = (data[4 * i + 2] / 255 - m2) / s2;
  }
  return new ort.Tensor("float32", f, [1, 3, size, size]);
}

// Jet colormap, same visual convention as the Python overlay: mid-scale == threshold.
function jet(v) {
  const r = Math.min(Math.max(1.5 - Math.abs(4 * v - 3), 0), 1);
  const g = Math.min(Math.max(1.5 - Math.abs(4 * v - 2), 0), 1);
  const b = Math.min(Math.max(1.5 - Math.abs(4 * v - 1), 0), 1);
  return [r * 255, g * 255, b * 255];
}

function drawOverlay(amap, threshold, size) {
  const src = $("input-canvas").getContext("2d").getImageData(0, 0, size, size).data;
  const ctx = $("output-canvas").getContext("2d");
  const out = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const [r, g, b] = jet(Math.min(Math.max(amap[i] / (2 * threshold), 0), 1));
    let R = 0.55 * src[4 * i] + 0.45 * r, G = 0.55 * src[4 * i + 1] + 0.45 * g, B = 0.55 * src[4 * i + 2] + 0.45 * b;
    if (amap[i] >= threshold) {  // contour: above-threshold pixel with a below-threshold 4-neighbour
      const x = i % size, y = (i / size) | 0;
      const edge = x === 0 || y === 0 || x === size - 1 || y === size - 1 ||
        amap[i - 1] < threshold || amap[i + 1] < threshold || amap[i - size] < threshold || amap[i + size] < threshold;
      if (edge) { R = G = B = 255; }
    }
    out.data.set([R, G, B, 255], 4 * i);
  }
  ctx.putImageData(out, 0, 0);
}

function showResult(cat, score, threshold, t) {
  const ratio = score / threshold;
  const defect = score >= threshold;
  const v = $("verdict");
  v.className = `verdict ${defect ? "bad" : "ok"}`;
  v.textContent = `${defect ? "DEFECT" : "OK"} · score = ${ratio.toFixed(2)} × threshold`;
  const rows = [
    ["Category", cat], ["Score", score.toFixed(4)], ["Threshold", threshold.toFixed(4)],
    ["Latency", `${t.total.toFixed(0)} ms (preprocess ${t.pre.toFixed(0)} · embed ${t.embed.toFixed(0)} · score ${t.score.toFixed(0)})`],
    ["Backend", state.backend === "webgpu" ? "WebGPU" : "WASM"],
  ];
  const dl = $("details");
  dl.replaceChildren();
  for (const [k, val] of rows) {
    dl.append(Object.assign(document.createElement("dt"), { textContent: k }),
              Object.assign(document.createElement("dd"), { textContent: val }));
  }
  if (ratio >= 3) {
    dl.append(Object.assign(document.createElement("dt"), { textContent: "Warning" }),
              Object.assign(document.createElement("dd"), { className: "warn",
                textContent: `score is ${ratio.toFixed(1)}× the threshold: this may not be a ${cat} image at all` }));
  }
}

async function inspect() {
  if (!state.hasImage || !state.embedder) return;
  if (state.running) { state.pending = true; return; }
  state.running = true;
  try {
    const cat = $("category").value;
    const size = state.manifest.image_size;
    const scorer = await scorerFor(cat);
    const t0 = performance.now();
    const x = toTensor($("input-canvas").getContext("2d", { willReadFrequently: true }), size);
    const t1 = performance.now();
    const { embedding } = await state.embedder.run({ input: x });
    const t2 = performance.now();
    const out = await scorer.run({ input: embedding });
    const t3 = performance.now();
    const threshold = state.manifest.categories[cat].threshold;
    const score = out.score.data[0];
    drawOverlay(out.anomaly_map.data, threshold, size);
    showResult(cat, score, threshold, { pre: t1 - t0, embed: t2 - t1, score: t3 - t2, total: t3 - t0 });
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  } finally {
    state.running = false;
    if (state.pending) { state.pending = false; inspect(); }
  }
}

async function setImage(source) {
  const bitmap = await createImageBitmap(source);
  const size = state.manifest.image_size;
  const ctx = $("input-canvas").getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, size, size);  // same square resize as the Python pipeline
  state.hasImage = true;
  await inspect();
}

function buildUi() {
  const m = state.manifest;
  const cats = Object.keys(m.categories);
  $("category").replaceChildren(...cats.map((c) => new Option(c.replace("_", " "), c)));
  const ex = $("examples");
  for (const c of cats) {
    for (const kind of ["good", "defect"]) {
      const b = document.createElement("button");
      b.type = "button";
      b.title = `${c} (${kind})`;
      const img = Object.assign(document.createElement("img"), { src: `samples/${c}_${kind}.png`, alt: `${c}, ${kind}`, loading: "lazy" });
      b.append(img, Object.assign(document.createElement("span"), { textContent: `${c.replace("_", " ")} · ${kind}` }));
      b.addEventListener("click", async () => {
        $("category").value = c;
        const res = await fetch(img.src);
        await setImage(await res.blob());
      });
      ex.append(b);
    }
  }
  const tbody = $("metrics").querySelector("tbody");
  for (const [c, i] of Object.entries(m.categories)) {
    const tr = document.createElement("tr");
    for (const v of [c, i.image_auroc.toFixed(3), i.aupro_30.toFixed(3), i.f1.toFixed(3), i.fpr.toFixed(3),
                     i.threshold.toFixed(3), `${i.mb} MB`]) {
      tr.append(Object.assign(document.createElement("td"), { textContent: v }));
    }
    tbody.append(tr);
  }
  $("model-note").textContent =
    `${m.backbone} (${m.layers.join(" + ")}) PatchCore, coreset memory bank ${m.variant.split("/r")[1] * 100}% ` +
    `of normal patches. Shared embedder ${m.embedder.mb} MB (FP16 weights, FP32 compute). ` +
    `Threshold policy: ${m.threshold_policy}, target false-reject rate ${m.target_fpr * 100}%. ` +
    `Metrics on the MVTec AD test split; carpet's high FPR is a documented val/test drift, not a bug.`;

  $("category").addEventListener("change", inspect);
  $("backend").addEventListener("change", async () => { await loadEmbedder(); await inspect(); });
  $("file").addEventListener("change", (e) => e.target.files[0] && setImage(e.target.files[0]));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) setImage(f);
  });
}

async function main() {
  if (new URLSearchParams(location.search).get("backend") === "webgpu") $("backend").value = "webgpu";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  try {
    const res = await fetch("models/manifest.json");
    if (!res.ok) throw new Error(`models/manifest.json: HTTP ${res.status}`);
    state.manifest = await res.json();
    buildUi();
    await loadEmbedder();
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the model: ${err.message}`);
  }
}

main();
