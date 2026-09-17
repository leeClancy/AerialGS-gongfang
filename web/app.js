const STAGE_META = {
  prepare_images: { label: "整理图片", device: "CPU" },
  feature_extract: { label: "特征提取", device: "GPU" },
  feature_match: { label: "特征匹配", device: "GPU" },
  view_graph_calibrator: { label: "视图图标定", device: "CPU" },
  global_mapper: { label: "全局重建", device: "CPU" },
  glomap_mapper: { label: "独立 GLOMAP", device: "CPU" },
  incremental_mapper: { label: "增量重建", device: "CPU" },
  validate_model: { label: "模型检查", device: "CPU" },
  undistort_scene: { label: "去畸变", device: "CPU" },
  export_scene: { label: "导出场景", device: "CPU" },
  train_gsplat: { label: "gsplat 训练", device: "GPU" },
};

const STATUS_LABEL = {
  queued: "排队",
  running: "进行中",
  succeeded: "完成",
  failed: "失败",
  cancelled: "已取消",
  pending: "等待",
};

const state = {
  projectId: null,
  jobId: null,
  ws: null,
  es: null,
  pickerTarget: null,
  pickerMode: "dir",
  pickerSuffix: "",
  runtimeArtifactId: null,
  logPinned: true,
  sourceDir: "",
  workDir: "",
  sourceDirs: [],
  toastedJob: null,
  followLive: true,
  applyingPreset: false,
};

const $ = (id) => document.getElementById(id);

function sourceKey(path) {
  return String(path || "").replace(/[\\/]+$/, "").toLowerCase();
}

function renderSources() {
  const box = $("source-chips");
  if (!box) return;
  if (!state.sourceDirs.length) {
    box.className = "chips empty";
    box.textContent = "还没有文件夹。";
    return;
  }
  box.className = "chips";
  box.innerHTML = state.sourceDirs.map((path, index) => `
    <div class="chip" title="${esc(path)}">
      <span>${esc(path)}</span>
      <button type="button" data-remove="${index}" aria-label="移除">×</button>
    </div>
  `).join("");
}

function addSources(paths) {
  const seen = new Set(state.sourceDirs.map(sourceKey));
  for (const raw of paths || []) {
    const path = String(raw || "").trim();
    if (!path) continue;
    const key = sourceKey(path);
    if (seen.has(key)) continue;
    seen.add(key);
    state.sourceDirs.push(path);
  }
  renderSources();
  return state.sourceDirs;
}

function jobParams() {
  return {
    preset: $("preset").value,
    sort_mode: $("sort").value,
    camera_model: $("camera").value,
    max_num_features: Number($("features").value),
    match_mode: $("match").value,
    overlap: Number($("overlap").value),
    quadratic_overlap: $("quad").checked,
    loop_detection: $("loop").checked,
    single_camera: $("single-cam").checked,
    mapper_backend: $("mapper").value,
    allow_incremental_fallback: $("fallback").checked,
    data_factor: Number($("factor").value),
    max_steps: Number($("steps").value),
    strategy: $("strategy").value,
    run_training: $("do-train").checked,
    save_ply: true,
    max_splats: Number($("max-splats").value) || 0,
    grow_grad2d: Number($("grow-grad").value),
    sh_degree: Number($("sh-degree").value),
    coarse_to_fine: $("coarse-fine").checked,
    opacity_reset_every: Number($("opacity-reset").value) || 0,
    num_threads: Number($("num-threads").value) || 0,
    max_image_size: Number($("max-image-size").value) || 0,
    max_num_matches: Number($("max-matches").value) || 0,
    undistort_images: $("undistort").checked,
    clear_cache: $("clear-cache").checked,
  };
}

function renderQueue(items) {
  const box = $("job-queue");
  if (!box) return;
  if (!items.length) {
    box.className = "queue-list empty";
    box.textContent = "队列空。";
    return;
  }
  box.className = "queue-list";
  box.innerHTML = items.map((job) => {
    const canResume = ["failed", "cancelled"].includes(job.status);
    return `
    <div class="queue-item ${job.id === state.jobId ? "active" : ""}" data-job="${esc(job.id)}" data-project="${esc(job.project_id)}">
      <div>
        <div class="n">${esc(job.project_name || job.source_dir || job.id)}</div>
        <div class="p">${esc(job.source_dir || "")}</div>
      </div>
      <div class="p">${esc(STATUS_LABEL[job.status] || job.status || "")}</div>
      <div class="queue-actions">
        ${canResume ? `<button type="button" class="ghost sm" data-act="resume">续跑</button>` : ""}
        <button type="button" class="ghost sm" data-act="delete">删除</button>
      </div>
    </div>`;
  }).join("");
}

async function refreshQueue() {
  const data = await api("/api/jobs");
  const items = data.jobs || [];
  renderQueue(items);
  const current = items.find((job) => job.id === state.jobId);
  const live = items.find((job) => job.status === "running") || items.find((job) => job.status === "queued");
  const terminal = !current || ["succeeded", "failed", "cancelled"].includes(current.status);
  if (live && terminal && state.jobId !== live.id && state.followLive) {
    if (current && current.status === "failed") {
      toast(`${current.project_name || "上一任务"} 失败，已跳到下一个文件夹`, "bad");
    }
    state.jobId = live.id;
    state.projectId = live.project_id;
    $("logs").innerHTML = "";
    connectLogs(live.id);
    renderQueue(items);
  }
  return items;
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 ** 2) return `${(v / 1024).toFixed(1)} KB`;
  if (v < 1024 ** 3) return `${(v / 1024 ** 2).toFixed(1)} MB`;
  return `${(v / 1024 ** 3).toFixed(2)} GB`;
}

function toast(message, kind = "") {
  const box = $("toast");
  box.hidden = false;
  box.className = `toast ${kind}`;
  box.textContent = message;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { box.hidden = true; }, 3600);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { message: text }; }
  if (!res.ok) {
    const msg = data.message || data.detail || (Array.isArray(data.blockers) ? data.blockers.join("；") : JSON.stringify(data));
    throw new Error(msg);
  }
  return data;
}

function renderDiag(d) {
  const ready = !!d.ready_for_tasks;
  const badge = $("hw-badge");
  const gpuShort = (d.gpu_name || "NVIDIA").replace(/^NVIDIA\s+/i, "");
  badge.textContent = ready
    ? `${gpuShort} · ${d.vram_mb || "?"} MB`
    : ((d.blockers && d.blockers[0]) || "未就绪");
  badge.style.color = ready ? "var(--ok)" : "var(--bad)";

  const used = (d.vram_mb && d.vram_free_mb != null) ? Math.max(d.vram_mb - d.vram_free_mb, 0) : null;
  const usedPct = (used != null && d.vram_mb) ? Math.round((used / d.vram_mb) * 100) : 0;
  const vram = d.vram_mb ? `${d.vram_mb} MB` : "未检测";
  const vramFree = d.vram_free_mb != null ? `已用 ${used} / 空闲 ${d.vram_free_mb} MB` : "特征与训练走 GPU";
  const disk = d.disk_free_gb != null ? `${Number(d.disk_free_gb).toFixed(1)} GB` : "-";
  const tiles = [
    { k: "GPU", v: d.gpu_name || "未检测到 NVIDIA", s: `驱动 ${d.driver_version || "-"}`, ok: d.nvidia_gpu },
    { k: "显存", v: vram, s: vramFree, ok: !!d.vram_mb, meter: usedPct },
    { k: "CPU / 磁盘", v: `${d.cpu_count || "-"} 线程`, s: `工作盘剩余 ${disk}`, ok: true },
    { k: "运行时", v: d.torch?.available ? `PyTorch ${d.torch.version}` : "PyTorch 未安装", s: d.gsplat?.available ? `gsplat ${d.gsplat.version}` : "gsplat 未安装", ok: !!(d.torch?.available && d.gsplat?.available) },
  ];
  $("metrics").innerHTML = tiles.map((t) => `
    <div class="spec ${t.ok ? "ok" : "bad"}">
      <span class="k">${esc(t.k)}</span>
      <strong class="v">${esc(t.v)}</strong>
      <span class="s">${esc(t.s)}</span>
      ${t.meter != null ? `<div class="meter"><i style="width:${t.meter}%"></i></div>` : ""}
    </div>
  `).join("");

  const extra = [];
  extra.push(d.colmap?.available ? `COLMAP 可用` : "COLMAP 缺失");
  extra.push(d.glomap?.available ? "独立 GLOMAP 已安装" : "独立 GLOMAP 未安装，默认仍可用 global_mapper");
  extra.push(`${d.os_name || ""} ${d.arch || ""} · Python ${d.python_version || ""}`.trim());
  $("alerts").innerHTML = [
    ...(d.blockers || []).map((x) => `<div class="alert bad">阻止：${esc(x)}</div>`),
    ...(d.warnings || []).map((x) => `<div class="alert warn">注意：${esc(x)}</div>`),
    `<div class="alert">${esc(extra.join(" · "))}</div>`,
  ].join("");
}

function filmItems(list, offset = 0, projectId = null) {
  return (list || []).map((img, i) => {
    const idx = offset + i;
    const size = img.width && img.height ? `${img.width}×${img.height}` : "";
    const meta = [size, img.exif_time || ""].filter(Boolean).join(" · ");
    const thumb = projectId
      ? `<img src="/api/projects/${esc(projectId)}/thumbs/${idx}" alt="">`
      : `<div class="thumb">${String(idx + 1).padStart(2, "0")}</div>`;
    return `
      <div class="film-item">
        ${thumb}
        <div class="idx">${String(idx + 1).padStart(3, "0")}</div>
        <div class="name" title="${esc(img.source || img.name)}">${esc(img.name)}</div>
        <div class="meta">${esc(meta)}</div>
      </div>
    `;
  }).join("");
}

function renderPreview(project) {
  const count = project.image_count || project.count || 0;
  const head = project.preview || (project.images || []).slice(0, 8);
  const tail = project.preview_tail || ((project.images || []).length > 8 ? (project.images || []).slice(-4) : []);
  const showTail = count > head.length && tail.length;
  const pid = project.id || state.projectId;
  $("preview").classList.remove("empty");
  $("preview").innerHTML = `
    <div class="s">${esc(project.name || pid)} · ${count} 张</div>
    <div class="film">
      ${filmItems(head, 0, pid)}
      ${showTail ? `<div class="ellipsis">中间 ${Math.max(count - head.length - tail.length, 0)} 张</div>${filmItems(tail, Math.max(count - tail.length, 0), pid)}` : ""}
    </div>
  `;
  $("run-hint").textContent = `${count} 张图，按开始会连续做完匹配、重建和训练。`;
  setFlow("import", "ok");
}

const FLOW_GROUPS = {
  match: ["prepare_images", "feature_extract", "feature_match"],
  recon: ["view_graph_calibrator", "global_mapper", "glomap_mapper", "incremental_mapper", "validate_model", "undistort_scene", "export_scene"],
  train: ["train_gsplat"],
};

function setFlow(id, cls) {
  const node = document.querySelector(`[data-flow="${id}"]`);
  if (!node) return;
  node.classList.remove("on", "ok", "bad");
  if (cls) node.classList.add(cls);
}

function groupClass(names, stages) {
  const items = names.map((n) => stages.find((s) => s.name === n)).filter(Boolean);
  if (!items.length) return "";
  if (items.some((s) => s.status === "failed" || s.status === "cancelled")) return "bad";
  if (items.some((s) => s.status === "running")) return "on";
  if (items.every((s) => s.status === "succeeded")) return "ok";
  if (items.some((s) => s.status === "succeeded")) return "on";
  return "";
}

const IDLE_STAGES = [
  "prepare_images", "feature_extract", "feature_match",
  "view_graph_calibrator", "global_mapper", "validate_model",
  "undistort_scene", "export_scene", "train_gsplat",
];

function renderProjects(items) {
  if (!items.length) {
    $("project-list").className = "well empty";
    $("project-list").textContent = "没有记录。";
    return;
  }
  $("project-list").className = "well";
  $("project-list").innerHTML = items.map((p) => `
    <div class="project-item ${p.id === state.projectId ? "active" : ""}" data-id="${esc(p.id)}">
      <div>
        <div class="n">${esc(p.name || p.id)}</div>
        <div class="p">${esc(p.source_dir || "")}</div>
      </div>
      <div class="queue-actions">
        <button type="button" class="ghost sm" data-act="delete">删除</button>
      </div>
    </div>
  `).join("");
}

function renderStages(job) {
  const stages = job.stages && job.stages.length
    ? job.stages
    : IDLE_STAGES.map((name) => ({ name, device: STAGE_META[name].device, status: "pending" }));
  const done = stages.filter((s) => s.status === "succeeded").length;
  const pct = stages.length ? Math.round((done / stages.length) * 100) : 0;
  $("progress-bar").style.width = `${pct}%`;
  $("stages").innerHTML = stages.map((s) => {
    const meta = STAGE_META[s.name] || { label: s.name, device: s.device || "" };
    return `
      <div class="stage ${esc(s.status || "")}">
        <div class="ttl">
          <strong>${esc(meta.label)}</strong>
          <span class="dev ${esc(s.device || meta.device)}">${esc(s.device || meta.device)}</span>
        </div>
        <div class="st">${esc(STATUS_LABEL[s.status] || s.status || "等待")}</div>
      </div>
    `;
  }).join("");

  setFlow("match", groupClass(FLOW_GROUPS.match, stages));
  setFlow("recon", groupClass(FLOW_GROUPS.recon, stages));
  setFlow("train", groupClass(FLOW_GROUPS.train, stages));
  if (state.projectId) setFlow("import", "ok");

  const queued = job.queue?.queued;
  const wait = Array.isArray(queued) ? queued.length : 0;
  if (job.id) {
    $("queue").textContent = `任务 ${job.id} · ${STATUS_LABEL[job.status] || job.status || "待命"} · 当前 ${STAGE_META[job.current_stage]?.label || job.current_stage || "-"} · 进度 ${pct}% · 等待 ${wait}`;
  }
  const chip = $("job-chip");
  chip.textContent = job.status ? `${STATUS_LABEL[job.status] || job.status}${job.current_stage ? " · " + (STAGE_META[job.current_stage]?.label || job.current_stage) : ""}` : "待命";
  chip.style.color = job.status === "failed" || job.status === "cancelled" ? "var(--bad)" : job.status === "running" ? "var(--run)" : job.status === "succeeded" ? "var(--ok)" : "";
}

function logClass(line) {
  if (/error|failed|traceback|oom|exception/i.test(line)) return "err";
  if (/\[GPU\]|feature_extract|feature_match|train_gsplat|CUDA/i.test(line)) return "gpu";
  if (/\[CPU\]|global_mapper|glomap|undistort|view_graph/i.test(line)) return "cpu";
  return "";
}

function appendLog(line) {
  const box = $("logs");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  const row = document.createElement("div");
  row.className = `log-line ${logClass(line)}`;
  row.textContent = line;
  box.appendChild(row);
  if (nearBottom || state.logPinned) box.scrollTop = box.scrollHeight;
}

function disconnectLogs() {
  if (state.ws) { try { state.ws.close(); } catch {} state.ws = null; }
  if (state.es) { try { state.es.close(); } catch {} state.es = null; }
}

function connectLogs(jobId) {
  disconnectLogs();
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/ws/jobs/${jobId}`);
  state.ws = ws;
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      appendLog(data.line || JSON.stringify(data));
    } catch {
      appendLog(ev.data);
    }
  };
  ws.onerror = () => {
    const es = new EventSource(`/api/jobs/${jobId}/logs/stream`);
    state.es = es;
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        appendLog(data.line || JSON.stringify(data));
      } catch {
        appendLog(ev.data);
      }
    };
  };
}

async function refreshJob() {
  await refreshQueue();
  if (!state.jobId) return;
  const job = await api(`/api/jobs/${state.jobId}`);
  renderStages(job);
  if (job.status === "succeeded" && state.projectId) await refreshResults();
  if (job.status === "failed" && job.error && state.toastedJob !== job.id) {
    state.toastedJob = job.id;
    toast(job.error, "bad");
  }
}

async function refreshResults() {
  if (!state.projectId) return;
  const r = await api(`/api/projects/${state.projectId}/results`);
  $("results").classList.remove("empty");
  $("results").innerHTML = `
    <div class="stat-grid">
      <div class="stat"><div class="k">Gaussian 数</div><div class="v">${esc(r.stats.num_gaussians ?? "-")}</div></div>
      <div class="stat"><div class="k">峰值显存</div><div class="v">${esc(r.stats.peak_mem_mb ?? "-")} MB</div></div>
      <div class="stat"><div class="k">训练耗时</div><div class="v">${esc(r.stats.elapsed_sec ?? "-")} s</div></div>
      <div class="stat"><div class="k">PLY 大小</div><div class="v">${esc(r.ply ? fmtBytes(r.ply_size) : "无")}</div></div>
    </div>
    <p class="hint">工作目录：${esc(r.work_dir)}<br>内置查看器是 point sprite 预览，不是精确椭球 splat 渲染。</p>
  `;
  const link = $("viewer-link");
  if (r.viewer_url) {
    link.hidden = false;
    link.href = r.viewer_url;
  } else {
    link.hidden = true;
  }
}

async function loadProjects() {
  const data = await api("/api/projects");
  renderProjects(data.projects || []);
}

async function selectProject(id) {
  const project = await api(`/api/projects/${id}`);
  state.projectId = id;
  state.sourceDirs = project.source_dir ? [project.source_dir] : [];
  renderSources();
  $("work").value = project.work_dir || "";
  state.sourceDir = project.source_dir || "";
  state.workDir = project.work_dir || "";
  const preview = await api(`/api/projects/${id}/preview`, {
    method: "POST",
    body: JSON.stringify({ sort_mode: $("sort").value }),
  });
  renderPreview({ ...project, count: preview.count, images: preview.images });
  await loadProjects();
  const jobs = project.jobs || [];
  const latest = jobs[0];
  if (latest) {
    state.jobId = latest.id;
    $("logs").innerHTML = "";
    renderStages(latest);
    connectLogs(latest.id);
    if (latest.status === "succeeded") await refreshResults();
  }
}

function setBusy(btn, busy, label) {
  btn.disabled = busy;
  if (label) btn.textContent = label;
}

async function ensureProject() {
  const source = state.sourceDirs[0];
  if (!source) throw new Error("请先加入至少一个源图文件夹");
  const work = $("work").value.trim() || null;
  if (state.projectId && state.sourceDir === source && state.workDir === (work || "")) {
    return { id: state.projectId };
  }
  const project = await api("/api/projects", {
    method: "POST",
    body: JSON.stringify({
      name: "",
      source_dir: source,
      work_dir: work,
      sort_mode: $("sort").value,
    }),
  });
  state.projectId = project.id;
  state.sourceDir = source;
  state.workDir = work || "";
  renderPreview(project);
  await loadProjects();
  return project;
}

$("btn-create").onclick = async () => {
  const btn = $("btn-create");
  setBusy(btn, true, "扫描中");
  setBusy($("btn-run"), true);
  try {
    const project = await ensureProject();
    toast(`已读入 ${project.image_count ?? ""} 张图`.trim(), "ok");
  } catch (err) {
    $("preview").classList.remove("empty");
    $("preview").textContent = String(err.message || err);
    toast(String(err.message || err), "bad");
  } finally {
    setBusy(btn, false, "只预览顺序");
    setBusy($("btn-run"), false, "加入队列并开始");
  }
};

$("btn-run").onclick = async () => {
  const btn = $("btn-run");
  setBusy(btn, true, "加入中");
  try {
    const dirs = state.sourceDirs.slice();
    if (!dirs.length) throw new Error("请先加入至少一个源图文件夹");
    const listed = await api("/api/jobs");
    const active = new Set(
      (listed.jobs || [])
        .filter((job) => job.status === "queued" || job.status === "running")
        .map((job) => sourceKey(job.source_dir || ""))
    );
    const fresh = dirs.filter((path) => !active.has(sourceKey(path)));
    if (!fresh.length) {
      toast("这些文件夹已经在队列里", "ok");
      return;
    }
    const work = $("work").value.trim() || null;
    const data = await api("/api/batch", {
      method: "POST",
      body: JSON.stringify({
        ...jobParams(),
        source_dirs: fresh,
        work_dir: work,
      }),
    });
    const created = data.jobs || [];
    const skipped = data.skipped || [];
    state.followLive = true;
    if (created.length) {
      const first = created[0];
      if (!state.jobId || ["succeeded", "failed", "cancelled"].includes((listed.jobs || []).find((j) => j.id === state.jobId)?.status || "")) {
        state.jobId = first.id;
        state.projectId = first.project_id;
        $("logs").innerHTML = "";
        renderStages(first);
        connectLogs(first.id);
      }
      renderPreview({
        id: first.project_id,
        name: first.project_name,
        image_count: first.image_count,
        preview: [],
        preview_tail: [],
      });
    }
    await loadProjects();
    await refreshQueue();
    const skipText = skipped.length ? `，跳过 ${skipped.length} 个空或无效文件夹` : "";
    toast(`已加入队列 ${created.length} 个文件夹${skipText}。失败会自动跳过并继续下一个。`, created.length ? "ok" : "bad");
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    setBusy(btn, false, "加入队列并开始");
  }
};

$("btn-cancel").onclick = async () => {
  if (!state.jobId) return;
  await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST", body: "{}" });
  toast("已请求取消");
};

$("btn-rerun").onclick = async () => {
  if (!state.jobId) return;
  const job = await api(`/api/jobs/${state.jobId}/rerun`, {
    method: "POST",
    body: JSON.stringify({ stage: $("rerun-stage").value }),
  });
  state.jobId = job.id;
  connectLogs(job.id);
  toast("已从选定阶段重跑");
};

$("btn-clear-log").onclick = () => { $("logs").innerHTML = ""; };
$("project-list").onclick = (ev) => {
  const item = ev.target.closest("[data-id]");
  if (!item) return;
  const actBtn = ev.target.closest("[data-act]");
  if (actBtn && actBtn.dataset.act === "delete") {
    ev.preventDefault();
    ev.stopPropagation();
    const projectId = item.dataset.id;
    api(`/api/projects/${projectId}/delete`, { method: "POST", body: "{}" })
      .then(() => {
        if (state.projectId === projectId) {
          state.projectId = null;
          state.jobId = null;
          $("logs").innerHTML = "";
        }
        toast("已删除项目");
        return Promise.all([loadProjects(), refreshQueue()]);
      })
      .catch((err) => toast(err.message, "bad"));
    return;
  }
  selectProject(item.dataset.id).catch((err) => toast(err.message, "bad"));
};

function parentPath(path) {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("\\"), trimmed.lastIndexOf("/"));
  if (idx <= 2) return trimmed.slice(0, 3);
  return trimmed.slice(0, idx);
}

async function openPicker(targetId) {
  state.pickerTarget = targetId;
  state.pickerMode = "dir";
  $("picker").hidden = false;
  const current = $(targetId).value || "F:\\";
  $("picker-path").value = current;
  $("drives").innerHTML = ["C:\\", "D:\\", "E:\\", "F:\\", "G:\\"].map((d) => `<button type="button" class="ghost sm drive">${d}</button>`).join("");
  await loadPicker(current);
}

async function loadPicker(path) {
  $("picker-path").value = path;
  try {
    const body = { path, include_files: state.pickerMode === "file" };
    if (state.pickerSuffix) body.suffix = state.pickerSuffix;
    const data = await api("/api/fs/list", { method: "POST", body: JSON.stringify(body) });
    const entries = data.entries || [];
    const dirs = entries.filter((e) => e.is_dir);
    const files = entries.filter((e) => !e.is_dir);
    const dirHtml = dirs.map((e) => `<div class="picker-item" data-path="${esc(e.path)}" data-dir="1"><span>${esc(e.name)}</span><span class="p">进入</span></div>`).join("");
    const fileHtml = files.map((e) => `<div class="picker-item" data-path="${esc(e.path)}" data-file="1"><span>${esc(e.name)}</span><span class="p">选择</span></div>`).join("");
    $("picker-list").innerHTML = (dirHtml + fileHtml) || `<div class="empty">没有可选项</div>`;
  } catch (err) {
    $("picker-list").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
}

async function pickFoldersNative() {
  toast("正在打开系统文件夹窗口，请看任务栏或其它屏幕");
  const data = await api("/api/fs/pick-folders", { method: "POST", body: "{}" });
  return data.paths || [];
}

async function pickFolderNative() {
  toast("正在打开系统文件夹窗口，请看任务栏或其它屏幕");
  const data = await api("/api/fs/pick-folder", { method: "POST", body: "{}" });
  return data.path || "";
}

function collectDropPayload(dt) {
  const paths = [];
  if (!dt) return { paths, text: "" };
  for (const file of dt.files || []) {
    if (file.path) paths.push(file.path);
    else if (file.name && /[\\/]/.test(file.name)) paths.push(file.name);
  }
  if (dt.items) {
    for (const item of dt.items) {
      const file = item.getAsFile && item.getAsFile();
      if (file && file.path) paths.push(file.path);
    }
  }
  const text = [
    dt.getData("text/uri-list"),
    dt.getData("text/plain"),
    dt.getData("application/x-moz-file"),
  ].filter(Boolean).join("\n");
  return { paths, text };
}

$("btn-browse-source").onclick = async () => {
  try {
    const paths = await pickFoldersNative();
    if (paths.length) addSources(paths);
    else toast("没有选择文件夹");
  } catch (err) {
    toast(err.message, "bad");
  }
};
$("btn-browse-work").onclick = async () => {
  try {
    const path = await pickFolderNative();
    if (path) $("work").value = path;
  } catch (err) {
    toast(err.message, "bad");
  }
};
$("btn-clear-sources").onclick = () => {
  state.sourceDirs = [];
  renderSources();
};
$("source-chips").onclick = (ev) => {
  const btn = ev.target.closest("[data-remove]");
  if (!btn) return;
  state.sourceDirs.splice(Number(btn.dataset.remove), 1);
  renderSources();
};
$("job-queue").onclick = (ev) => {
  const actBtn = ev.target.closest("[data-act]");
  const item = ev.target.closest("[data-job]");
  if (!item) return;
  if (actBtn) {
    ev.preventDefault();
    ev.stopPropagation();
    const jobId = item.dataset.job;
    if (actBtn.dataset.act === "resume") {
      api(`/api/jobs/${jobId}/resume`, { method: "POST", body: "{}" })
        .then(() => {
          toast("已加入队列续跑");
          state.jobId = jobId;
          state.projectId = item.dataset.project;
          state.followLive = true;
          $("logs").innerHTML = "";
          connectLogs(jobId);
          return refreshQueue();
        })
        .catch((err) => toast(err.message, "bad"));
      return;
    }
    if (actBtn.dataset.act === "delete") {
      api(`/api/jobs/${jobId}/delete`, { method: "POST", body: "{}" })
        .then(() => {
          if (state.jobId === jobId) {
            state.jobId = null;
            $("logs").innerHTML = "";
          }
          toast("已删除任务");
          return refreshQueue();
        })
        .catch((err) => toast(err.message, "bad"));
      return;
    }
  }
  state.jobId = item.dataset.job;
  state.projectId = item.dataset.project;
  state.followLive = false;
  $("logs").innerHTML = "";
  connectLogs(state.jobId);
  refreshJob().catch((err) => toast(err.message, "bad"));
};

function bindDrop(el) {
  if (!el) return;
  const stop = (ev) => { ev.preventDefault(); ev.stopPropagation(); };
  el.addEventListener("dragenter", (ev) => { stop(ev); el.classList.add("drag"); });
  el.addEventListener("dragover", (ev) => { stop(ev); el.classList.add("drag"); });
  el.addEventListener("dragleave", (ev) => {
    if (!el.contains(ev.relatedTarget)) el.classList.remove("drag");
  });
  el.addEventListener("drop", async (ev) => {
    stop(ev);
    el.classList.remove("drag");
    const { paths, text } = collectDropPayload(ev.dataTransfer);
    try {
      const data = await api("/api/fs/drop", { method: "POST", body: JSON.stringify({ text, paths }) });
      if ((data.paths || []).length) {
        addSources(data.paths);
        return;
      }
      toast("浏览器没给出路径，改为打开系统窗口");
      const picked = await pickFoldersNative();
      if (picked.length) addSources(picked);
      else toast("没有选择文件夹");
    } catch (err) {
      toast(err.message, "bad");
    }
  });
}

document.addEventListener("dragover", (ev) => ev.preventDefault());
document.addEventListener("drop", (ev) => ev.preventDefault());

$("picker-close").onclick = () => { $("picker").hidden = true; };
$("picker-go").onclick = () => loadPicker($("picker-path").value);
$("picker-up").onclick = () => loadPicker(parentPath($("picker-path").value));
$("drives").onclick = (ev) => {
  const btn = ev.target.closest(".drive");
  if (btn) loadPicker(btn.textContent);
};
$("picker-list").onclick = (ev) => {
  const item = ev.target.closest("[data-path]");
  if (!item) return;
  if (item.dataset.file) {
    importRuntimeFile(item.dataset.path);
    return;
  }
  loadPicker(item.dataset.path);
};
$("picker-choose").onclick = () => {
  if (state.pickerMode === "file") return;
  if (state.pickerTarget) $(state.pickerTarget).value = $("picker-path").value;
  $("picker").hidden = true;
};
$("picker").addEventListener("click", (ev) => {
  if (ev.target === $("picker")) $("picker").hidden = true;
});

function bindNotes() {
  document.querySelectorAll(".help").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      btn.focus();
    });
  });
}

const PRESET_FILL = {
  fast: {
    factor: 8,
    steps: 7000,
    maxSplats: 1800000,
    grow: 0.0009,
    sh: 0,
    coarse: true,
    opacity: 0,
    sfm: "low",
    lead: "缩小 8 倍 · 7 千步 · 约 180 万高斯 · COLMAP 低",
  },
  balanced: {
    factor: 4,
    steps: 15000,
    maxSplats: 4500000,
    grow: 0.0002,
    sh: 1,
    coarse: true,
    opacity: 0,
    sfm: "medium",
    lead: "缩小 4 倍 · 1.5 万步 · 约 450 万高斯 · COLMAP 中 · 12GB 默认",
  },
  quality: {
    factor: 2,
    steps: 30000,
    maxSplats: 8400000,
    grow: 0.0002,
    sh: 3,
    coarse: true,
    opacity: 3000,
    sfm: "high",
    lead: "缩小 2 倍 · 3 万步 · 约 840 万高斯 · COLMAP 高",
  },
};

function applyPreset(name) {
  const preset = PRESET_FILL[name];
  if (!preset) return;
  state.applyingPreset = true;
  $("preset").value = name;
  $("factor").value = preset.factor;
  $("steps").value = preset.steps;
  $("max-splats").value = preset.maxSplats;
  $("grow-grad").value = preset.grow;
  $("sh-degree").value = String(preset.sh);
  $("coarse-fine").checked = preset.coarse;
  $("opacity-reset").value = preset.opacity;
  $("sfm-quality").value = preset.sfm;
  applySfmQuality();
  $("preset-lead").textContent = preset.lead;
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.preset === name);
  });
  state.applyingPreset = false;
}

function markPresetCustom() {
  if (state.applyingPreset) return;
  document.querySelectorAll(".preset-btn").forEach((btn) => btn.classList.remove("on"));
  $("preset-lead").textContent = "已手动改参数 · 点低/中/高可重新套用";
}

function applySfmQuality() {
  const sel = $("sfm-quality");
  if (!sel) return;
  const opt = sel.selectedOptions[0];
  if (!opt || sel.value === "custom") return;
  $("features").value = opt.dataset.features || $("features").value;
  $("max-image-size").value = opt.dataset.size || "0";
  $("max-matches").value = opt.dataset.matches || "0";
}

async function importRuntimeFile(path) {
  if (!state.runtimeArtifactId) return;
  try {
    await api("/api/runtime/import", {
      method: "POST",
      body: JSON.stringify({ artifact_id: state.runtimeArtifactId, path }),
    });
    $("picker").hidden = true;
    toast("已导入，继续安装", "ok");
    await refreshRuntime();
  } catch (err) {
    toast(err.message, "bad");
  }
}

function runtimeStateLabel(item) {
  if (item.installed || item.state === "done") return "完成";
  if (item.state === "downloading") return `下载 ${item.progress || 0}%`;
  if (item.state === "verifying") return "校验";
  if (item.state === "installing") return "安装";
  if (item.state === "failed") return "失败";
  return "等待";
}

function renderRuntime(data) {
  const modal = $("runtime-setup");
  if (!modal) return;
  const needed = !!data.needed || data.phase === "downloading" || data.phase === "installing" || data.phase === "needs_manual" || data.restart_required;
  if (needed) modal.hidden = false;
  else if (data.ready && data.phase === "ready" && !data.restart_required) {
    if (!state.runtimeForced) modal.hidden = true;
  }
  const items = data.artifacts || [];
  const done = items.filter((x) => x.installed || x.state === "done").length;
  const pct = items.length ? Math.round((done / items.length) * 100) : 0;
  $("runtime-bar").style.width = `${data.phase === "ready" ? 100 : pct}%`;
  $("runtime-msg").textContent = data.message || data.error || (data.ready ? "运行时已就绪" : "正在准备运行时");
  $("runtime-items").innerHTML = items.map((item) => `
    <div class="rt-item ${item.state === "failed" ? "failed" : (item.installed ? "done" : "")}">
      <div class="top"><strong>${esc(item.label)}</strong><span class="st">${esc(runtimeStateLabel(item))}</span></div>
      ${item.error ? `<div class="err">${esc(item.error)}</div>` : ""}
    </div>
  `).join("");
  const failed = items.filter((x) => x.state === "failed" || (data.manual_needed && !x.installed && x.required));
  const showManual = data.phase === "needs_manual" || data.manual_needed;
  $("runtime-manual").hidden = !showManual;
  if (showManual) {
    $("runtime-manual-list").innerHTML = (failed.length ? failed : items.filter((x) => !x.installed)).map((item) => `
      <div class="rt-card">
        <strong>${esc(item.label)}</strong>
        <code>${esc(item.url)}</code>
        <span class="sha">${item.sha256 ? `SHA256 ${esc(item.sha256)}` : "保存为 " + esc(item.filename)}</span>
        <div class="row">
          <button type="button" class="ghost sm" data-open-url="${esc(item.url)}">浏览器打开</button>
          <button type="button" class="ghost sm" data-copy-url="${esc(item.url)}">复制地址</button>
          <button type="button" class="primary sm" data-pick="${esc(item.id)}" data-name="${esc(item.filename)}">选择已下载的文件</button>
        </div>
      </div>
    `).join("");
  }
  $("runtime-relaunch").hidden = !data.restart_required && data.phase !== "ready";
  $("runtime-close").hidden = !data.ready || !!data.needed;
}

async function refreshRuntime() {
  const data = await api("/api/runtime/status");
  state.runtime = data;
  renderRuntime(data);
  return data;
}

async function bootRuntime() {
  try {
    const health = await api("/api/health");
    if (health.fake) return;
    const data = await refreshRuntime();
    if (data.needed && !data.busy) {
      await api("/api/runtime/install", { method: "POST", body: "{}" }).catch(() => {});
      await refreshRuntime();
    }
    state.runtimeTimer = setInterval(() => {
      refreshRuntime().catch(() => {});
    }, 1200);
  } catch {}
}

$("runtime-retry").onclick = async () => {
  await api("/api/runtime/install", { method: "POST", body: "{}" });
  toast("已重新开始自动下载");
};
$("runtime-open-cache").onclick = async () => {
  try {
    const data = await api("/api/runtime/open-cache", { method: "POST", body: "{}" });
    toast(`缓存目录：${data.path}`);
  } catch (err) {
    toast(err.message, "bad");
  }
};
$("runtime-relaunch").onclick = async () => {
  await api("/api/runtime/relaunch", { method: "POST", body: "{}" });
  toast("正在重启…");
};
$("runtime-close").onclick = () => { $("runtime-setup").hidden = true; };
$("runtime-manual-list").onclick = (ev) => {
  const open = ev.target.closest("[data-open-url]");
  if (open) {
    window.open(open.dataset.openUrl, "_blank", "noopener");
    return;
  }
  const copy = ev.target.closest("[data-copy-url]");
  if (copy) {
    navigator.clipboard.writeText(copy.dataset.copyUrl).then(() => toast("已复制地址", "ok"));
    return;
  }
  const pick = ev.target.closest("[data-pick]");
  if (pick) {
    state.runtimeArtifactId = pick.dataset.pick;
    (async () => {
      try {
        const data = await api("/api/fs/pick-file", { method: "POST", body: "{}" });
        if (data.path) {
          await importRuntimeFile(data.path);
          return;
        }
      } catch (err) {
        toast(err.message, "bad");
        return;
      }
      state.pickerMode = "file";
      state.pickerSuffix = "";
      const name = pick.dataset.name || "";
      const dot = name.lastIndexOf(".");
      if (dot >= 0) state.pickerSuffix = name.slice(dot);
      $("picker").hidden = false;
      $("picker-path").value = "F:\\";
      loadPicker("F:\\");
    })();
  }
};

async function boot() {
  bindNotes();
  renderSources();
  bindDrop($("source-drop"));
  renderStages({});
  document.querySelector(".preset-bar")?.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".preset-btn");
    if (btn?.dataset.preset) applyPreset(btn.dataset.preset);
  });
  applyPreset("balanced");
  $("sfm-quality").onchange = () => {
    if ($("sfm-quality").value === "custom") {
      markPresetCustom();
      return;
    }
    applySfmQuality();
    if (!state.applyingPreset) markPresetCustom();
  };
  [
    "factor", "steps", "max-splats", "grow-grad", "sh-degree", "opacity-reset", "strategy",
  ].forEach((id) => {
    $(id)?.addEventListener("input", markPresetCustom);
    $(id)?.addEventListener("change", markPresetCustom);
  });
  ["features", "max-image-size", "max-matches"].forEach((id) => {
    $(id)?.addEventListener("input", () => {
      if (!state.applyingPreset) $("sfm-quality").value = "custom";
      markPresetCustom();
    });
  });
  $("coarse-fine")?.addEventListener("change", markPresetCustom);
  await bootRuntime();
  try {
    renderDiag(await api("/api/diagnostics"));
  } catch (err) {
    $("metrics").innerHTML = `<div class="spec bad"><span class="k">诊断</span><strong class="v">${esc(err.message)}</strong></div>`;
  }
  try { await loadProjects(); } catch {}
  try {
    const jobs = await api("/api/jobs");
    const running = (jobs.jobs || []).find((j) => j.status === "running" || j.status === "queued");
    if (running) {
      state.jobId = running.id;
      state.projectId = running.project_id;
      renderStages(running);
      connectLogs(running.id);
    }
  } catch {}
  setInterval(() => refreshJob().catch(() => {}), 2000);
}

boot();
