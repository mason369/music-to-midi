/* Project state lives on the server. Selection never starts inference. */
class MusicProjects {
  constructor(root) {
    this.root = root;
    this.project = null;
    this.songId = null;
    this.timer = null;
    this.pending = false;
    this.stems = {};
    this.apiBase = state.apiBase;
    this.abort = new AbortController();
  }
  el(selector) {
    return this.root.querySelector(selector);
  }
  async request(path, method = "GET", body) {
    const options = { method };
    if (body instanceof FormData) options.body = body;
    else if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    return api(`/api/v1/projects${path}`, options);
  }
  async guard(fn) {
    try {
      return await fn();
    } catch (error) {
      this.el(".project-status").textContent =
        `${this.words.error}: ${error.message}`;
      toast(error.message, "error");
    }
  }
  async init() {
    this.allLabels = (await this.request("/capabilities")).labels;
    this.words = this.allLabels[state.language];
    const w = this.words;
    this.root.innerHTML = `<h2>${escapeHtml(w.title)}</h2><p>${escapeHtml(w.empty)}</p>
      <input class="project-name" aria-label="${escapeHtml(w.name)}" value="${escapeHtml(w.title)}">
      <div class="project-toolbar">${["new", "refresh", "import", "export", "add", "profile_load", "profile_save"].map((k) => `<button data-action="${k}" type="button">${escapeHtml(w[k])}</button>`).join("")}</div>
      <select class="project-list" aria-label="${escapeHtml(w.open)}"></select><select class="project-song" aria-label="${escapeHtml(w.song)}"></select>
      <fieldset class="project-stems"><legend>${escapeHtml(w.stems)}</legend><div></div></fieldset>
      <div class="project-toolbar"><select class="project-scope"><option value="selected">${escapeHtml(w.selected)}</option><option value="all">${escapeHtml(w.all)}</option></select>
      ${["apply", "result", "run", "stop"].map((k) => `<button data-action="${k}" type="button">${escapeHtml(w[k])}</button>`).join("")}</div>
      <p class="project-status" aria-live="polite"></p><div class="project-song-list"></div>
      <input class="project-audio" type="file" accept="audio/*" multiple hidden><input class="project-import" type="file" accept=".mtmproject" hidden><input class="project-profile" type="file" accept=".json" hidden>`;
    this.root
      .querySelectorAll("[data-action]")
      .forEach((b) =>
        b.addEventListener("click", () =>
          this.guard(() => this.action(b.dataset.action)),
        ),
      );
    this.el(".project-list").addEventListener("change", () =>
      this.guard(() => this.open(this.el(".project-list").value)),
    );
    this.el(".project-song").addEventListener("change", () => {
      this.songId = this.el(".project-song").value;
      this.loadSettings();
    });
    this.el(".project-audio").addEventListener("change", (e) =>
      this.guard(() => this.addFiles(e.target.files)),
    );
    this.el(".project-import").addEventListener("change", (e) =>
      this.guard(() => this.import(e.target.files[0])),
    );
    this.el(".project-profile").addEventListener("change", (e) =>
      this.guard(async () => {
        const f = e.target.files[0];
        if (f) await this.configure(JSON.parse(await f.text()));
      }),
    );
    $("#modeSelect").addEventListener("change", () => this.renderStems(), {
      signal: this.abort.signal,
    });
    this.saveQueue = Promise.resolve();
    const persist = (event) => {
      const row = event.target.closest(".track-row");
      if (!row || !this.project || this.project.active) return;
      if (event.type === "click" && !event.target.matches(".mute,.solo"))
        return;
      this.saveQueue = this.saveQueue.then(() =>
        this.guard(() => this.saveMixer()),
      );
    };
    $("#trackStack").addEventListener("change", persist, {
      signal: this.abort.signal,
    });
    $("#trackStack").addEventListener("click", persist, {
      signal: this.abort.signal,
    });
    await this.refresh();
    const saved = localStorage.getItem(`musicToMidiProject:${state.apiBase}`);
    if (
      saved &&
      [...this.el(".project-list").options].some((o) => o.value === saved)
    )
      await this.open(saved);
  }
  async refresh() {
    const projects = await this.request("");
    const select = this.el(".project-list");
    select.replaceChildren(new Option(this.words.open, ""));
    for (const p of projects)
      select.add(
        new Option(`${p.name} · ${this.words[p.status]}`, p.storage_id),
      );
    if (this.project) select.value = this.project.storage_id;
    return projects;
  }
  translate() {
    this.words = this.allLabels[state.language];
    const w = this.words;
    this.root.querySelector("h2").textContent = w.title;
    this.root.querySelector("p").textContent = w.empty;
    this.root.setAttribute("aria-label", w.title);
    this.root
      .querySelectorAll("[data-action]")
      .forEach((b) => (b.textContent = w[b.dataset.action]));
    this.el(".project-stems legend").textContent = w.stems;
    this.el(".project-scope").options[0].text = w.selected;
    this.el(".project-scope").options[1].text = w.all;
    this.el(".project-song").setAttribute("aria-label", w.song);
    this.el(".project-list").setAttribute("aria-label", w.open);
    this.render();
    this.renderStems();
  }
  async open(id) {
    if (!id) return;
    this.project = await this.request(`/${encodeURIComponent(id)}`);
    this.el(".project-list").value = id;
    localStorage.setItem(`musicToMidiProject:${state.apiBase}`, id);
    this.render();
    this.loadSettings();
    if (this.project.active) this.poll();
  }
  song() {
    return this.project?.songs.find((s) => s.id === this.songId);
  }
  render() {
    const p = this.project;
    if (!p) return;
    const select = this.el(".project-song");
    select.replaceChildren();
    for (const s of p.songs)
      select.add(new Option(`${s.name} · ${this.words[s.status]}`, s.id));
    if (!p.songs.some((s) => s.id === this.songId))
      this.songId = p.songs[0]?.id || null;
    select.value = this.songId || "";
    const x = p.progress || {};
    this.el(".project-status").textContent =
      `${p.name} · ${this.words[p.status]} · ${p.counts.succeeded}/${p.counts.total}\n${x.index || 0}/${x.total || p.counts.total} · ${x.percent || 0}% ${x.message || ""}\n${p.service_error || p.error || ""}`;
    this.el(".project-song-list").replaceChildren(
      ...p.songs.map((s, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = `${i + 1}. ${s.name} · ${this.words[s.status]}`;
        b.addEventListener("click", () => {
          this.songId = s.id;
          select.value = s.id;
          this.loadSettings();
        });
        return b;
      }),
    );
    this.root.querySelectorAll("button,input,select").forEach((c) => {
      c.disabled =
        Boolean(p.active || this.pending) && c.dataset.action !== "stop";
    });
    updateReadyState();
    if (p.active) {
      $("#startButton").disabled = true;
      $("#stopButton").disabled = false;
      $("#readyText").textContent = this.el(".project-status").textContent;
    }
  }
  loadSettings() {
    const workflow = this.song()?.workflow || this.project?.workflow;
    if (!workflow) return;
    const o = workflow.primary;
    state.selectedMode = o.processing_mode;
    renderModes();
    for (const [selector, key] of [
      ["#backendSelect", "transcription_backend"],
      ["#yourmt3Select", "yourmt3_model"],
      ["#muscriptorSelect", "muscriptor_model"],
      ["#trackModeSelect", "midi_track_mode"],
      ["#tempoModeSelect", "tempo_mode"],
      ["#muscriptorProcessingChainSelect", "muscriptor_processing_chain"],
      ["#quantizeGridSelect", "quantize_grid"],
    ])
      $(selector).value = o[key];
    $("#customBpm").value = o.custom_bpm || "";
    $("#quantizeNotes").checked = o.quantize_notes;
    setSelectedValues(
      $("#muscriptorInstrumentsSelect"),
      o.muscriptor_instruments || [],
    );
    this.stems = structuredClone(workflow.stems);
    updateConditionalControls();
    updateStartLabel();
    this.renderStems();
    const song = this.song();
    if (song) {
      state.audioFile = { name: song.name, size: song.source.size };
      state.sourceObjectUrl = artifactUrl(this.fileAsset(song.source));
      $("#fileInspector").hidden = false;
      $("#fileName").textContent = song.name;
      $("#fileMeta").textContent = formatBytes(song.source.size);
      updateReadyState();
    }
  }
  allowedStems() {
    return (
      {
        vocal_split: ["vocals", "accompaniment"],
        six_stem_split: ["bass", "drums", "guitar", "piano", "vocals", "other"],
      }[state.selectedMode] || []
    );
  }
  renderStems() {
    const stems = this.allowedStems();
    this.el(".project-stems").hidden = !stems.length;
    const target = this.el(".project-stems div");
    target.replaceChildren();
    for (const stem of stems) {
      const row = document.createElement("label");
      row.className = "project-stem";
      const check = document.createElement("input");
      check.type = "checkbox";
      check.setAttribute("aria-label", stem);
      check.checked = Boolean(this.stems[stem]);
      const text = document.createElement("span");
      text.textContent = stem;
      const route = document.createElement("select");
      route.setAttribute("aria-label", `${stem} ${this.words.route}`);
      for (const item of state.capabilities.manual_midi_routes)
        route.add(new Option(item.label, item.id));
      if (this.stems[stem]) route.value = this.stems[stem].route;
      const update = () => {
        if (check.checked)
          this.stems[stem] = {
            ...(this.stems[stem] || {}),
            route: route.value,
          };
        else delete this.stems[stem];
      };
      check.addEventListener("change", update);
      route.addEventListener("change", update);
      route.addEventListener("wheel", (e) => e.preventDefault(), {
        passive: false,
      });
      row.append(check, text, route);
      target.append(row);
    }
  }
  workflow() {
    return {
      name: this.words.title,
      primary: buildInferenceOptions(),
      stems: Object.fromEntries(
        Object.entries(this.stems).filter(([s]) =>
          this.allowedStems().includes(s),
        ),
      ),
    };
  }
  async configure(workflow, ids) {
    if (!this.project) throw new Error(this.words.empty);
    const song_ids =
      ids === undefined
        ? this.el(".project-scope").value === "all"
          ? null
          : [this.songId].filter(Boolean)
        : ids;
    this.project = await this.request(
      `/${this.project.storage_id}/workflow`,
      "PUT",
      { workflow, song_ids, revision: this.project.revision },
    );
    this.render();
    this.loadSettings();
  }
  async addFiles(files) {
    const list = [...files];
    if (!list.length) return;
    if (this.project?.active || this.pending)
      throw new Error(this.words.processing_hint);
    if (!this.project)
      this.project = await this.request("", "POST", {
        name: this.el(".project-name").value,
        workflow: this.workflow(),
      });
    this.pending = true;
    this.render();
    try {
      for (const file of list) {
        const form = new FormData();
        form.append("file", file, file.name);
        this.project = await this.request(
          `/${this.project.storage_id}/songs`,
          "POST",
          form,
        );
      }
      localStorage.setItem(
        `musicToMidiProject:${state.apiBase}`,
        this.project.storage_id,
      );
      this.songId =
        this.project.songs.find((s) => s.name === list[0].name)?.id ||
        this.project.songs[0]?.id;
      await this.refresh();
      this.loadSettings();
    } finally {
      this.pending = false;
      this.render();
    }
  }
  async start(ids = null, track_ids = null, primary_only = false) {
    if (!this.project) throw new Error(this.words.empty);
    await this.saveQueue;
    this.project = await this.request(
      `/${this.project.storage_id}/run`,
      "POST",
      { song_ids: ids, track_ids, primary_only },
    );
    this.render();
    this.poll();
  }
  async saveMixer() {
    const tracks = state.tracks.filter(
      (t) =>
        t.projectId === this.project.storage_id &&
        t.projectSongId === this.songId,
    );
    if (!tracks.length) return;
    const song = this.song(),
      stems = structuredClone(song.workflow.stems),
      mixer = structuredClone(song.mixer);
    for (const track of tracks) {
      const id = track.serverTrackId;
      mixer[id] = {
        muted: track.muted,
        solo: track.solo,
        gain_db: track.gainDb,
        offset: track.offset,
      };
      if (track.midiEnabled && track.route)
        stems[id] = {
          ...(stems[id] || {}),
          route: track.route,
          muscriptor_instruments: track.muscriptorInstruments,
        };
      else delete stems[id];
    }
    this.project = await this.request(
      `/${this.project.storage_id}/songs/${song.id}/state`,
      "PUT",
      { revision: this.project.revision, stems, mixer },
    );
    this.stems = structuredClone(stems);
    this.render();
    this.renderStems();
  }
  async addTracks(files) {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file, file.name);
      this.project = await this.request(
        `/${this.project.storage_id}/songs/${this.songId}/tracks`,
        "POST",
        form,
      );
    }
    await this.openResult();
  }
  poll() {
    if (this.timer) clearTimeout(this.timer);
    const id = this.project.storage_id;
    this.timer = setTimeout(
      () =>
        this.guard(async () => {
          this.project = await this.request(`/${id}`);
          this.render();
          if (this.project.active) this.poll();
          else {
            this.timer = null;
            if (this.project.service_error)
              throw new Error(this.project.service_error);
            await this.openResult();
          }
        }),
      700,
    );
  }
  fileAsset(a) {
    return {
      ...a,
      name: a.path.split("/").pop(),
      download_url: `/api/v1/projects/${this.project.storage_id}/files/${a.path.split("/").map(encodeURIComponent).join("/")}`,
    };
  }
  job(song, step, stem = null) {
    return {
      id: `project-${this.project.storage_id}-${song.id}-${step.signature}`,
      status: step.status,
      request: step.options,
      result: step.result,
      artifacts: step.artifacts.map((a) => this.fileAsset(a)),
      original_filename: song.name,
      parent_job_id: null,
      track_id: stem,
      project_id: this.project.storage_id,
      project_song_id: song.id,
    };
  }
  async openResult() {
    const song = this.song(),
      step = song?.checkpoints[song.primary_key];
    if (!step || step.status !== "succeeded") return;
    const job = this.job(song, step);
    state.currentJob = job;
    renderPrimaryResult(job, { restoreTracks: false });
    $("#deleteJob").hidden = true;
    if (step.result.manual_midi_required) {
      stopTransport();
      state.tracks = [
        ...step.artifacts.filter((a) => a.kind === "audio_track"),
        ...Object.values(song.extra_tracks || {}),
      ].map((a, i) => {
        const option = song.workflow.stems[a.track_id],
          view = song.mixer[a.track_id] || {};
        const track = Object.assign(
          makeTrack({
            id: a.track_id,
            name: a.track_id,
            color: TRACK_COLORS[i % TRACK_COLORS.length],
            audioUrl: artifactUrl(this.fileAsset(a)),
            fileName: a.path.split("/").pop(),
            serverTrackId: a.track_id,
          }),
          {
            projectId: this.project.storage_id,
            projectSongId: song.id,
            muted: view.muted || false,
            solo: view.solo || false,
            gainDb: view.gain_db || 0,
            offset: view.offset || 0,
            route: option?.route || "",
            midiEnabled: Boolean(option),
            muscriptorInstruments: option?.muscriptor_instruments || [],
          },
        );
        const child = song.checkpoints[song.track_keys[a.track_id]];
        if (child) {
          track.midiJob = this.job(song, child, a.track_id);
          const midi = child.artifacts.find((x) => x.kind === "midi");
          track.midiArtifact = midi ? this.fileAsset(midi) : null;
          track.statusKey =
            child.status === "succeeded"
              ? "track.midi_completed"
              : child.status === "failed"
                ? "track.previous_failed"
                : "track.previous_cancelled";
          track.statusVars = { error: child.error || "" };
        }
        return track;
      });
      renderMixer();
      await Promise.all(state.tracks.map(loadTrackAudio));
      redrawWaveforms();
    }
  }
  async convertTrack(track, options) {
    if (this.project.active) {
      this.project = await this.request(
        `/${this.project.storage_id}/cancel`,
        "POST",
      );
      return;
    }
    const song = this.project.songs.find((s) => s.id === track.projectSongId),
      workflow = structuredClone(song.workflow);
    workflow.stems[track.serverTrackId] = options;
    await this.configure(workflow, [song.id]);
    await this.start([song.id], [track.serverTrackId]);
  }
  async import(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file, file.name);
    this.project = await this.request("/import", "POST", form);
    await this.refresh();
    await this.open(this.project.storage_id);
  }
  async action(a) {
    if (a === "new") {
      this.project = await this.request("", "POST", {
        name: this.el(".project-name").value,
        workflow: this.workflow(),
      });
      await this.refresh();
      await this.open(this.project.storage_id);
    } else if (a === "refresh") {
      await this.refresh();
      if (this.project) await this.open(this.project.storage_id);
    } else if (a === "add") this.el(".project-audio").click();
    else if (a === "import") this.el(".project-import").click();
    else if (a === "profile_load") this.el(".project-profile").click();
    else if (a === "apply") await this.configure(this.workflow());
    else if (a === "run") await this.start();
    else if (a === "result") await this.openResult();
    else if (a === "stop" && this.project) {
      this.project = await this.request(
        `/${this.project.storage_id}/cancel`,
        "POST",
      );
      this.render();
    } else if (a === "export" && this.project) {
      const r = await fetch(
        `${state.apiBase}/api/v1/projects/${this.project.storage_id}/archive`,
      );
      if (!r.ok) throw new Error(await r.text());
      this.download(await r.blob(), `${this.project.name}.mtmproject`);
    } else if (a === "profile_save")
      this.download(
        new Blob([JSON.stringify(this.workflow(), null, 2)], {
          type: "application/json",
        }),
        "workflow.json",
      );
  }
  download(blob, name) {
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
let musicProjects = null;
function resultEndpoint(jobId) {
  return `${jobId.startsWith("project-") ? "/api/v1/project-results" : "/api/v1/jobs"}/${encodeURIComponent(jobId)}`;
}
async function initializeProjects() {
  if (!musicProjects || musicProjects.apiBase !== state.apiBase) {
    if (musicProjects) {
      clearTimeout(musicProjects.timer);
      musicProjects.abort.abort();
    }
    const instance = new MusicProjects(document.querySelector("#projectPanel"));
    await instance.init();
    musicProjects = instance;
  }
}
async function setAudioFiles(files) {
  if (!files.length) return;
  try {
    await initializeProjects();
    await musicProjects.addFiles(files);
    setAudioFile(files[0]);
  } catch (error) {
    toast(error.message, "error");
  }
}
