"""Shared SoundFont controls for Gradio and the standalone browser client."""

SOUNDFONT_JS = r"""
(function () {
  "use strict";
  if (window.MidiSoundFontPicker) return;
  function node(tag, text) {
    var element = document.createElement(tag);
    if (text) element.textContent = text;
    return element;
  }
  var style = node('style');
  style.textContent = '.soundfont-panel{margin:10px 0;padding:10px;border:1px solid #365f8d;border-radius:7px;min-width:0}.soundfont-panel summary{cursor:pointer;font-weight:600}.soundfont-panel p{font-size:12px;line-height:1.6;overflow-wrap:anywhere}.soundfont-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:end}.soundfont-controls label{display:flex;flex-direction:column;gap:4px;min-width:0;max-width:100%;font-size:12px}.soundfont-controls select,.soundfont-controls input{max-width:100%;min-width:0;width:210px}.soundfont-controls select,.soundfont-controls button{padding:7px;border:1px solid #456582;border-radius:5px;background:#172941;color:#e0e0e0;min-height:34px}.soundfont-panel audio{max-width:100%;width:100%;margin-top:8px}';
  document.head.appendChild(style);
  function Picker(options) {
    var self = this;
    this.options = options;
    this.strings = options.strings;
    this.libraries = new Map();
    this.assignments = new Map();
    this.root = node('details'); this.root.className = 'soundfont-panel';
    this.root.appendChild(node('summary', this.strings.title));
    this.root.appendChild(node('p', this.strings.notice));
    var controls = node('div'); controls.className = 'soundfont-controls';
    this.root.appendChild(controls);
    function select(label) {
      var wrap = node('label', label), field = node('select');
      field.addEventListener('wheel', function (event) { event.preventDefault(); }, { passive: false });
      wrap.appendChild(field); controls.appendChild(wrap); return field;
    }
    this.library = select(this.strings.library);
    this.library.appendChild(new Option(this.strings.default, ''));
    var fileLabel = node('label', this.strings.file);
    this.file = node('input'); this.file.type = 'file'; this.file.accept = '.sf2,.sf3';
    fileLabel.appendChild(this.file); controls.appendChild(fileLabel);
    this.importButton = node('button', this.strings.import); this.importButton.type = 'button';
    controls.appendChild(this.importButton);
    this.source = select(this.strings.source);
    this.preset = select(this.strings.preset);
    this.applyButton = node('button', this.strings.apply); this.applyButton.type = 'button';
    controls.appendChild(this.applyButton);
    this.status = node('p'); this.status.setAttribute('role', 'status'); this.root.appendChild(this.status);
    this.library.onchange = function () { self.assignments.clear(); self.refresh(); self.status.textContent = self.strings.pending; };
    this.source.onchange = function () { self.showPreset(); };
    this.preset.onchange = function () {
      if (self.preset.value) self.assignments.set(self.source.value, JSON.parse(self.preset.value));
      else self.assignments.delete(self.source.value);
      self.status.textContent = self.strings.pending;
    };
    this.root.ontoggle = function () { if (self.root.open) self.refresh(); };
    this.importButton.onclick = async function () {
      var file = self.file.files[0];
      if (!file) { self.status.textContent = self.strings.choose_file; return; }
      if (!/\.sf[23]$/i.test(file.name) || file.size <= 12 || file.size > 1073741824) {
        self.status.textContent = self.strings.invalid_file; return;
      }
      self.importButton.disabled = true;
      self.status.textContent = self.strings.importing;
      try {
        var library = await options.importFile(file);
        if (!library || !library.id || !Array.isArray(library.presets) || !library.presets.length) throw new Error(self.strings.invalid_file);
        if (!self.libraries.has(library.id)) self.library.appendChild(new Option(library.name, library.id));
        self.libraries.set(library.id, library); self.library.value = library.id;
        self.assignments.clear(); self.refresh();
        self.status.textContent = self.strings.imported.replace('{count}', library.presets.length);
      } catch (error) { self.status.textContent = self.strings.failed.replace('{error}', error.message); }
      finally { self.importButton.disabled = false; }
    };
    this.applyButton.onclick = async function () {
      self.refresh(); self.applyButton.disabled = true;
      try {
        var selection = self.selection();
        self.status.textContent = self.strings.applying;
        await options.apply(selection);
        self.status.textContent = '';
      } catch (error) { self.status.textContent = self.strings.failed.replace('{error}', error.message); }
      finally { self.applyButton.disabled = false; }
    };
    this.refresh();
  }
  Picker.prototype.refresh = function () {
    var self = this, old = this.source.value;
    this.sources = this.options.getSources();
    this.source.replaceChildren();
    this.sources.forEach(function (source) { self.source.appendChild(new Option(source.name, source.program + ':' + source.is_drum)); });
    if (Array.from(this.source.options).some(function (item) { return item.value === old; })) this.source.value = old;
    this.source.disabled = !this.library.value;
    this.preset.disabled = !this.library.value;
    this.showPreset();
  };
  Picker.prototype.showPreset = function () {
    var self = this, library = this.libraries.get(this.library.value);
    this.preset.replaceChildren(new Option(this.strings.choose, ''));
    if (!library || this.source.selectedIndex < 0) return;
    var source = this.sources[this.source.selectedIndex];
    library.presets.filter(function (p) { return (p.bank === 128) === source.is_drum; }).forEach(function (p) {
      self.preset.appendChild(new Option(p.name + ' · ' + p.bank + ':' + p.program, JSON.stringify([p.bank, p.program])));
    });
    var selected = this.assignments.get(this.source.value) || [source.is_drum ? 128 : 0, source.program];
    var value = JSON.stringify(selected);
    if (Array.from(this.preset.options).some(function (item) { return item.value === value; })) this.preset.value = value;
  };
  Picker.prototype.selection = function () {
    var self = this, library = this.libraries.get(this.library.value);
    if (!library) return null;
    var assignments = this.sources.map(function (source) {
      var selected = self.assignments.get(source.program + ':' + source.is_drum) || [source.is_drum ? 128 : 0, source.program];
      if (!library.presets.some(function (p) { return p.bank === selected[0] && p.program === selected[1]; })) {
        throw new Error(self.strings.missing_preset.replace('{instrument}', source.name));
      }
      return {source_program: source.program, is_drum: source.is_drum, bank: selected[0], program: selected[1]};
    });
    return {id: library.id, assignments: assignments};
  };
  window.MidiSoundFontPicker = Picker;
})();
"""

SOUNDFONT_STRING_KEYS = (
    "title",
    "notice",
    "library",
    "default",
    "file",
    "import",
    "source",
    "preset",
    "apply",
    "pending",
    "choose_file",
    "invalid_file",
    "importing",
    "imported",
    "failed",
    "applying",
    "choose",
    "missing_preset",
    "render",
    "ready",
    "download",
)
