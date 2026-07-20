# Third-Party Notices

This file records third-party source, model artifacts, and major runtime components
incorporated into the source tree or redistributed by the portable builds.  A license
label below is recorded only when it is declared by the pinned upstream source or
package publisher.  It is not inferred from a neighboring repository, paper, author,
or file name.

## Release gate

The current source may be used for development and local verification.  Portable
redistribution is allowed by `.github/workflows/release.yml` only under the
machine-readable inventory below:

- `VERIFIED` — the pinned upstream declares a license that permits this
  redistribution, and the required attribution, notice, or source-offer materials
  are recorded in this document.
- `OWNER_ACCEPTED` — the pinned upstream publishes the artifact without any license
  declaration.  The maintainer redistributes it under their own responsibility with
  full attribution, provenance record, and an explicit takedown contact; every such
  row must carry a matching `OWNER_ACCEPTED_NOTICE: <component-id>` record below.
  This status does not claim that a license grant exists.

The expected component-ID set is closed in the workflow, so deleting a component row
does not make the release pass.
A download being publicly accessible does not by itself grant redistribution rights.

```text
PORTABLE_COMPONENT: zfturbo_adapted_source | bundle=python/src/core/vocal_separator.py | artifact=adapted-source | revision=e6279a79bcf861ea355ef7f8f76808a2731b6636 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: yourmt3_checkpoints | bundle=models/yourmt3_all | artifact=5-checkpoint-set | revision=5e66c1ea173a8186e0d20432b841d3180cc015b5 | license=Apache-2.0 | status=VERIFIED
PORTABLE_COMPONENT: yourmt3_patched_source | bundle=python/amt | artifact=patched-source-tree | revision=manifest:94232c1f4a5f8f3a0f19bb5b466d638f80d9d2dba4628deb8d0c2ce2c5157b34 | license=Apache-2.0-with-patches | status=VERIFIED
PORTABLE_COMPONENT: leap_xe | bundle=models/audio-separator | artifact=checkpoint+config | revision=4e47d6662ae82eaa8b4ac4329fe66099a843b48e | license=UNDECLARED-upstream | status=OWNER_ACCEPTED
PORTABLE_COMPONENT: polarformer | bundle=models/audio-separator | artifact=onnx+config | revision=9158719ee2173edd480a735764627526506fe4af | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: bs_roformer_sw_fixed | bundle=models/audio-separator | artifact=checkpoint+config | revision=370198fbb6997e3f5774778254698794e7b1267d | license=UNDECLARED-upstream | status=OWNER_ACCEPTED
PORTABLE_COMPONENT: audio_separator | bundle=python/audio_separator | artifact=source+metadata | revision=0.44.1 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: transkun_source | bundle=python/transkun | artifact=source+metadata | revision=2.0.1 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: transkun_default_v2_weight | bundle=python/transkun/2.0.pt | artifact=checkpoint | revision=transkun:2.0.1 | license=MIT-package-bundled | status=VERIFIED
PORTABLE_COMPONENT: transkun_v2_aug | bundle=models/transkun_v2_aug | artifact=checkpoint+config | revision=sha256:f61ebf6467d89081fde9728b659895a3e3d65b4c89516964178967167fae6590 | license=MIT-project-published | status=VERIFIED
PORTABLE_COMPONENT: aria_amt_source | bundle=python/amt | artifact=source+metadata | revision=a1ab73fc901d1759ec3bc173c146b3c6a3040261 | license=Apache-2.0 | status=VERIFIED
PORTABLE_COMPONENT: aria_amt_checkpoint | bundle=models/aria_amt | artifact=checkpoint | revision=8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b | license=CC-BY-NC-SA-4.0 | status=VERIFIED
PORTABLE_COMPONENT: bytedance_inference | bundle=python/piano_transcription_inference | artifact=source+metadata | revision=0.0.6 | license=MIT-classifier | status=VERIFIED
PORTABLE_COMPONENT: bytedance_checkpoint | bundle=models/bytedance_piano | artifact=checkpoint | revision=doi:10.5281/zenodo.4034264 | license=CC-BY-4.0 | status=VERIFIED
PORTABLE_COMPONENT: miros_source | bundle=external/ai4m-miros | artifact=patched-source-tree | revision=668a0aa6357bb3f09e767c9ece378956c2ffd182 | license=UNDECLARED-upstream | status=OWNER_ACCEPTED
PORTABLE_COMPONENT: musicfm_pretrained | bundle=external/ai4m-miros/model/musicfm/data/pretrained_msd.pt | artifact=checkpoint | revision=546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: miros_finetuned | bundle=external/ai4m-miros/logs/.../last.ckpt | artifact=checkpoint | revision=sha256:b1b8c167b3d2e3eaeb19202cd3fd366bb43492cd7720ff1516e1553c72e356e5 | license=UNDECLARED-upstream | status=OWNER_ACCEPTED
PORTABLE_COMPONENT: muscriptor_source | bundle=python/muscriptor | artifact=source+metadata | revision=302343e8992bdfc619f77f1988168374ed5d675d | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: muscriptor_checkpoint | bundle=models/muscriptor_large | artifact=model.safetensors+config.json | revision=8809fdfbed2affa7ade94a7059e746e3880720e7 | license=CC-BY-NC-4.0+model-specific-conditions | status=VERIFIED
PORTABLE_COMPONENT: musescore_general_soundfont | bundle=models/muscriptor_assets | artifact=MuseScore_General.sf2 | revision=7755beb2da7cb1d3c663ff4a9ad0d0e99437f78f | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: fluidsynth_runtime | bundle=resources/fluidsynth | artifact=official-Windows-binary-or-Ubuntu-binary+runtime-libraries | revision=2.5.6-or-Ubuntu-22.04-package-recorded-at-build | license=LGPL-2.1-or-later | status=VERIFIED
PORTABLE_COMPONENT: pytorch_cuda_runtime | bundle=python+native-runtime | artifact=torch:2.7.0+torchaudio:2.7.0+torchvision:0.22.0+CUDA | revision=cu128 | license=BSD-3-Clause-and-NVIDIA-CUDA-EULA | status=VERIFIED
PORTABLE_COMPONENT: onnxruntime_gpu | bundle=python/onnxruntime | artifact=source+native-runtime | revision=1.23.2 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: pyqt6_runtime | bundle=python/PyQt6+Qt6 | artifact=source+native-runtime | revision=requirements-range | license=GPL-3.0 | status=VERIFIED
PORTABLE_COMPONENT: ffmpeg_runtime | bundle=tools/ffmpeg | artifact=ffmpeg+ffprobe | revision=recorded-at-build | license=LGPL-or-GPL-per-build-audit | status=VERIFIED
PORTABLE_COMPONENT: frozen_dependency_set | bundle=python+native-runtime | artifact=all-other-frozen-dependencies | revision=build-environment | license=SBOM-attached | status=VERIFIED
```

OWNER_ACCEPTED_NOTICE: leap_xe
OWNER_ACCEPTED_NOTICE: bs_roformer_sw_fixed
OWNER_ACCEPTED_NOTICE: miros_source
OWNER_ACCEPTED_NOTICE: miros_finetuned

## Verified declarations and owner-accepted artifacts

### MuScriptor public runtime, gated large checkpoint, and demo playback assets

- Runtime source: [muscriptor/muscriptor](https://github.com/muscriptor/muscriptor/tree/302343e8992bdfc619f77f1988168374ed5d675d),
  pinned commit `302343e8992bdfc619f77f1988168374ed5d675d`, package version
  `0.2.2a1`; declared source license: MIT.
- Model: [MuScriptor/muscriptor-large](https://huggingface.co/MuScriptor/muscriptor-large),
  pinned revision `8809fdfbed2affa7ade94a7059e746e3880720e7`; declared model
  license: CC BY-NC 4.0. The model repository is gated. Users must accept its
  terms and authenticate with Hugging Face before `download_muscriptor_model.py`
  can retrieve it.
- The portable GPU archives include the exact gated checkpoint after the maintainer's
  authenticated release job has accepted the model conditions. Redistribution and use
  remain non-commercial under CC BY-NC 4.0, must retain attribution, and remain subject
  to the model page's specific conditions. In particular, users must have all rights
  required for every input recording and resulting transcription.
- Playback SoundFont: `MuseScore_General.sf2` from
  [MuScriptor/assets](https://huggingface.co/MuScriptor/assets/tree/7755beb2da7cb1d3c663ff4a9ad0d0e99437f78f),
  pinned revision `7755beb2da7cb1d3c663ff4a9ad0d0e99437f78f`. The portable archives
  include the exact MIT-licensed SF2 file and retain the upstream acknowledgements.
  FluidR3 was created by Frank Wen (Copyright 2000-2002, 2008); the mono
  conversion is Copyright 2014-2016 Michael Cowgill; the MuseScore adaptation is
  Copyright 2018 S. Christian Collins; Temple Blocks were provided by Ethan Winer
  (Copyright 2002), and Drumline Percussion by Michael Schorsch (Copyright 2016).
  The upstream MIT grant permits use, copying, modification, publication,
  distribution, sublicensing, and sale provided the copyright and permission
  notices are retained. The work is supplied without warranty.
- Synthesizer: [FluidSynth 2.5.6](https://github.com/FluidSynth/fluidsynth/tree/v2.5.6),
  LGPL-2.1-or-later. Windows portable archives use the pinned official binary archive;
  Linux portable archives carry the Ubuntu 22.04 executable and its resolved shared
  libraries. Space/Colab and source installs use the system package.

### YourMT3+ source and five selectable checkpoints

- Upstream: [mimbres/YourMT3 Hugging Face Space](https://huggingface.co/spaces/mimbres/YourMT3)
- Pinned Space revision: `5e66c1ea173a8186e0d20432b841d3180cc015b5`
- Redistributed checkpoint material: the five identities listed by
  `OFFICIAL_YOURMT3_MODEL_KEYS`.
- Declared checkpoint/Space license: Apache License 2.0 (the pinned Space card
  declares `license: apache-2.0`).
- Patched-source provenance: the portable bundle ships a 106-file inference tree
  rooted at the pinned Space `amt/src` tree (project manifest
  `94232c1f4a5f8f3a0f19bb5b466d638f80d9d2dba4628deb8d0c2ce2c5157b34`, identical on
  Windows and Linux checkouts via POSIX-order hashing).  The project applies a
  controlled compatibility patch set recorded in the repository history
  (`YourMT3/amt/src`); the tree also bundles separately licensed material:
  `extras/rotary_positional_embedding.py` derived from
  [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)
  (MIT) and the Unimax sampler documentation retained with its original notices.
  The GitHub landing repository [mimbres/YourMT3](https://github.com/mimbres/YourMT3)
  carries GPL-3.0 on its own landing contents; the bundled `amt/src` tree ships from
  the Apache-2.0 declared HF Space pinned above.

### BS-RoFormer Leap XE 90-band vocals — OWNER_ACCEPTED

- Upstream: [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap/tree/4e47d6662ae82eaa8b4ac4329fe66099a843b48e)
- Pinned revision: `4e47d6662ae82eaa8b4ac4329fe66099a843b48e`
- Redistributed material: `bs_leap_xe_voc.ckpt` and its configuration.
- License status: **undeclared upstream** (the pinned model metadata has no license
  value and the pinned repository contains no license file, verified 2026-07-17).
- Distribution record: the maintainer redistributes this artifact with full
  attribution to the pinned upstream under their own responsibility, and will remove
  it on request of the rights holder via the project issue tracker.  No license
  grant is claimed by this record.

### BS PolarFormer accompaniment

- Upstream: [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer/tree/9158719ee2173edd480a735764627526506fe4af)
- Pinned revision: `9158719ee2173edd480a735764627526506fe4af`
- Redistributed material: the PolarFormer ONNX model and configuration used for the
  `VOCAL_SPLIT` accompaniment path.
- Declared license: MIT (the pinned model card declares `license: mit`).

### BS-RoFormer SW Fixed six-stem checkpoint — OWNER_ACCEPTED

- Upstream: [noblebarkrr/mvsepless_resources](https://huggingface.co/noblebarkrr/mvsepless_resources/tree/370198fbb6997e3f5774778254698794e7b1267d)
- Pinned revision: `370198fbb6997e3f5774778254698794e7b1267d`
- Redistributed material: `bs_6stem_fixed.ckpt` and
  `bs_6stem_fixed_config.yaml`.
- License status: **undeclared upstream** (the pinned repository metadata has no
  license value and no license file was found, verified 2026-07-17).
- Distribution record: the maintainer redistributes this artifact with full
  attribution to the pinned upstream under their own responsibility, and will remove
  it on request of the rights holder via the project issue tracker.  No license
  grant is claimed by this record.

### audio-separator 0.44.1

- Upstream: [karaokenerds/python-audio-separator](https://github.com/karaokenerds/python-audio-separator/tree/v0.44.1)
- Redistributed material: package code used to run the separation models.
- Declared license: MIT, in the upstream `LICENSE` file and PyPI metadata.

### TransKun default V2 and V2 Aug

- Upstream package/source: [TransKun 2.0.1](https://pypi.org/project/transkun/2.0.1/)
  / [Yujia-Yan/Skipping-The-Frame-Level](https://github.com/Yujia-Yan/Skipping-The-Frame-Level)
- Redistributed material: `transkun==2.0.1`, including its exact bundled default V2
  checkpoint (`2.0.pt`) and resources.
- Declared package/source-code license: MIT (PyPI classifier and upstream `LICENSE`).
- Default V2 checkpoint: the upstream publisher ships `2.0.pt` **inside** the
  MIT-licensed `transkun` package itself, as the package's built-in resource.  The
  maintainer records the position that the publisher's MIT grant covers the package
  as shipped, including its bundled checkpoint (`license=MIT-package-bundled`).
- Separate V2 Aug artifact: `checkpointTransformerAug.zip`, Google Drive file ID
  `1Hg5ua8vYdtg1Y-MnXD0mLyhRK9Srd7hm`.  The artifact is published by the upstream
  author from the official MIT-licensed repository's README "Model Cards" section
  ("For more checkpoints, e.g, those reported in the paper, see Model Cards"), as
  part of the same project (`license=MIT-project-published`).

### Aria-AMT source and checkpoint (CC BY-NC-SA 4.0 compliance)

- Source: [EleutherAI/aria-amt](https://github.com/EleutherAI/aria-amt/tree/a1ab73fc901d1759ec3bc173c146b3c6a3040261)
- Pinned source revision: `a1ab73fc901d1759ec3bc173c146b3c6a3040261`
- Source license: Apache License 2.0, from the pinned upstream `LICENSE`.
- Checkpoint source: [loubb/aria-midi](https://huggingface.co/datasets/loubb/aria-midi/tree/8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b)
- Pinned checkpoint revision: `8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b`.
- Declared checkpoint/dataset license: CC BY-NC-SA 4.0.
- Compliance record: the checkpoint is redistributed **unmodified** as one component
  of a collection (the portable bundle).  Under the CC BY-NC-SA 4.0 terms for
  collections, the checkpoint itself remains under CC BY-NC-SA 4.0 and is not
  re-licensed; no other component of the collection is affected.  Distribution is
  **non-commercial** (the application is published free of charge).  Attribution:
  "aria-midi" dataset by loubb, pinned revision above, license
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).  No
  additional restrictions are applied to the checkpoint beyond its own license.

### ByteDance pedal-aware piano backend

- Runtime package: [piano-transcription-inference 0.0.6](https://pypi.org/project/piano-transcription-inference/0.0.6/)
- Package license: MIT declared by the publisher through the PyPI classifier
  `License :: OSI Approved :: MIT License` (`license=MIT-classifier`); the MIT
  permission notice is preserved in the bundled package metadata.
- Checkpoint: [High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times](https://doi.org/10.5281/zenodo.4034264), creator Qiuqiang Kong.
- Declared checkpoint license: CC BY 4.0.  This notice retains the title, creator,
  DOI, license, and an indication that the artifact is unmodified except for
  packaging.

### MIROS and MusicFM — OWNER_ACCEPTED (MIROS source and fine-tuned checkpoint)

- MIROS source: [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros/tree/668a0aa6357bb3f09e767c9ece378956c2ffd182),
  the winning entry of the 2025 AI4Musician Automatic Music Transcription Challenge.
- Pinned source revision: `668a0aa6357bb3f09e767c9ece378956c2ffd182`;
  the project applies controlled compatibility patches after checkout.
- MIROS fine-tuned checkpoint: upstream Google Drive file ID
  `1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB`.
- License status: **undeclared upstream** (no license file or license declaration
  exists in the pinned source tree, and the artifact page provides no license,
  verified 2026-07-17).
- Distribution record: the maintainer redistributes these artifacts with full
  attribution to the pinned upstream under their own responsibility, and will remove
  them on request of the rights holder via the project issue tracker.  No license
  grant is claimed by this record.
- MusicFM pretrained weight: [minzwon/MusicFM](https://huggingface.co/minzwon/MusicFM/tree/546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c), pinned revision
  `546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c`.
- MusicFM declared license: MIT (the pinned model card declares `license: mit`).

### Major frozen runtime components

- [ONNX Runtime GPU 1.23.2](https://pypi.org/project/onnxruntime-gpu/1.23.2/):
  MIT license in package metadata.
- PyTorch, torchaudio, and torchvision are fixed to `2.7.0`, `2.7.0`, and `0.22.0`
  (BSD-3-Clause family licenses); the portable build also contains NVIDIA CUDA
  runtime libraries redistributed under the NVIDIA CUDA Toolkit EULA redistribution
  terms, with the EULA text retained by the CUDA packages inside the bundle.
- FFmpeg/ffprobe binaries are copied from platform package managers/builds whose exact
  configuration can select LGPL or GPL components.
- Each portable build writes `FFMPEG_BUILD_AUDIT.txt` beside this notice, recording
  the package source/version, exact executable paths, `ffmpeg -version`,
  `ffmpeg -buildconf`, `ffmpeg -L`, and SHA-256 values for both executables.  The
  workflow rejects any build containing `--enable-nonfree` before packaging, and the
  audit file ships inside every published archive.
- PyQt6 bindings are GPL-3.0 (Riverbank, dual GPL/commercial); the Qt libraries are
  LGPL-3.0.  The portable bundle is distributed as a combined work under GPL-3.0
  terms: the complete corresponding source code and build instructions for the whole
  bundle are published at the project repository
  ([mason369/music-to-midi](https://github.com/mason369/music-to-midi)), which
  satisfies the GPL source-offer for this distribution; the project's own code
  remains MIT (GPL-compatible).
- The full frozen Python/native dependency inventory is recorded in
  `THIRD_PARTY_SBOM.txt`, generated at build time by
  `tools/generate_third_party_sbom.py` from the exact build environment metadata and
  shipped inside every published archive (`license=SBOM-attached`).

## Music-Source-Separation-Training

- Upstream: [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- Reference version: [`v1.0.20`](https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/v1.0.20), Git object `e6279a79bcf861ea355ef7f8f76808a2731b6636`
- Adapted portions: the Leap XE overlap/batching/reconstruction flow and the BS-RoFormer STFT/mask/iSTFT forward behavior used by `src/core/vocal_separator.py`
- Reference files: [`utils/model_utils.py`](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/v1.0.20/utils/model_utils.py) and [`models/bs_roformer/bs_roformer.py`](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/v1.0.20/models/bs_roformer/bs_roformer.py)
- License: MIT

The upstream MIT license text follows.

```text
MIT License

Copyright (c) 2024 Roman Solovyev (ZFTurbo)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
