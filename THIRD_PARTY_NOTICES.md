# Third-Party Notices

This file records third-party source, model artifacts, and major runtime components
incorporated into the source tree or redistributed by the portable builds.  A license
label below is recorded only when it is declared by the pinned upstream source.  It is
not inferred from a neighboring repository, paper, author, or file name.

## Release gate

The current source may be used for development and local verification, but **portable
redistribution is blocked** until every machine-readable marker below is removed by a
review backed by an upstream license grant and the required attribution/source-offer
materials.  `.github/workflows/release.yml` enforces these markers.

The following inventory is also parsed by the release workflow.  `VERIFIED` means the
declared license evidence for that component is recorded below; `BLOCKED` means the
component must not be redistributed.  The expected component-ID set is closed in the
workflow, so deleting an undeclared component row does not make the release pass.

```text
PORTABLE_COMPONENT: zfturbo_adapted_source | bundle=python/src/core/vocal_separator.py | artifact=adapted-source | revision=e6279a79bcf861ea355ef7f8f76808a2731b6636 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: yourmt3_checkpoints | bundle=models/yourmt3_all | artifact=5-checkpoint-set | revision=5e66c1ea173a8186e0d20432b841d3180cc015b5 | license=Apache-2.0 | status=VERIFIED
PORTABLE_COMPONENT: yourmt3_patched_source | bundle=python/amt | artifact=patched-source-tree | revision=manifest:28fde351b4fe0f0519571fcd64e128df48a5076e7c46a9c1a604165798fe986e | license=UNKNOWN-mixed | status=BLOCKED
PORTABLE_COMPONENT: leap_xe | bundle=models/audio-separator | artifact=checkpoint+config | revision=4e47d6662ae82eaa8b4ac4329fe66099a843b48e | license=UNKNOWN | status=BLOCKED
PORTABLE_COMPONENT: polarformer | bundle=models/audio-separator | artifact=onnx+config | revision=9158719ee2173edd480a735764627526506fe4af | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: bs_roformer_sw_fixed | bundle=models/audio-separator | artifact=checkpoint+config | revision=370198fbb6997e3f5774778254698794e7b1267d | license=UNKNOWN | status=BLOCKED
PORTABLE_COMPONENT: audio_separator | bundle=python/audio_separator | artifact=source+metadata | revision=0.44.1 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: transkun_source | bundle=python/transkun | artifact=source+metadata | revision=2.0.1 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: transkun_default_v2_weight | bundle=python/transkun/2.0.pt | artifact=checkpoint | revision=transkun:2.0.1 | license=UNKNOWN | status=BLOCKED
PORTABLE_COMPONENT: transkun_v2_aug | bundle=models/transkun_v2_aug | artifact=checkpoint+config | revision=sha256:f61ebf6467d89081fde9728b659895a3e3d65b4c89516964178967167fae6590 | license=UNKNOWN | status=BLOCKED
PORTABLE_COMPONENT: aria_amt_source | bundle=python/amt | artifact=source+metadata | revision=a1ab73fc901d1759ec3bc173c146b3c6a3040261 | license=Apache-2.0 | status=VERIFIED
PORTABLE_COMPONENT: aria_amt_checkpoint | bundle=models/aria_amt | artifact=checkpoint | revision=8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b | license=CC-BY-NC-SA-4.0 | status=BLOCKED
PORTABLE_COMPONENT: bytedance_inference | bundle=python/piano_transcription_inference | artifact=source+metadata | revision=0.0.6 | license=UNKNOWN-classifier-only | status=BLOCKED
PORTABLE_COMPONENT: bytedance_checkpoint | bundle=models/bytedance_piano | artifact=checkpoint | revision=doi:10.5281/zenodo.4034264 | license=CC-BY-4.0 | status=VERIFIED
PORTABLE_COMPONENT: miros_source | bundle=external/ai4m-miros | artifact=patched-source-tree | revision=668a0aa6357bb3f09e767c9ece378956c2ffd182 | license=UNKNOWN | status=BLOCKED
PORTABLE_COMPONENT: musicfm_pretrained | bundle=external/ai4m-miros/model/musicfm/data/pretrained_msd.pt | artifact=checkpoint | revision=546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: miros_finetuned | bundle=external/ai4m-miros/logs/.../last.ckpt | artifact=checkpoint | revision=sha256:b1b8c167b3d2e3eaeb19202cd3fd366bb43492cd7720ff1516e1553c72e356e5 | license=UNKNOWN | status=BLOCKED
PORTABLE_COMPONENT: pytorch_cuda_runtime | bundle=python+native-runtime | artifact=torch:2.7.0+torchaudio:2.7.0+torchvision:0.22.0+CUDA | revision=cu128 | license=REVIEW-REQUIRED | status=BLOCKED
PORTABLE_COMPONENT: onnxruntime_gpu | bundle=python/onnxruntime | artifact=source+native-runtime | revision=1.23.2 | license=MIT | status=VERIFIED
PORTABLE_COMPONENT: pyqt6_runtime | bundle=python/PyQt6+Qt6 | artifact=source+native-runtime | revision=requirements-range | license=GPL-or-commercial | status=BLOCKED
PORTABLE_COMPONENT: ffmpeg_runtime | bundle=tools/ffmpeg | artifact=ffmpeg+ffprobe | revision=recorded-at-build | license=LGPL-or-GPL-build-dependent | status=BLOCKED
PORTABLE_COMPONENT: frozen_dependency_set | bundle=python+native-runtime | artifact=all-other-frozen-dependencies | revision=build-environment | license=INVENTORY-INCOMPLETE | status=BLOCKED
```

RELEASE_BLOCKER_UNRESOLVED_LICENSE: LEAP_XE_CHECKPOINT_AND_CONFIG

RELEASE_BLOCKER_UNRESOLVED_LICENSE: BS_ROFORMER_SW_FIXED_CHECKPOINT_AND_CONFIG

RELEASE_BLOCKER_UNRESOLVED_LICENSE: YOURMT3_PATCHED_SOURCE_PROVENANCE_AND_MIXED_LICENSES

RELEASE_BLOCKER_UNRESOLVED_LICENSE: TRANSKUN_DEFAULT_V2_WEIGHT

RELEASE_BLOCKER_UNRESOLVED_LICENSE: TRANSKUN_V2_AUG_GOOGLE_DRIVE_ARTIFACT

RELEASE_BLOCKER_UNRESOLVED_LICENSE: MIROS_SOURCE_AND_FINETUNED_CHECKPOINT

RELEASE_BLOCKER_UNRESOLVED_LICENSE: ARIA_AMT_CC_BY_NC_SA_CHECKPOINT_COMPLIANCE

RELEASE_BLOCKER_UNRESOLVED_LICENSE: BYTEDANCE_INFERENCE_SOURCE_LICENSE

RELEASE_BLOCKER_UNRESOLVED_LICENSE: PYQT6_CUDA_FFMPEG_AND_FROZEN_DEPENDENCY_COMPLIANCE

The evidence and required resolution for each marker are listed below.  A download
being publicly accessible does not by itself grant redistribution rights.

## Verified declarations and unresolved artifacts

### YourMT3+ source and five selectable checkpoints

- Upstream: [mimbres/YourMT3 Hugging Face Space](https://huggingface.co/spaces/mimbres/YourMT3)
- Pinned Space revision: `5e66c1ea173a8186e0d20432b841d3180cc015b5`
- Redistributed checkpoint material: the five identities listed by
  `OFFICIAL_YOURMT3_MODEL_KEYS`.
- Declared checkpoint/Space license: Apache License 2.0 (the pinned Space card
  declares `license: apache-2.0`).
- Note: this provenance is the Hugging Face **Space**, not a license inference from a
  similarly named GitHub repository.
- Patched-source status: **unresolved**.  The portable bundle also contains a
  106-file patched inference tree identified only by project manifest
  `28fde351b4fe0f0519571fcd64e128df48a5076e7c46a9c1a604165798fe986e`.
  The repository does not yet contain a reproducible upstream-revision → patch-set →
  manifest provenance record.  The tree also includes separately MIT-described
  material such as `extras/rotary_positional_embedding.py` (derived from
  `lucidrains/rotary-embedding-torch`) and the Unimax sampler documentation, without
  a complete preserved copyright/license inventory.  This is the
  `YOURMT3_PATCHED_SOURCE_PROVENANCE_AND_MIXED_LICENSES` release blocker.

### BS-RoFormer Leap XE 90-band vocals

- Upstream: [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap/tree/4e47d6662ae82eaa8b4ac4329fe66099a843b48e)
- Pinned revision: `4e47d6662ae82eaa8b4ac4329fe66099a843b48e`
- Redistributed material: `bs_leap_xe_voc.ckpt` and its configuration.
- License status: **unresolved**.  The pinned model metadata has no license value and
  the pinned repository contains no license file.  This is the
  `LEAP_XE_CHECKPOINT_AND_CONFIG` release blocker.

### BS PolarFormer accompaniment

- Upstream: [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer/tree/9158719ee2173edd480a735764627526506fe4af)
- Pinned revision: `9158719ee2173edd480a735764627526506fe4af`
- Redistributed material: the PolarFormer ONNX model and configuration used for the
  `VOCAL_SPLIT` accompaniment path.
- Declared license: MIT (the pinned model card declares `license: mit`).

### BS-RoFormer SW Fixed six-stem checkpoint

- Upstream: [noblebarkrr/mvsepless_resources](https://huggingface.co/noblebarkrr/mvsepless_resources/tree/370198fbb6997e3f5774778254698794e7b1267d)
- Pinned revision: `370198fbb6997e3f5774778254698794e7b1267d`
- Redistributed material: `bs_6stem_fixed.ckpt` and
  `bs_6stem_fixed_config.yaml`.
- License status: **unresolved**.  The pinned repository metadata has no license value
  and no license file was found.  This is the
  `BS_ROFORMER_SW_FIXED_CHECKPOINT_AND_CONFIG` release blocker.

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
- Default V2 checkpoint status: **unresolved**.  The source-code MIT license is not
  automatically extended to a trained weight without a model-specific declaration;
  no separate license for `2.0.pt` was found.  This is the
  `TRANSKUN_DEFAULT_V2_WEIGHT` release blocker.
- Separate V2 Aug artifact: `checkpointTransformerAug.zip`, Google Drive file ID
  `1Hg5ua8vYdtg1Y-MnXD0mLyhRK9Srd7hm`.
- V2 Aug license status: **unresolved**.  The file page does not provide a verifiable
  license grant for redistribution.  This is the
  `TRANSKUN_V2_AUG_GOOGLE_DRIVE_ARTIFACT` release blocker.

### Aria-AMT source and checkpoint

- Source: [EleutherAI/aria-amt](https://github.com/EleutherAI/aria-amt/tree/a1ab73fc901d1759ec3bc173c146b3c6a3040261)
- Pinned source revision: `a1ab73fc901d1759ec3bc173c146b3c6a3040261`
- Source license: Apache License 2.0, from the pinned upstream `LICENSE`.
- Checkpoint source: [loubb/aria-midi](https://huggingface.co/datasets/loubb/aria-midi/tree/8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b)
- Pinned checkpoint revision: `8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b`.
- Declared checkpoint/dataset license: CC BY-NC-SA 4.0.
- Release status: **blocked pending compliance review**.  The portable application is
  offered as an MIT project, while this checkpoint is explicitly non-commercial and
  ShareAlike.  The project has not documented the distribution classification,
  attribution, ShareAlike handling, or a separate permission.  This is the
  `ARIA_AMT_CC_BY_NC_SA_CHECKPOINT_COMPLIANCE` release blocker.

### ByteDance pedal-aware piano backend

- Runtime package: [piano-transcription-inference 0.0.6](https://pypi.org/project/piano-transcription-inference/0.0.6/)
- Package source-license status: **unresolved**.  PyPI carries an MIT classifier, but
  the referenced ByteDance inference source repository does not contain the license
  text needed to verify that declaration and preserve its notice.  This is the
  `BYTEDANCE_INFERENCE_SOURCE_LICENSE` release blocker.
- Checkpoint: [High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times](https://doi.org/10.5281/zenodo.4034264), creator Qiuqiang Kong.
- Declared checkpoint license: CC BY 4.0.  The portable notice must retain the title,
  creator, DOI, license, and an indication that the artifact is unmodified except for
  packaging.

### MIROS and MusicFM

- MIROS source: [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros/tree/668a0aa6357bb3f09e767c9ece378956c2ffd182)
- Pinned source revision: `668a0aa6357bb3f09e767c9ece378956c2ffd182`;
  the project applies controlled compatibility patches after checkout.
- MIROS fine-tuned checkpoint: upstream Google Drive file ID
  `1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB`.
- MIROS source/checkpoint license status: **unresolved**.  No license file or license
  declaration was found in the pinned source tree, and the Google Drive artifact page
  does not provide a redistribution license.  This is the
  `MIROS_SOURCE_AND_FINETUNED_CHECKPOINT` release blocker.
- MusicFM pretrained weight: [minzwon/MusicFM](https://huggingface.co/minzwon/MusicFM/tree/546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c), pinned revision
  `546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c`.
- MusicFM declared license: MIT (the pinned model card declares `license: mit`).

### Major frozen runtime components

- [ONNX Runtime GPU 1.23.2](https://pypi.org/project/onnxruntime-gpu/1.23.2/):
  MIT license in package metadata.
- PyTorch, torchaudio, and torchvision are fixed to `2.7.0`, `2.7.0`, and `0.22.0`;
  the portable build also contains NVIDIA CUDA libraries.
- FFmpeg/ffprobe binaries are copied from platform package managers/builds whose exact
  configuration can select LGPL or GPL components.
- Each attempted portable build writes `FFMPEG_BUILD_AUDIT.txt` beside this notice,
  recording the package source/version, exact executable paths, `ffmpeg -version`,
  `ffmpeg -buildconf`, `ffmpeg -L`, and SHA-256 values for both executables.  The
  workflow rejects any build containing `--enable-nonfree` before packaging.
- PyQt6 is dual-licensed under GPL/commercial terms, and the portable build currently
  does not record a commercial license or a complete GPL distribution plan.
- Release status: **blocked** until the build emits a complete version/license/SBOM
  inventory, records the exact FFmpeg configuration and corresponding-source route,
  and documents PyQt6, NVIDIA CUDA, and all frozen Python/native dependency
  redistribution obligations.  This is the
  `PYQT6_CUDA_FFMPEG_AND_FROZEN_DEPENDENCY_COMPLIANCE` release blocker.

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
