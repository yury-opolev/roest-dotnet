<#
.SYNOPSIS
    Stages the native runtime artifacts (ONNX graphs + reference tensors needed
    at inference) and zips them into danish-voice-native-runtime.zip for a
    GitHub Release. Run from anywhere; paths are resolved relative to the repo.

.NOTES
    The bundle contains derived Røst-v3 weights (OpenRAIL) — see NOTICE.md.
#>
[CmdletBinding()]
param(
    [string]$OutZip = "danish-voice-native-runtime.zip"
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$nx = Join-Path $repo "native-onnx"
$stage = Join-Path $env:TEMP "dv-runtime-bundle"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
$null = New-Item -ItemType Directory -Path (Join-Path $stage "onnx_models")
$null = New-Item -ItemType Directory -Path (Join-Path $stage "refs")

# ONNX graphs needed at runtime. T3 uses the KV-cache graphs (prefill + decode)
# which share one external-data weights file; the no-cache t3_backbone is not
# shipped (T3Model only falls back to it when the KV graphs are absent).
$onnx = @(
    "t3_prefill.onnx", "t3_decode.onnx", "t3_kv_weights.data",
    "conformer_encoder_dyn.onnx", "cfm_decoder_z.onnx",
    "voc_f0_predictor.onnx", "voc_conv_stack.onnx"
)
foreach ($f in $onnx) {
    Copy-Item (Join-Path $nx "onnx_models\$f") (Join-Path $stage "onnx_models\$f")
}

# reference tensors / configs needed at runtime
$refs = @(
    # configs + tokenizer
    "grapheme_tokenizer.json", "text_specials.json", "t3_config.json", "vocoder_consts.json",
    "shapes.json", "shapes_t3.json", "shapes_t3loop.json", "shapes_flow.json",
    "shapes_flow2.json", "shapes_cond.json",
    # T3 tables + head
    "t3_speech_head_weight.bin",
    "t3_text_emb_weight.bin", "t3_speech_emb_weight.bin",
    "t3_text_pos_emb_weight.bin", "t3_speech_pos_emb_weight.bin",
    # flow linears
    "flow_input_embedding_weight.bin",
    "flow_encoder_proj_weight.bin", "flow_encoder_proj_bias.bin",
    "flow_spk_affine_weight.bin", "flow_spk_affine_bias.bin",
    # per-voice conditioning (Mic + Nic)
    "cond_mic_t3_cond_emb.bin", "cond_mic_prompt_feat.bin", "cond_mic_xvector.bin",
    "cond_mic_prompt_token.json", "cond_mic_meta.json",
    "cond_nic_t3_cond_emb.bin", "cond_nic_prompt_feat.bin", "cond_nic_xvector.bin",
    "cond_nic_prompt_token.json", "cond_nic_meta.json"
)
foreach ($f in $refs) {
    Copy-Item (Join-Path $nx "refs\$f") (Join-Path $stage "refs\$f")
}

Copy-Item (Join-Path $repo "NOTICE.md") (Join-Path $stage "NOTICE.md")
Copy-Item (Join-Path $nx "RUNTIME.md") (Join-Path $stage "RUNTIME.md")

# The full bundle exceeds GitHub's 2 GiB per-asset limit, so split into two:
#   part1 = the big shared T3 weights file; part2 = everything else.
# Both preserve the onnx_models/ + refs/ layout so they unzip into one folder.
$part1 = Join-Path $repo "roest-dotnet-runtime-part1-t3.zip"
$part2 = Join-Path $repo "roest-dotnet-runtime-part2.zip"
Remove-Item $part1, $part2 -Force -ErrorAction SilentlyContinue

$weights = "t3_kv_weights.data"
$p1stage = Join-Path $env:TEMP "dv-rel-part1"
if (Test-Path $p1stage) { Remove-Item $p1stage -Recurse -Force }
$null = New-Item -ItemType Directory -Path (Join-Path $p1stage "onnx_models")
Move-Item (Join-Path $stage "onnx_models\$weights") (Join-Path $p1stage "onnx_models\$weights")

Write-Host "Compressing part1 (T3 weights)..."
Compress-Archive -Path (Join-Path $p1stage "onnx_models") -DestinationPath $part1 -CompressionLevel NoCompression
Write-Host "Compressing part2 (graphs + refs)..."
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $part2 -CompressionLevel Fastest

"part1: {0:F2} GB" -f ((Get-Item $part1).Length / 1GB)
"part2: {0:F2} GB" -f ((Get-Item $part2).Length / 1GB)
Write-Host "Wrote $part1 and $part2"
