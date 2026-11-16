$ErrorActionPreference='Stop'
$project=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$evaluation=Join-Path $project 'evaluation\21B'
$comparison=Join-Path $project 'eval_comparison\8DNA'
$ssh='C:\Program Files\Git\usr\bin\ssh.exe'
$scp='C:\Program Files\Git\usr\bin\scp.exe'
New-Item -ItemType Directory -Force -Path $evaluation,$comparison | Out-Null
$completed=@{}
$records=@()
function LocalSHA256([string]$path) {
  $stream=[System.IO.File]::OpenRead($path)
  $algorithm=[System.Security.Cryptography.SHA256]::Create()
  try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
  finally { $stream.Dispose(); $algorithm.Dispose() }
}
$deadline=[DateTime]::UtcNow.AddHours(24)
while([DateTime]::UtcNow -lt $deadline) {
  foreach($stage in @('epoch01_preview','final_epoch30')) {
    if($completed.ContainsKey($stage)) { continue }
    $remote='/home/nd/rest_8dna_ck21b/results'
    & $ssh -o BatchMode=yes -o ConnectTimeout=10 LabServer63 "test -f $remote/figure_$stage.json" 2>$null
    if($LASTEXITCODE -ne 0) { continue }
    $metadata=Join-Path $evaluation "figure_$stage.json"
    & $scp "LabServer63:$remote/figure_$stage.json" $metadata
    if($LASTEXITCODE -ne 0) { continue }
    $item=Get-Content -LiteralPath $metadata -Raw | ConvertFrom-Json
    $required=if($stage -eq 'final_epoch30') {245760} else {8192}
    if($item.checkpoint_global_step -lt $required) { throw 'Unqualified checkpoint stage' }
    $filename=if($stage -eq 'final_epoch30') {'ck2_references_and_candidates.png'} else {'ck2_references_and_candidates_epoch01_preview.png'}
    $incoming=Join-Path $evaluation "$filename.incoming"
    & $scp "LabServer63:$remote/ck2_references_and_candidates_$stage.png" $incoming
    if($LASTEXITCODE -ne 0) { continue }
    $hash=LocalSHA256 $incoming
    if($hash -ne $item.figure_sha256) { throw 'Downloaded image hash mismatch' }
    $destination=Join-Path $comparison $filename
    if(Test-Path -LiteralPath $destination) {
      if((LocalSHA256 $destination) -ne $hash) { throw "Existing comparison differs: $destination" }
    } else { Copy-Item -LiteralPath $incoming -Destination $destination }
    Move-Item -LiteralPath $incoming -Destination (Join-Path $evaluation $filename) -Force
    $records += [PSCustomObject]@{stage=$stage;path="eval_comparison/8DNA/$filename";sha256=$hash;checkpoint_global_step=$item.checkpoint_global_step;metadata="evaluation/21B/figure_$stage.json"}
    $manifest=[PSCustomObject]@{scope='CK2 bx38 static qualitative comparison; native 8DNA training; D6 reference versus native collapsed paths disclosed';source_server='LabServer63:/home/nd/rest_8dna_ck21b';files=$records}
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $evaluation 'manifest.json')
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $comparison 'ck2_manifest.json')
    $completed[$stage]=$true
    "Downloaded $stage at $([DateTime]::UtcNow.ToString('o'))" | Add-Content -LiteralPath (Join-Path $evaluation 'collect.log')
  }
  if($completed.ContainsKey('final_epoch30')) { break }
  Start-Sleep -Seconds 45
}
