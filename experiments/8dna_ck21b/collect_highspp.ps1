$ErrorActionPreference='Stop'
$project=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$stem='8dna_ck2_epoch18_4096spp'
$destination=Join-Path $project "evaluation\21B\$stem"
$comparison=Join-Path $project 'eval_comparison\8DNA'
$remote="/home/nd/rest_8dna_ck21b/results/$stem"
$ssh='C:\Program Files\Git\usr\bin\ssh.exe'
$scp='C:\Program Files\Git\usr\bin\scp.exe'
New-Item -ItemType Directory -Force -Path $destination,$comparison | Out-Null
function FileSHA([string]$path) {
  $stream=[System.IO.File]::OpenRead($path)
  $algorithm=[System.Security.Cryptography.SHA256]::Create()
  try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
  finally { $stream.Dispose(); $algorithm.Dispose() }
}
$deadline=[DateTime]::UtcNow.AddHours(24)
while([DateTime]::UtcNow -lt $deadline) {
  & $ssh -o BatchMode=yes -o ConnectTimeout=10 LabServer63 "test -f $remote/matched4096_complete.json" 2>$null
  if($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 45; continue }
  & $scp "LabServer63:$remote/matched4096_complete.json" (Join-Path $destination 'matched4096_complete.json')
  if($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 45; continue }
  $record=Get-Content -LiteralPath (Join-Path $destination 'matched4096_complete.json') -Raw | ConvertFrom-Json
  if($record.status -ne 'complete' -or $record.spp_completed -ne 4096 -or $record.epochs_completed -ne 18 -or $record.checkpoint_global_step -ne 147456) { throw 'Unexpected render identity' }
  if($record.rest_gi_spp -ne 4096 -or $record.rest_gi_replicate -ne 'r0') { throw 'Unexpected ReST-GI sampling condition' }
  $files=@("$stem.png","$stem.npy","$stem.exr",'ck2_references_and_candidates_epoch18_matched4096spp.png')
  foreach($name in $files) {
    $incoming=Join-Path $destination "$name.incoming"
    & $scp "LabServer63:$remote/$name" $incoming
    if($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
    if((FileSHA $incoming) -ne $record.files.$name) { throw "Hash mismatch: $name" }
    Move-Item -LiteralPath $incoming -Destination (Join-Path $destination $name) -Force
    if($name.EndsWith('.png')) {
      $review=Join-Path $comparison $name
      if((Test-Path -LiteralPath $review) -and ((FileSHA $review) -ne $record.files.$name)) { throw "Existing preview differs: $review" }
      Copy-Item -LiteralPath (Join-Path $destination $name) -Destination $review
    }
  }
  Copy-Item -LiteralPath (Join-Path $destination 'matched4096_complete.json') -Destination (Join-Path $comparison "$stem.json")
  Write-Output "Collected and SHA256-verified $stem at $([DateTime]::UtcNow.ToString('o'))"
  exit 0
}
throw 'Timed out waiting for high-spp render'
