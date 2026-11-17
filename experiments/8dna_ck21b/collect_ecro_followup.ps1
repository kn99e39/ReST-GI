$ErrorActionPreference='Stop'
$project=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$stem='ecro_ck2_wl100_near_4096spp'
$destination=Join-Path $project "evaluation\21B\$stem"
$comparison=Join-Path $project 'eval_comparison\ECRO'
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
  & $ssh -o BatchMode=yes -o ConnectTimeout=10 LabServer63 "test -f $remote/complete.json" 2>$null
  if($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 45; continue }
  & $scp "LabServer63:$remote/complete.json" (Join-Path $destination 'complete.json')
  if($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 45; continue }
  $record=Get-Content -LiteralPath (Join-Path $destination 'complete.json') -Raw | ConvertFrom-Json
  if($record.status -ne 'complete' -or $record.spp -ne 4096 -or $record.reference_used_at_inference -ne $false -or $record.learned_parameters_unchanged -ne $true) { throw 'Unexpected ECRO render identity' }
  if($record.checkpoint_sha256 -ne 'b3550787e08c8a4d41c975eb485be459591d84cf14a7c010a36984d10ac62e5a') { throw 'Unexpected ECRO checkpoint' }
  foreach($entry in $record.files.PSObject.Properties) {
    $name=$entry.Name
    if([IO.Path]::GetFileName($name) -ne $name) { throw 'Unsafe artifact filename' }
    $incoming=Join-Path $destination "$name.incoming"
    & $scp "LabServer63:$remote/$name" $incoming
    if($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
    if((FileSHA $incoming) -ne $entry.Value) { throw "Hash mismatch: $name" }
    Move-Item -LiteralPath $incoming -Destination (Join-Path $destination $name) -Force
    if($name.EndsWith('.png')) { Copy-Item -LiteralPath (Join-Path $destination $name) -Destination (Join-Path $comparison $name) }
  }
  Copy-Item -LiteralPath (Join-Path $destination 'complete.json') -Destination (Join-Path $comparison "$stem.json")
  Write-Output "Collected and SHA256-verified $stem at $([DateTime]::UtcNow.ToString('o'))"
  exit 0
}
throw 'Timed out waiting for ECRO follow-up'
