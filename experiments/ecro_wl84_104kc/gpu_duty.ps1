param(
  [string]$Match = 'ecro_wl84_104kc',     # command-line substring of the WL104K-C worker processes (python.exe only)
  [double]$OnSeconds = 1.0,                # run window (seconds)
  [double]$Target = 0.45,                  # target GPU share of the WL104K-C workers (owner cap 0.50)
  [string]$Log = '',                       # CSV log, one row per cycle
  [string]$UtilFile = '',                  # nvidia-smi 100 ms timestamped utilization log written by this supervisor
  [string]$StopFile = ''                   # supervisor exits (after resuming everything) when this file exists
)
# WL104K-C GPU duty-cycle supervisor (owner instruction 2026-11-24: the batch uses at most 50 % of the GPU).
# Alternately resumes and suspends (NtSuspendProcess / NtResumeProcess) every python.exe whose command line contains $Match.
# Worker code is unchanged. Share estimate over the last ~30 s: mean GPU utilization of all samples minus the mean utilization
# in the late part of the suspended windows (>= 0.6 s after suspension: other applications only, queued kernels drained).
# The off window adapts so that the estimated share stays at $Target (queued kernels after a suspension are thereby counted).
# Every suspended process is resumed before the supervisor exits (finally block; also on a stop file). Suspensions never nest.
Add-Type -Namespace W104 -Name N -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
'@
$suspended = @{}
$off = $OnSeconds
$lateOff = [System.Collections.ArrayList]::new()     # [start, end] DateTime of late-off windows
$share = $null; $base = $null; $all = $null
function Get-Targets {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*$Match*" })
}
function Resume-All {
  foreach ($procId in @($suspended.Keys)) {
    try { [void][W104.N]::NtResumeProcess($suspended[$procId].Handle) } catch {}
    $suspended.Remove($procId)
  }
}
function Update-Share {
  if (-not $UtilFile -or -not (Test-Path $UtilFile)) { return }
  $now = Get-Date
  $rows = @(Get-Content $UtilFile -Tail 320 | ForEach-Object {
      $f = $_.Split(','); if ($f.Count -ge 2) {
        try { [pscustomobject]@{ t = [datetime]::ParseExact($f[0].Trim(), 'yyyy/MM/dd HH:mm:ss.fff', $null); u = [int]$f[1].Trim() } } catch {} } } |
      Where-Object { $_ -and ($now - $_.t).TotalSeconds -le 30 })
  if ($rows.Count -lt 50) { return }
  while ($lateOff.Count -gt 0 -and ($now - $lateOff[0][1]).TotalSeconds -gt 60) { $lateOff.RemoveAt(0) }
  $b = @($rows | Where-Object { $r = $_; @($lateOff | Where-Object { $r.t -ge $_[0] -and $r.t -le $_[1] }).Count -gt 0 })
  $script:all = ($rows | Measure-Object -Property u -Average).Average / 100.0
  if ($b.Count -ge 5) { $script:base = ($b | Measure-Object -Property u -Average).Average / 100.0 }
  if ($null -ne $script:base) { $script:share = [Math]::Max(0.0, $script:all - $script:base) }
}
if ($Log -and -not (Test-Path $Log)) { Set-Content -Path $Log -Value 'time,n_targets,on_s,off_s,util_all,util_others,share' }
$smi = $null
if ($UtilFile) {
  $smi = Start-Process -FilePath nvidia-smi -ArgumentList '--query-gpu=timestamp,utilization.gpu', '--format=csv,noheader,nounits', '-lms', '100' -RedirectStandardOutput $UtilFile -WindowStyle Hidden -PassThru
}
$sw = [System.Diagnostics.Stopwatch]::new()
try {
  while ($true) {
    if ($StopFile -and (Test-Path $StopFile)) { break }
    $sw.Restart()
    Resume-All
    $targets = Get-Targets
    $rest = [int](1000 * $OnSeconds - $sw.ElapsedMilliseconds); if ($rest -gt 0) { Start-Sleep -Milliseconds $rest }
    $onActual = $sw.Elapsed.TotalSeconds
    $sw.Restart()
    $offStart = Get-Date
    foreach ($p in $targets) {
      try {
        $proc = Get-Process -Id $p.ProcessId -ErrorAction Stop
        [void][W104.N]::NtSuspendProcess($proc.Handle)
        $suspended[$p.ProcessId] = $proc
      } catch {}
    }
    Update-Share
    if ($null -ne $share -and $targets.Count -gt 0) {
      if ($share -gt $Target) { $off = [Math]::Min($off * 1.08, 8.0 * $OnSeconds) }
      elseif ($share -lt $Target - 0.05) { $off = [Math]::Max($off * 0.96, 0.8 * $OnSeconds) }
    }
    if ($Log) { Add-Content -Path $Log -Value ("{0},{1},{2:F3},{3:F3},{4},{5},{6}" -f (Get-Date -Format o), $targets.Count, $onActual, $off, $all, $base, $share) }
    $rest = [int](1000 * $off - $sw.ElapsedMilliseconds); if ($rest -gt 0) { Start-Sleep -Milliseconds $rest }
    if ($targets.Count -gt 0 -and $off -ge 0.8) { [void]$lateOff.Add(@($offStart.AddSeconds(0.6), (Get-Date))) }
  }
} finally {
  Resume-All
  if ($smi) { try { Stop-Process -Id $smi.Id -ErrorAction SilentlyContinue } catch {} }
}
