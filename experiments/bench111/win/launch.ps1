# WL111: start a .cmd file detached from the ssh session (WMI/CIM process creation; survives ssh disconnect).
param([Parameter(Mandatory=$true)][string]$Cmd, [string]$Dir = 'C:\Projects\bench111')
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = "cmd.exe /c `"$Cmd`""; CurrentDirectory = $Dir}
Write-Output ("pid=" + $r.ProcessId + " rc=" + $r.ReturnValue)
