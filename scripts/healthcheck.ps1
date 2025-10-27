param(
  [string]$Image = "flower-classification:latest",
  [string]$Name = "flower-api",
  [int]$Port = 8000,
  [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

if (-not $NoBuild) {
  docker build -t $Image .
}

try { docker rm -f $Name | Out-Null } catch {}

docker run -d --name $Name -p $Port:8000 $Image | Out-Null

Write-Host "Waiting for service to be ready..."
$success = $false
for ($i=0; $i -lt 45; $i++) {
  try {
    $resp = curl.exe -s "http://localhost:$Port/health"
    if ($LASTEXITCODE -eq 0 -and $resp) {
      Write-Host "Service is up"
      Write-Output $resp
      $success = $true
      break
    }
  } catch {}
  Start-Sleep -Seconds 2
}

if (-not $success) {
  Write-Host "Healthcheck failed after retries"
  docker ps -a | Out-Host
  docker logs $Name | Out-Host
  try { docker rm -f $Name | Out-Null } catch {}
  exit 56
}
