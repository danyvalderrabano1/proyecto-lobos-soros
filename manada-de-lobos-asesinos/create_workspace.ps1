# Helper PowerShell script to create the local workspace structure
$root = "manada-de-lobos-asesinos"
New-Item -ItemType Directory -Path $root -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $root "src") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $root ".vscode") -Force | Out-Null

"Created folders under $root"

# Create sample files if they don't exist
$files = @(
    @{ path = "$root\README.md"; text = "Manada de Lobos Asesinos - scaffold" },
    @{ path = "$root\ADMIN.md"; text = "Admin notes" },
    @{ path = "$root\src\main.py"; text = "print('Hello')" }
)

foreach ($f in $files) {
    if (-not (Test-Path $f.path)) {
        $f.text | Out-File -FilePath $f.path -Encoding UTF8
        Write-Host "Created $($f.path)"
    } else {
        Write-Host "Exists: $($f.path)"
    }
}

Write-Host "Workspace scaffold is ready. Open it with: code $root"
