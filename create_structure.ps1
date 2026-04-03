param(
    [string]$ProjectRoot = 'C:\Users\ELVYN\OneDrive\Desktop\TFM\TFM_Model'
)

$folders = @(
    $ProjectRoot,
    "$ProjectRoot\app",
    "$ProjectRoot\data",
    "$ProjectRoot\data\raw",
    "$ProjectRoot\data\interim",
    "$ProjectRoot\data\processed",
    "$ProjectRoot\data\external",
    "$ProjectRoot\docs",
    "$ProjectRoot\docs\notes",
    "$ProjectRoot\docs\papers",
    "$ProjectRoot\models",
    "$ProjectRoot\models\artifacts",
    "$ProjectRoot\models\checkpoints",
    "$ProjectRoot\notebooks",
    "$ProjectRoot\outputs",
    "$ProjectRoot\outputs\figures",
    "$ProjectRoot\outputs\metrics",
    "$ProjectRoot\outputs\tables",
    "$ProjectRoot\src",
    "$ProjectRoot\src\data",
    "$ProjectRoot\src\evaluation",
    "$ProjectRoot\src\ingestion",
    "$ProjectRoot\src\models",
    "$ProjectRoot\src\utils",
    "$ProjectRoot\tests"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

Write-Host "Estructura creada en: $ProjectRoot" -ForegroundColor Green
