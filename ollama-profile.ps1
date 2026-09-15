param(
    [ValidateSet("lightweight", "development", "validation")]
    [string]$Profile = "development",
    [switch]$Pull
)

$models = @{
    lightweight = "qwen3:8b"
    development = "qwen3:14b"
    validation = "qwen3:32b"
}

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw "Ollama is not available in PATH. Install Ollama or start the portable copy first."
}

$model = $models[$Profile]
$env:OLLAMA_PROFILE = $Profile
Write-Host "Profile: $Profile"
Write-Host "Model:   $model"

if ($Pull) {
    & $ollama.Source pull $model
}

Write-Host "Run: python agent.py chat `"your question`""
