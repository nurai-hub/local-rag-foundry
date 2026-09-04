Write-Host "Foundry Local sunucusu baslatiliyor..." -ForegroundColor Cyan
$serverOutput = foundry server start 2>&1 | Out-String
Write-Host $serverOutput

if ($serverOutput -match "http://127\.0\.0\.1:(\d+)") {
    $port = $matches[1]
    Write-Host "Port bulundu: $port" -ForegroundColor Green
} else {
    Write-Host "Port bulunamadi, varsayilan port kullanilacak." -ForegroundColor Yellow
    $port = "40975"
}

$env:FOUNDRY_BASE_URL = "http://127.0.0.1:$port/v1"

Write-Host "Modeller yukleniyor..." -ForegroundColor Cyan
foundry model load qwen2.5-1.5b
foundry model load qwen3-embedding-0.6b

Write-Host "RAG uygulamasi baslatiliyor..." -ForegroundColor Cyan
python rag.py
