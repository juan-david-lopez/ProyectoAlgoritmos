# Script para verificar el tamaño de la imagen Docker
# Ejecutar después de que termine docker build

Write-Host "`n╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║         📊 TAMAÑO REAL DE LA IMAGEN DOCKER                      ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Verificar si la imagen existe
$imageExists = docker images bibliometric-analysis:test --format "{{.Repository}}" 2>$null

if ($imageExists) {
    # Obtener información de la imagen
    $imageInfo = docker images bibliometric-analysis:test --format "{{.Size}}" 2>$null
    Write-Host "✅ Imagen construida exitosamente" -ForegroundColor Green
    Write-Host ""
    Write-Host "📦 TAMAÑO FINAL:" -ForegroundColor Cyan
    Write-Host "   $imageInfo" -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host ""
    
    # Obtener detalles completos
    Write-Host "📋 DETALLES DE LA IMAGEN:" -ForegroundColor Cyan
    docker images bibliometric-analysis:test --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedAt}}"
    Write-Host ""
    
    # Comparar con límite de Render
    Write-Host "📊 ANÁLISIS:" -ForegroundColor Cyan
    $sizeStr = $imageInfo -replace 'GB|MB|KB', ''
    $sizeNum = [double]$sizeStr
    
    if ($imageInfo -like "*GB*") {
        if ($sizeNum -lt 2) {
            Write-Host "   ✅ Cabrá en Render (límite 2 GB)" -ForegroundColor Green
            Write-Host "   ✅ Margen: $([math]::Round(2 - $sizeNum, 2)) GB disponibles" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Excede límite de Render (2 GB)" -ForegroundColor Red
            Write-Host "   ⚠️  Sobrepeso: $([math]::Round($sizeNum - 2, 2)) GB" -ForegroundColor Red
        }
    } else {
        Write-Host "   ✅✅ Excelente! Mucho menor a 2 GB" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "🔍 INSPECCIÓN DETALLADA:" -ForegroundColor Cyan
    docker history bibliometric-analysis:test --human --no-trunc | Select-Object -First 15
    
} else {
    Write-Host "❌ La imagen no existe todavía" -ForegroundColor Red
    Write-Host "   Ejecuta primero: docker build -t bibliometric-analysis:test ." -ForegroundColor Yellow
}

Write-Host ""
