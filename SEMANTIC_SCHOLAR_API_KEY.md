# Cómo Obtener y Usar una API Key de Semantic Scholar

## ¿Por qué necesitas una API Key?

Sin API key:
- ⚠️ Rate limit: ~100 requests / 5 minutos
- ⚠️ Esperas largas entre requests (60s)
- ⚠️ Proceso de descarga lento

Con API key:
- ✅ Rate limit: 1 request / segundo (mucho más rápido)
- ✅ Sin esperas largas
- ✅ Proceso más eficiente

## Paso 1: Obtener tu API Key

1. **Visita el sitio oficial:**
   ```
   https://www.semanticscholar.org/product/api
   ```

2. **Regístrate o inicia sesión:**
   - Haz clic en "Request API Key" o "Get Started"
   - Crea una cuenta con tu correo institucional (recomendado)
   - O usa tu correo personal

3. **Solicita la API Key:**
   - Completa el formulario con:
     - Tu nombre
     - Email
     - Propósito: "Academic Research - Bibliometric Analysis"
     - Organización: "Universidad del Quindío" (o tu institución)
   
4. **Recibe tu API Key:**
   - Te llegará por correo electrónico
   - Guárdala en un lugar seguro
   - Ejemplo: `abc123def456ghi789jkl012mno345pqr678`

## Paso 2: Configurar tu API Key

### Opción 1: Variables de Entorno (Recomendado)

**En Windows PowerShell:**
```powershell
# Temporal (solo para esta sesión)
$env:SEMANTIC_SCHOLAR_API_KEY = "TU_API_KEY_AQUI"

# Permanente (agregar al perfil)
[Environment]::SetEnvironmentVariable("SEMANTIC_SCHOLAR_API_KEY", "TU_API_KEY_AQUI", "User")
```

**En Linux/Mac:**
```bash
# Temporal
export SEMANTIC_SCHOLAR_API_KEY="TU_API_KEY_AQUI"

# Permanente (agregar a ~/.bashrc o ~/.zshrc)
echo 'export SEMANTIC_SCHOLAR_API_KEY="TU_API_KEY_AQUI"' >> ~/.bashrc
source ~/.bashrc
```

### Opción 2: Archivo .env

1. **Crea o edita el archivo `.env`:**
   ```
   bibliometric-analysis/.env
   ```

2. **Agrega tu API key:**
   ```
   SEMANTIC_SCHOLAR_API_KEY=TU_API_KEY_AQUI
   ```

3. **El sistema la cargará automáticamente**

### Opción 3: Pasarla directamente al scraper

```python
from src.scrapers.semantic_scholar_api import SemanticScholarAPI

scraper = SemanticScholarAPI(api_key="TU_API_KEY_AQUI")
```

## Paso 3: Verificar que funciona

Ejecuta el test:

```bash
cd ProyectoAlgoritmos
python test_semantic_scholar.py
```

Deberías ver:
- ✅ Sin mensajes de "Rate limited"
- ✅ Descarga más rápida
- ✅ 1 request por segundo

## Límites con API Key

| Característica | Sin API Key | Con API Key |
|----------------|-------------|-------------|
| Requests/minuto | ~20 | 60 |
| Requests/día | ~288 | 86,400 |
| Velocidad | Lenta | Rápida |
| Espera entre requests | 3-60s | 1s |

## Mejores Prácticas

1. **No compartas tu API key** - Es personal y única
2. **No la subas a Git** - Usa `.gitignore` para `.env`
3. **Monitorea tu uso** - Revisa límites en el dashboard
4. **Sé respetuoso** - No abuses del servicio

## Solución de Problemas

### "API Key inválida"
- Verifica que copiaste la key correctamente
- Revisa que no tenga espacios al inicio/final
- Confirma que la variable de entorno está configurada

### "Rate limit excedido (incluso con API key)"
- Espera unos minutos y reintenta
- Verifica que la API key esté siendo usada
- Reduce `max_results` temporalmente

### "No puedo obtener API key"
- El sistema funciona sin API key (solo es más lento)
- Considera usar APIs alternativas (arXiv, PubMed)
- Contacta soporte de Semantic Scholar

## Alternativas sin API Key

Si no puedes obtener una API key, el sistema funciona pero con limitaciones:

1. **Reduce `max_results`:**
   ```python
   # En lugar de 100, usa 50
   max_results_per_source = 50
   ```

2. **Ejecuta en horarios de baja demanda:**
   - Madrugada (2-6 AM)
   - Fines de semana

3. **Divide las descargas:**
   - Descarga por lotes pequeños
   - Guarda resultados intermedios

## Recursos Adicionales

- **Documentación oficial:** https://api.semanticscholar.org/
- **Dashboard de uso:** https://www.semanticscholar.org/product/api#api-dashboard
- **Soporte:** api-support@semanticscholar.org

---

## Estado Actual

✅ **Sistema funciona sin API key** (con limitaciones)
🎯 **Recomendado**: Obtener API key para mejor rendimiento
📊 **Impacto**: Reduce tiempo de descarga de ~30 min a ~3 min (para 100 artículos)

---
*Última actualización: 29 de octubre de 2025*
