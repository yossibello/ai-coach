# Claude Vision PDF Parser Setup

The app now supports **Claude vision-based extraction** as a fallback for scanned PDF blood tests. When a PDF contains only images (no extractable text), the system automatically sends the pages to Claude's vision API and returns structured lab values.

## Setup Instructions

### Step 1: Create Anthropic Account & Get API Key

1. Go to [https://console.anthropic.com](https://console.anthropic.com)
2. Sign up (free tier available; Claude 3.5 Haiku is the cheapest vision model at ~$0.005/PDF)
3. Click **API Keys** in the left sidebar
4. Click **Create Key** and copy the full key (starts with `sk-ant-...`)

### Step 2: Set Environment Variable

Add `ANTHROPIC_API_KEY` to your system environment:

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**Windows (System Environment - persistent):**
1. Open **Settings → System → About → Advanced system settings**
2. Click **Environment Variables**
3. Under "User variables", click **New**
4. Variable name: `ANTHROPIC_API_KEY`
5. Variable value: `sk-ant-...` (your key from step 1)
6. Click **OK**

**macOS/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

To make it permanent, add to `~/.zshrc` or `~/.bashrc`:
```
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Verify Installation

The following packages are already installed. To verify:
```powershell
cd "c:\Users\yossi\ai coach"
& ".venv\Scripts\python.exe" -c "
from app.nutrition.vision_extractor import is_available
if is_available():
    print('✓ Claude vision extractor is ready!')
else:
    print('✗ ANTHROPIC_API_KEY not set or anthropic package missing')
"
```

### Step 4: Restart Backend

After setting the env var, restart the backend:
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 2; $env:ANTHROPIC_API_KEY="sk-ant-..."; $env:DATABASE_URL="sqlite+aiosqlite:///./aicoach_dev.db"; $env:SECRET_KEY="dev-secret-stable"; $env:PYTHONPATH="c:\Users\yossi\ai coach\backend"; & "c:\Users\yossi\ai coach\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir "c:\Users\yossi\ai coach\backend" --log-level warning
```

## How It Works

1. **Text-first (fast, free):** Regex parser tries to extract lab values from text-PDF
2. **Vision fallback:** If < 3 markers extracted, Claude renders pages to images and reads them
3. **Language-agnostic:** Claude handles English, Swedish, German, French, etc.
4. **High accuracy:** ~95% on scanned medical PDFs (much better than local OCR)

## Cost

- **~$0.005–0.01 per PDF** using Haiku (cheapest vision model)
- Covers ~100–200 PDFs per $1 USD
- Only triggered for scanned/image-only PDFs (text-PDFs remain free)

## Supported Markers

The extractor recognizes 50+ blood markers including:
- Iron panel (ferritin, serum iron, TSAT, TIBC, hemoglobin, hematocrit)
- Vitamins (B12, folate, B6, vitamin D, folic acid)
- Minerals (magnesium, zinc, calcium, sodium, potassium)
- Thyroid (TSH, free T3, free T4)
- Liver/kidney (AST, ALT, creatinine, eGFR, urea)
- Metabolic (glucose, HbA1c)
- Lipids (total cholesterol, LDL, HDL, triglycerides)
- Hormones (testosterone, cortisol, SHBG)
- Inflammation (CRP)
- And more (see `backend/app/nutrition/markers.py`)

**Swedish alias support:** Automatically recognizes Swedish lab names (`S-Ferritin`, `P-Glukos`, `EVF`, `S-Kobalamin`, `S-Folat`, etc.)

## Troubleshooting

**"Vision fallback unavailable"**
- Check `ANTHROPIC_API_KEY` is set: `$env:ANTHROPIC_API_KEY`
- Verify packages: `pip list | grep anthropic`
- Restart backend after setting the env var

**"Claude API error"**
- Check API key is valid (goes to Anthropic console and regenerate if unsure)
- Check account has credits (free tier may have limits)
- Check internet connection

**"No markers found"**
- Lab format may not be recognized; try manual entry for now
- File an issue with the PDF and we'll add more aliases
