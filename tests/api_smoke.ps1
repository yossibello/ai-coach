$ErrorActionPreference = "Continue"
$base = "http://127.0.0.1:8001"

function Test-Step($name, $sb) {
    try {
        $r = & $sb
        Write-Host "PASS  $name" -ForegroundColor Green
        if ($r) { Write-Host "      -> $($r | ConvertTo-Json -Depth 2 -Compress)" }
        return $r
    } catch {
        $msg = $_.Exception.Message
        if ($_.ErrorDetails.Message) { $msg = $_.ErrorDetails.Message }
        Write-Host "FAIL  $name  ::  $msg" -ForegroundColor Red
        return $null
    }
}

Write-Host "=== API SMOKE TEST ===" -ForegroundColor Cyan

Test-Step "GET /health" { Invoke-RestMethod "$base/health" }

# Use unique email each run
$email = "audit-$(Get-Random)@test.com"
$signupBody = @{ email=$email; password="Test12345!"; name="Audit User" } | ConvertTo-Json
$signup = Test-Step "POST /auth/signup" {
    Invoke-RestMethod "$base/api/v1/auth/signup" -Method Post -Body $signupBody -ContentType "application/json"
}

$loginBody = @{ username=$email; password="Test12345!" }
$login = Test-Step "POST /auth/login" {
    Invoke-RestMethod "$base/api/v1/auth/login" -Method Post -Body $loginBody
}

if (-not $login) { Write-Host "Cannot continue without token"; exit 1 }
$h = @{ Authorization = "Bearer $($login.access_token)" }

Test-Step "GET /auth/me" { Invoke-RestMethod "$base/api/v1/auth/me" -Headers $h }
Test-Step "GET /profile" { Invoke-RestMethod "$base/api/v1/profile" -Headers $h }

# Update profile so coach has FTP etc
$profBody = @{
    age=35; weight_kg=72; height_cm=178; sex="male"
    ftp=250; max_hr=185; resting_hr=50
    cycling_experience_years=5; experience_level="intermediate"
    primary_goal="ftp_improvement"
} | ConvertTo-Json
Test-Step "PUT /profile" { Invoke-RestMethod "$base/api/v1/profile" -Method Put -Body $profBody -ContentType "application/json" -Headers $h }

Test-Step "GET /activities" { Invoke-RestMethod "$base/api/v1/activities" -Headers $h }
Test-Step "GET /fitness/progression" { Invoke-RestMethod "$base/api/v1/fitness/progression" -Headers $h }
Test-Step "GET /coach/recommendation" { Invoke-RestMethod "$base/api/v1/coach/recommendation" -Headers $h }
Test-Step "GET /coach/recommendation/multi-horizon" { Invoke-RestMethod "$base/api/v1/coach/recommendation/multi-horizon" -Headers $h }
Test-Step "GET /strava/auth-url" { Invoke-RestMethod "$base/api/v1/strava/auth-url" -Headers $h }
Test-Step "GET /nutrition/blood-tests" { Invoke-RestMethod "$base/api/v1/nutrition/blood-tests" -Headers $h }
Test-Step "GET /nutrition/supplements" { Invoke-RestMethod "$base/api/v1/nutrition/supplements" -Headers $h }
Test-Step "GET /tracking/intake" { Invoke-RestMethod "$base/api/v1/tracking/intake" -Headers $h }
Test-Step "GET /tracking/performance-tests" { Invoke-RestMethod "$base/api/v1/tracking/performance-tests" -Headers $h }

Write-Host "=== DONE ===" -ForegroundColor Cyan
