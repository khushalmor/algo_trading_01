
Write-Host "Starting Market Feed..."
python run_market_feed.py --csv complete.csv $args
if ($LASTEXITCODE -ne 0) {
    Write-Host "Script failed with error code $LASTEXITCODE" -ForegroundColor Red
}
Read-Host -Prompt "Press Enter to exit"
