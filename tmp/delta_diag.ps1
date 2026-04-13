$path = "data/processed/merged_credit_rating_common_3groups.csv"
$rows = Import-Csv $path
$features = @('current_ratio','debt_equity_ratio','gross_profit_margin','operating_profit_margin','ebit_margin','pretax_profit_margin','net_profit_margin','asset_turnover','roe','roa','operating_cashflow_ps','free_cashflow_ps')

$parsed = foreach($r in $rows){
  $dt = $null
  try { $dt = [datetime]::Parse($r.rating_date) } catch { }
  if($r.ticker -and $dt){
    [pscustomobject]@{
      ticker = $r.ticker
      rating_agency = $r.rating_agency
      rating_date = $dt
      row = $r
    }
  }
}

$tickerGroups = $parsed | Group-Object ticker
$totalTickers = $tickerGroups.Count
$totalRows = $parsed.Count
$rowsInMultiObsTickers = ($tickerGroups | Where-Object { $_.Count -ge 2 } | Measure-Object -Property Count -Sum).Sum
if(-not $rowsInMultiObsTickers){ $rowsInMultiObsTickers = 0 }

Write-Output "TOTAL_ROWS_PARSED=$totalRows"
Write-Output "TOTAL_TICKERS=$totalTickers"
Write-Output "ROWS_IN_TICKERS_WITH_GE2_OBS=$rowsInMultiObsTickers"
Write-Output "SHARE_ROWS_WITH_POSSIBLE_DELTA=$([math]::Round(($rowsInMultiObsTickers / [math]::Max($totalRows,1))*100,2))%"

$deltaStats = @{}
foreach($f in $features){
  $deltaStats[$f] = [pscustomobject]@{NonNull=0; Zero=0; NonZero=0}
}

foreach($g in $tickerGroups){
  $ordered = $g.Group | Sort-Object rating_date, rating_agency
  for($i=1; $i -lt $ordered.Count; $i++){
    $prev = $ordered[$i-1].row
    $curr = $ordered[$i].row
    foreach($f in $features){
      $a = 0.0; $b = 0.0
      $okA = [double]::TryParse([string]$prev.$f, [ref]$a)
      $okB = [double]::TryParse([string]$curr.$f, [ref]$b)
      if($okA -and $okB){
        $delta = $b - $a
        $st = $deltaStats[$f]
        $st.NonNull++
        if([math]::Abs($delta) -lt 1e-12){ $st.Zero++ } else { $st.NonZero++ }
      }
    }
  }
}

Write-Output "\nDELTA_DIAGNOSTICS"
$report = foreach($f in $features){
  $st = $deltaStats[$f]
  $nonNull = [math]::Max($st.NonNull,1)
  [pscustomobject]@{
    feature = $f
    non_null_delta_rows = $st.NonNull
    non_zero_pct = [math]::Round(($st.NonZero/$nonNull)*100,2)
    zero_pct = [math]::Round(($st.Zero/$nonNull)*100,2)
  }
}
$report | Format-Table -AutoSize | Out-String -Width 260 | Write-Output

Write-Output "SAME_DAY_REPEAT_CHECK"
$sameDayPairs = 0
$sameDayPairsTotal = 0
foreach($g in $tickerGroups){
  $ordered = $g.Group | Sort-Object rating_date, rating_agency
  for($i=1; $i -lt $ordered.Count; $i++){
    $prev = $ordered[$i-1]
    $curr = $ordered[$i]
    if($prev.rating_date.Date -eq $curr.rating_date.Date){
      $sameDayPairsTotal++
      $allEq = $true
      foreach($f in $features){
        if([string]$prev.row.$f -ne [string]$curr.row.$f){ $allEq = $false; break }
      }
      if($allEq){ $sameDayPairs++ }
    }
  }
}
Write-Output "SAME_DAY_ADJACENT_PAIRS=$sameDayPairsTotal"
Write-Output "SAME_DAY_IDENTICAL_FINANCIAL_PAIRS=$sameDayPairs"
if($sameDayPairsTotal -gt 0){ Write-Output "SAME_DAY_IDENTICAL_RATE=$([math]::Round(($sameDayPairs/[double]$sameDayPairsTotal)*100,2))%" }
