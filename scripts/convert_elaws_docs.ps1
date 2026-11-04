# Convert the e-Laws legacy Word (.doc) downloads to UTF-8 text so Spark can read them.
# e-Laws only offers HTML (rendered by JavaScript) or .doc, and Python has no good
# .doc reader, so this uses Microsoft Word itself. Windows + Word only; run once
# after scripts/download_sources.py. The .txt files are what get uploaded to Databricks.
$ErrorActionPreference = "Stop"
$docs = Join-Path $PSScriptRoot "..\data\raw\docs" | Resolve-Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    foreach ($f in Get-ChildItem $docs -Filter *.doc) {
        $out = [IO.Path]::ChangeExtension($f.FullName, ".txt")
        $d = $word.Documents.Open($f.FullName, $false, $true)   # ReadOnly
        # 7 = wdFormatUnicodeText; 65001 = UTF-8 encoding
        $d.SaveAs2([ref]$out, [ref]7, [ref]$false, [ref]"", [ref]$false, [ref]"",
                   [ref]$false, [ref]$false, [ref]$false, [ref]$false, [ref]$false, [ref]65001)
        $d.Close([ref]$false)
        Write-Host ("{0} -> {1} ({2:N0} bytes)" -f $f.Name, (Split-Path $out -Leaf), (Get-Item $out).Length)
    }
} finally {
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
