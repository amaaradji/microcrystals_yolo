# Run 5-fold cross-validation with best config (batch=16, lr=0.001, 100 epochs)
# UPDATE THESE TWO PATHS before running:
$DataDir    = "D:\microcrystals\notebooks881616-test\Nanocrystals2-12"
$ResultsDir = "D:\microcrystals\results\cross_validation"

python cross_validate.py `
    --data-dir    $DataDir `
    --results-dir $ResultsDir `
    --folds       5 `
    --epochs      100 `
    --gpu         0

Write-Host "`nDone. Summary at: $ResultsDir\cv_summary.txt"
