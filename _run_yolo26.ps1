# Run YOLO26m training and comparison experiment
# Prerequisites: ultralytics 8.4.72+ installed (supports yolo26m.pt)
# GPU 0 = non-augmented,  GPU 1 = augmented  (adjust --gpu as needed)
#
# UPDATE THESE THREE PATHS before running:
$NoAugYaml  = "D:\microcrystals\notebooks881616-test\Nanocrystals2-14\data.yaml"
$AugYaml    = "D:\microcrystals\notebooks881616-test\Nanocrystals2-12\data.yaml"
$ResultsDir = "D:\microcrystals\results\yolo26_comparison"

python run_yolo26_comparison.py `
    --no-aug-yaml  $NoAugYaml `
    --aug-yaml     $AugYaml `
    --results-dir  $ResultsDir `
    --model        yolo26m.pt `
    --gpu          0

Write-Host "`nDone. Results in: $ResultsDir"
Write-Host "Copy the YOLO26m rows from comparison_results_with_yolo26.csv into Table 2 in the LaTeX file."
