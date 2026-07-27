param(
    [Parameter(Mandatory=$true)][string]$BddRoot,
    [string]$Output = "data/bdd100k_yolo",
    [int]$MaxImagesPerSplit = 0,
    [double]$ValTestFraction = 0.20,
    [int]$Seed = 42
)

$argsList = @(
    "-m", "road_detection.bdd100k_to_yolo",
    "--bdd-root", $BddRoot,
    "--output", $Output,
    "--splits", "train", "val",
    "--copy-mode", "hardlink",
    "--val-test-fraction", "$ValTestFraction",
    "--seed", "$Seed"
)

if ($MaxImagesPerSplit -gt 0) {
    $argsList += @("--max-images-per-split", "$MaxImagesPerSplit")
}

python $argsList
