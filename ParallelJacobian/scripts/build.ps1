if (-not (Test-Path -Path "../build")) {
    New-Item -ItemType Directory -Path "../build"
}

Set-Location "../build"

cmake ..

cmake --build .

Set-Location "../scripts"