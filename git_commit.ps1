# Git commit script - Create PR branch

# Create feature branch
git checkout -b refactor/v2-algorithm-docs

# Stage all changes
git add -A

# Commit with concise message
git commit -m "refactor: improve height conversion algorithm and documentation

- Use Mahoney (2001) geodetic algorithm for geopotential-to-geometric height
- Add ERA5/IFS documentation references to README
- Add log_utils.py for unified logging
- Clean up project structure (move dev files to temp/)"

# Push branch
git push -u origin refactor/v2-algorithm-docs

Write-Host ""
Write-Host "Branch pushed! Create PR at:"
Write-Host "https://github.com/YOUR_USERNAME/nwm/pull/new/refactor/v2-algorithm-docs"
