instructions = """
1. Checkout `main` branch.
2. Edit and stage files (no commit yet).
    i. Update CHANGELOG.md
        - Add the new version heading below "Unreleased".
        - Write "_No changes yet._" in "Unreleased" section.
        - Add the link for the new version tag. (bottom of changelog)
        - Update the "Unreleased" comparison link (bottom) to the new tag.
    ii. Update `pyproject.toml` to new version.
3. Rebuild the package artifacts:
    $ make build
4. Publish to PyPI:
    $ make publish
5. Commit change as "Release vX.Y.Z" and push.
6. Create a GitHub release.
    - See: https://github.com/patcon/valency-anndata/releases/new
    - Name the new tag `vX.Y.Z`
    - Set the title `vX.Y.Z`
    - Copy the new changelog text into the body.
    - Attach the wheel and tar build files from `dist/`
"""

print(instructions.strip())
