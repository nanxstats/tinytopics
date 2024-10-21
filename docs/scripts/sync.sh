#!/bin/zsh

# Render .qmd to .md and convert to .py
sync_article() {
    local article_name=$1
    local article_path="docs/articles/$article_name.qmd"
    local example_output="examples/$article_name.py"

    # Render .qmd to .md
    quarto render "$article_path"

    # Convert .qmd to .ipynb
    quarto convert "$article_path"

    # Convert .ipynb to .py using nbconvert from venv
    python -m nbconvert --to python "docs/articles/$article_name.ipynb" --output "../../$example_output"

    # Clean up
    rm "docs/articles/$article_name.ipynb"

    # Format .py using black from venv
    python -m black "$example_output"
}

# Sync README.md with modified image path for docs/index.md
sed 's|docs/assets/logo.png|assets/logo.png|g' README.md > docs/index.md

# Sync articles
for article in get-started benchmark; do
    sync_article "$article"
done
