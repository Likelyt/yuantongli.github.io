---
title: Website update workflow
date: 2026-07-29
hidden: true
---

# Website update workflow

A quick note on the new workflow for updating this site:

1. Edit `.jemdoc` or `.md` files.
2. Run `.venv/bin/python build_blog.py` to regenerate the blog index and posts.
3. Run `.venv/bin/python jemdoc.py *.jemdoc` to regenerate all HTML pages.
4. Commit and push to `origin main`.

The site is hosted on GitHub Pages at `Likelyt/yuantongli.github.io` and served from `liyuantong93.com`.
