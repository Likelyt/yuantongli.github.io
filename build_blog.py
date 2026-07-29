#!/usr/bin/env python3
"""Generate blog pages from Markdown files in blog/.

This produces a blog index and individual posts in a clean, readable style
inspired by Lilian Weng's blog (PaperMod theme).

Run this before building jemdoc pages:
    .venv/bin/python build_blog.py
"""
import os
import re
import glob
from datetime import datetime
import markdown

BLOG_DIR = 'blog'

POST_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="../jemdoc.css" type="text/css" />
<link rel="stylesheet" href="../blog.css" type="text/css" />
</head>
<body class="blog-post-page">
<header class="blog-header">
  <a href="../index.html" class="blog-logo">Yuantong Li</a>
  <nav class="blog-nav">
    <a href="../index.html">Home</a>
    <a href="../blog.html">Blog</a>
  </nav>
</header>
<main class="blog-main">
  <p class="post-meta">{date}</p>
  {content}
</main>
</body>
</html>
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blog | Yuantong Li</title>
<link rel="stylesheet" href="jemdoc.css" type="text/css" />
<link rel="stylesheet" href="blog.css" type="text/css" />
</head>
<body class="blog-page">
<header class="blog-header">
  <a href="index.html" class="blog-logo">Yuantong Li</a>
  <nav class="blog-nav">
    <a href="index.html">Home</a>
    <a href="blog.html">Blog</a>
  </nav>
</header>
<main class="blog-main">
  <h1>Blog</h1>
  <p class="blog-intro">Notes, updates, and short write-ups.</p>
  <ul class="blog-list">
{posts}
  </ul>
</main>
</body>
</html>
"""


def parse_frontmatter(text):
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            fm = parts[1].strip()
            body = parts[2].strip()
            meta = {}
            for line in fm.split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
            return meta, body
    return {}, text


def extract_title(body):
    m = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return 'Untitled'


def extract_date(md_path, meta):
    if 'date' in meta:
        return meta['date']
    m = re.match(r'(\d{4}-\d{2}-\d{2})', os.path.basename(md_path))
    if m:
        return m.group(1)
    return datetime.fromtimestamp(os.path.getmtime(md_path)).strftime('%Y-%m-%d')


def main():
    os.makedirs(BLOG_DIR, exist_ok=True)
    posts = []

    for md_path in sorted(glob.glob(os.path.join(BLOG_DIR, '*.md'))):
        with open(md_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        meta, body = parse_frontmatter(raw)
        title = meta.get('title', extract_title(body))
        date = extract_date(md_path, meta)
        slug = os.path.splitext(os.path.basename(md_path))[0]
        html_path = os.path.join(BLOG_DIR, slug + '.html')

        content = markdown.markdown(body)

        html = POST_TEMPLATE.format(title=title, date=date, content=content)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        posts.append({
            'title': title,
            'date': date,
            'slug': slug,
        })

    # Sort by date descending (newest first).
    posts.sort(key=lambda x: x['date'], reverse=True)

    list_items = []
    for post in posts:
        list_items.append(
            '    <li><span class="post-date">{}</span>'
            '<a href="blog/{}.html">{}</a></li>'.format(
                post['date'], post['slug'], post['title']))

    with open('blog.html', 'w', encoding='utf-8') as f:
        f.write(INDEX_TEMPLATE.format(posts='\n'.join(list_items)))


if __name__ == '__main__':
    main()
