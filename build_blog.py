#!/usr/bin/env python3
"""Generate blog pages from Markdown files in blog/.

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
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="../jemdoc.css" type="text/css" />
<style>
#layout-content {{ max-width: 800px; margin: 2em auto; padding: 0 1em; }}
</style>
</head>
<body>
<div id="layout-content">
<p><a href="../blog.html">&larr; Back to blog</a></p>
<p><em>{date}</em></p>
{content}
</div>
</body>
</html>
"""

INDEX_HEADER = """# jemdoc: menu{MENU}{blog.html}

= Blog

Notes, updates, and short write-ups.

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

    with open('blog.jemdoc', 'w', encoding='utf-8') as f:
        f.write(INDEX_HEADER)
        for post in posts:
            f.write('- [blog/{}.html {} — {}]\n'.format(
                post['slug'], post['date'], post['title']))
        f.write('\n')


if __name__ == '__main__':
    main()
