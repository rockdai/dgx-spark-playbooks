# Frontend Migration Plan

## Decision

Stop investing further in Docusaurus theming.
Build a custom frontend layer for `DGX Spark 中文社区`.

## Why

- NVIDIA's site is not a classic markdown docs site. It is a custom app-style frontend.
- Docusaurus is becoming a constraint for layout, routing, branding, and interaction patterns.
- We already have usable content in `website/docs/` that can be reused as the content source.

## Recommended direction

Use Next.js App Router as the new frontend layer.

## Suggested structure

- `website-next/`
  - `app/`
  - `components/`
  - `content/` or direct reuse of `../website/docs/`
  - `lib/markdown.ts`
  - `public/`

## Content strategy

Short term:
- Reuse existing markdown files under `website/docs/`
- Parse frontmatter and markdown ourselves
- Map selected playbooks into custom routes

Medium term:
- Normalize metadata for each playbook
- Split large playbooks into structured sections or tabs
- Introduce typed content schema

## First milestone

1. Scaffold a Next.js app
2. Recreate landing page in custom frontend
3. Render `intro.md`
4. Render one playbook page, starting with `connect-to-your-spark`
5. Preserve branding changes already made:
   - name: `DGX Spark 中文社区`
   - top-right CTA: `立即购买`
   - footer links including `DataV.AI`
   - no theme switch
   - no favicon

## Migration note

Docusaurus can remain temporarily for content reference, but should stop being the main delivery target once Next.js frontend is usable.
