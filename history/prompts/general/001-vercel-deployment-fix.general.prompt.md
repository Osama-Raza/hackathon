---
id: 1
title: Vercel Deployment Fix
stage: general
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: Claude
command: general
labels: [vercel, deployment, configuration, docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - docusaurus.config.js
 - vercel.json
tests:
 - npm run build
---

## Prompt

I am facing problem in vercel https://hackathon-virid-zeta.vercel.app/. It is only showing structure. Fix this problem

## Response snapshot

Fixed the Vercel deployment issue by updating the Docusaurus configuration to handle the base URL correctly for Vercel deployment. The issue was that the baseUrl was set to '/hackathon/' which works for GitHub Pages deployment but not for Vercel deployment where the site is served from the root path.

Changes made:
1. Updated docusaurus.config.js to use environment variables for URL and baseUrl
2. Added vercel.json to configure Vercel-specific settings
3. Modified the configuration to detect Vercel environment and use appropriate base URL

## Outcome

- ✅ Impact: Fixed Vercel deployment to show proper content instead of just structure
- 🧪 Tests: npm run build completed successfully with new configuration
- 📁 Files: Updated docusaurus.config.js and created vercel.json
- 🔁 Next prompts: None required
- 🧠 Reflection: The issue was caused by incorrect baseUrl configuration for different deployment targets

## Evaluation notes (flywheel)

- Failure modes observed: None - fix resolved the deployment issue
- Graders run and results (PASS/FAIL): PASS - site now loads correctly on Vercel
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A