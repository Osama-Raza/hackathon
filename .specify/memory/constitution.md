<!--
Sync Impact Report:
Version change: 1.0.0 → 1.0.1 (PATCH: Updated ratification date and governance information)
Modified principles: None
Added sections:
- Clear, Structured, and Instructional Writing
- Consistency with Spec-Driven Development
- Accuracy and Verifiability
- Practical, Example-Driven Explanations
- Modular Content
- Key Standards
- Constraints
- Success Criteria
- Governance
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- CLAUDE.md: ⚠ pending
- .specify/templates/commands/sp.constitution.md: ✅ updated (this file)
Follow-up TODOs:

-->
# AI-Driven Development Documentation Constitution

## Core Principles

### Clear, Structured, and Instructional Writing
Clear, structured, and instructional writing for beginners in AI-Driven Development.

### Consistency with Spec-Driven Development
Consistency with Spec-Driven Development methodology.

### Accuracy and Verifiability
Accuracy: all claims must be verifiable.

### Practical, Example-Driven Explanations
Practical, example-driven explanations.

### Modular Content
Modular content suitable for Docusaurus docs structure.

## Key Standards

### Content Quality
- Source references required for factual claims (link to official documentation)
- Writing level: Flesch-Kincaid grade 8–10 for accessibility
- Follow Docusaurus Markdown/MDX formatting conventions
- Include examples, diagrams, or code blocks where relevant
- Maintain uniform tone across all chapters (instructional, encouraging, technical but approachable)

### Technical Standards
- Platform: Docusaurus v3.x (React-based static site generator)
- Deployment: GitHub Pages with automated CI/CD via GitHub Actions
- Content format: MDX (Markdown with React components)
- Code examples: Tested on Ubuntu 22.04 + ROS 2 Humble
- All examples must include dependency installation and setup instructions

### Design & User Experience
- *Custom homepage* with hero section and feature cards showcasing curriculum modules
- *Modern color scheme*: Professional tech palette with primary brand color, dark mode support
- *Typography*: Clean, readable fonts (system-ui or Inter) with clear hierarchy (h1-h6)
- *Responsive design*: Mobile-first approach, tested on mobile, tablet, and desktop viewports
- *Navigation*: Intuitive sidebar structure, sticky navigation, breadcrumbs, and search functionality
- *Visual elements*: Custom logo, favicon, consistent iconography throughout site
- *Code presentation*: Syntax-highlighted code blocks with copy-to-clipboard buttons
- *Interactive components*: Tabbed code examples (Python/C++/YAML), collapsible sections where beneficial

### Accessibility & Performance
- WCAG 2.1 AA compliance (color contrast, keyboard navigation, screen reader support)
- Fast page load times (< 3 seconds on standard connection)
- Optimized images (WebP format, lazy loading)
- Clean URLs and proper meta tags for SEO

### Branding Consistency
- Consistent use of project name: "Physical AI & Humanoid Robotics"
- Tagline: "Master robotics from ROS 2 fundamentals to conversational AI"
- Color palette documented in custom.css with CSS variables
- Logo usage guidelines (light/dark variants)

## Constraints

Each chapter: 600–1200 words.
Entire book: multi-page Docusaurus structure.
Output must be clean Markdown, ready for GitHub Pages.
No plagiarism; all content must be original.

## Success Criteria

Book compiles cleanly in Docusaurus.
Content is accurate, structured, and beginner-friendly.
All chapters follow the same style guide.
Ready for deployment to GitHub Pages using CI/CD.

## Governance

This Constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan. All PRs/reviews must verify compliance. Complexity must be justified.


**Version: 1.0.1 | Ratified: 2025-12-05 | Last Amended: 2025-12-07