# Translation Workflow Documentation

## Overview
This document outlines the complete workflow for translating the Physical AI & Humanoid Robotics documentation from English to Urdu.

## Translation Process

### 1. Preparation Phase
- Ensure English content is finalized before starting translation
- Review the [translation guidelines](./translation-guidelines.md) document
- Set up proper development environment with RTL support

### 2. Translation Workflow
1. **Complete English chapter first** - Ensure the English version is finalized and tested
2. **Extract translatable content** - Separate text content from code blocks and diagrams
3. **Translate chapter content to Urdu** - Maintain technical accuracy while ensuring readability
4. **Add Urdu comments to complex code examples** - Explain complex logic in Urdu
5. **Review translation with native Urdu technical reviewer** - Ensure technical accuracy
6. **Test RTL rendering and fix layout issues** - Verify proper display in RTL mode
7. **Mark chapter as translation-complete** - Update status tracking

### 3. File Structure
- English content: `docs/week-XX/chapter-name.md`
- Urdu content: `i18n/ur/docusaurus-plugin-content-docs/current/week-XX/chapter-name.md`
- Use Docusaurus i18n structure for proper locale management

### 4. Quality Assurance
- Verify all links work correctly in both languages
- Test language switcher functionality
- Confirm RTL layout works properly on all pages
- Ensure build process succeeds for both locales

### 5. Tools Recommended
- Code editor with RTL support
- Translation memory tools for consistency
- Technical reference materials for accurate terminology
- Browser testing tools for RTL verification

## Review Process
- Technical accuracy review by native Urdu speaker with robotics/AI background
- Consistency review to ensure terminology alignment
- Layout and functionality testing in RTL mode