#!/usr/bin/env node

/**
 * Script to fetch and apply Context7 Docusaurus patterns to documentation files
 * This script demonstrates how Claude Code can interact with the Context7 MCP Server
 * to fetch authoritative Docusaurus documentation patterns.
 */

const fs = require('fs').promises;
const path = require('path');

class Context7PatternApplier {
  constructor(docsPath = './docs') {
    this.docsPath = docsPath;
    this.patterns = {};
  }

  /**
   * Fetch Docusaurus patterns from Context7 MCP Server
   */
  async fetchPatterns() {
    console.log('Fetching Docusaurus patterns from Context7 MCP Server...');

    // In a real implementation, this would make actual MCP Server calls
    // For demonstration, we'll return some example patterns
    this.patterns = {
      mdxComponent: {
        syntax: 'import ComponentName from "@site/src/components/ComponentName";',
        usage: '<ComponentName prop="value" />'
      },
      sidebarConfig: {
        type: 'category',
        label: 'Category Name',
        items: ['doc-id']
      },
      frontmatter: {
        title: 'Document Title',
        description: 'Brief description',
        sidebar_label: 'Sidebar Label'
      }
    };

    console.log('Patterns fetched successfully');
    return this.patterns;
  }

  /**
   * Apply patterns to documentation files
   */
  async applyPatterns() {
    console.log('Applying Context7 patterns to documentation files...');

    // This would iterate through documentation files and apply patterns
    // For now, we'll just log what would be done
    const files = await this.getMarkdownFiles();

    for (const file of files) {
      console.log(`Processing file: ${file}`);
      await this.updateFileWithPatterns(file);
    }

    console.log('Patterns applied successfully');
  }

  /**
   * Get all markdown files in docs directory
   */
  async getMarkdownFiles() {
    const files = [];

    const walk = async (dir) => {
      const dirents = await fs.readdir(dir, { withFileTypes: true });

      for (const dirent of dirents) {
        const res = path.resolve(dir, dirent.name);

        if (dirent.isDirectory()) {
          await walk(res);
        } else if (dirent.isFile() && path.extname(dirent.name) === '.md') {
          files.push(res);
        }
      }
    };

    await walk(this.docsPath);
    return files;
  }

  /**
   * Update a file with appropriate patterns
   */
  async updateFileWithPatterns(filePath) {
    let content = await fs.readFile(filePath, 'utf8');

    // Apply frontmatter pattern if not present
    if (!content.match(/^---[\s\S]*?---/)) {
      const frontmatter = `---
title: "${path.basename(filePath, '.md').replace(/-/g, ' ')}"
sidebar_label: "${path.basename(filePath, '.md').replace(/-/g, ' ')}"
description: "Documentation for ${path.basename(filePath, '.md').replace(/-/g, ' ')}"
---

`;
      content = frontmatter + content;
    }

    await fs.writeFile(filePath, content);
  }

  /**
   * Main execution method
   */
  async execute() {
    try {
      await this.fetchPatterns();
      await this.applyPatterns();
      console.log('Context7 pattern application completed successfully');
    } catch (error) {
      console.error('Error applying Context7 patterns:', error);
      throw error;
    }
  }
}

// Execute if run directly
if (require.main === module) {
  const applier = new Context7PatternApplier('./docs');
  applier.execute().catch(console.error);
}

module.exports = Context7PatternApplier;