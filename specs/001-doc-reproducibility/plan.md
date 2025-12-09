# Technical Plan: Docusaurus-based AI/Spec-Driven Book Project

**Feature Branch**: `001-doc-reproducibility`
**Created**: 2025-12-06
**Status**: Draft
**Input**: Feature Specification (`specs/001-doc-reproducibility/spec.md`) and user-defined plan structure.

## I. Architecture Sketch
**Goal**: Visualize the high-level system components and their interactions.
**Format**: Textual description.

The core of this project involves Claude CLI acting as the orchestrator for generating documentation. It will interact with the Context7 MCP Server to fetch authoritative Docusaurus documentation and guidelines. These guidelines will inform the generation of Docusaurus-based static site content, which will then be deployed to GitHub Pages for hosting. For reproducibility, a dedicated Ubuntu 22.04 environment, equipped with ROS 2 Humble and a ROS 2 Simulation Environment, will be used to validate all code examples and simulation files.

**Key Components**:
- **Claude CLI**: The primary agent orchestrating documentation generation, tool interaction, and workflow management.
- **Context7 MCP Server**: Provides up-to-date Docusaurus configuration syntax, sidebar structure, MDX rules, component usage, and theme customization examples. Acts as the single source of truth for Docusaurus best practices.
- **Docusaurus Static Site Generator**: The framework used to build the documentation website from markdown and MDX files.
- **GitHub Pages**: The hosting platform for the generated Docusaurus documentation, enabling continuous deployment.
- **Ubuntu 22.04 + ROS 2 Humble**: The target environment for ensuring reproducibility of all code and simulation examples.
- **ROS 2 Simulation Environment**: Provides the necessary tools and libraries for running URDF, SDF, and launch files included in the documentation examples.

## II. Section Structure & Content Strategy
**Goal**: Define the logical organization of the book and content principles.
**Format**: Outlines with brief descriptions.

The top-level sections of the book will be based on the following 13-week curriculum outline:
## 13-Week Curriculum Structure
**Week 1-2: Introduction To Physical AI**
- Foundations of Physical AI and embodied intelligence
- Overview of humanoid robotics landscape
- Sensor systems overview
**Week 3-5: ROS 2 Fundamentals**
- ROS 2 architecture and core concepts
- Nodes, topics, services, actions
- Building ROS 2 packages with Python

**Week 6-7: Robot Simulation with Gazebo**
- Gazebo simulation environment setup
- URDF and SDF robot description formats
- Physics simulation

**Week 8-10: NVIDIA Isaac Platform**
- NVIDIA Isaac SDK and Isaac Sim
- AI-powered perception and manipulation
- Sim-to-real transfer techniques

**Week 11-12: Humanoid Robot Development**
- Humanoid robot kinematics and dynamics
- Bipedal locomotion and balance control
- Natural human-robot interaction design

**Week 13: Conversational Robotics**
- Integrating GPT models for conversational AI
- Speech recognition and NLU
- Multi-modal interaction
 

**Content Principles**:
- **Clear, Structured, and Instructional Writing**: Each section will be written with a focus on clarity, using a logical flow and instructional tone. Complex topics will be broken down into digestible parts, suitable for beginners.
- **Accuracy and Verifiability**: All technical information, code examples, and instructions will be thoroughly checked against official documentation (from Context7 and ROS 2) and verified through automated testing to ensure factual correctness and reproducibility.
- **Practical, Example-Driven Explanations**: Concepts will be illustrated with practical, runnable code examples and real-world scenarios. Examples will be self-contained and demonstrative of the discussed topic.
- **Modular Content**: The documentation will be structured into modular units (pages, components, code blocks) to facilitate easy navigation, reuse, and integration into the Docusaurus framework. This supports Docusaurus's sidebar structure and component usage.

## III. Research & Discovery Approach
**Goal**: Outline methods for gathering information and verifying concepts.
**Format**: Step-by-step process.

**Key Activities**:
### Utilizing Context7 for Docusaurus Documentation

- The system MUST query the Context7 MCP Server at the beginning of each major writing phase:
  - *Research Phase* – Retrieve latest Docusaurus configuration rules, MDX syntax, and component usage.
  - *Foundation Phase* – Retrieve sidebar patterns, file structures, and theme customization references.
  - *Analysis Phase* – Retrieve examples necessary to support architectural decisions.
  - *Synthesis (Writing) Phase* – Retrieve code snippets and patterns needed for each new chapter.

- The system MUST re-query Context7 whenever:
  1. A new Docusaurus feature or update is detected.
  2. A configuration file needs creation or validation.
  3. A theme override or custom component is being added.
  4. MDX syntax guidance or component usage is required.
  5. A version change affects structure, config, or theme API.

- Application of Context7 patterns MUST occur:
  - Immediately during content generation.
  - At the moment a related file or section is being written (e.g., sidebar, hero component, MDX examples).
  - Before finalizing or approving each chapter.
  - During validation to ensure compliance with Docusaurus best practices.

- Each generated file (config files, MDX pages, components) MUST include:
  - Verified syntax retrieved from Context7.
  - Updated examples aligned with current Docusaurus standards.
  - Any relevant configuration or theme patterns retrieved during querying.
  
2.  **Exploring ROS 2 Documentation for Full Path Structures**:
    *   Research best practices and official guidelines for specifying full path structures within URDF, SDF, and launch files in ROS 2 Humble.
    *   Investigate mechanisms for ensuring these paths are robust and resolve correctly across different environments, especially considering the Ubuntu 22.04 baseline.
3.  **Identifying Dependency Management Strategies for Reproducible Examples**:
    *   Investigate standard ROS 2 dependency management tools (e.g., `rosdep`, `colcon`).
    *   Develop and document clear, complete dependency installation commands for all examples, ensuring they are executable on a clean Ubuntu 22.04 + ROS 2 Humble installation.
    *   Consider containerization (e.g., Docker) for examples to enhance reproducibility, if applicable and not adding undue complexity.

## IV. Quality Validation & Metrics
**Goal**: Define how the documentation's quality and reproducibility will be measured and maintained.
**Format**: Specific checks and tools.

**Reproducibility Checks**:
- **Automated Script Execution**: Develop and implement automated scripts that provision a clean Ubuntu 22.04 + ROS 2 Humble environment, execute all example setup commands, and verify successful completion and expected output.
- **Simulation File Validation**: Ensure all URDF, SDF, and launch files are syntactically correct and can be launched within the ROS 2 simulation environment. Verify full path structures are correctly implemented and resolved.
- **Dependency Resolution**: Automated checks to confirm that all dependency installation commands successfully resolve and install required packages without errors.

**Docusaurus Standards Checks**:
- **MDX Linter/Formatter**: Integrate MDX linting and formatting tools into the CI/CD pipeline to enforce consistent syntax, styling, and best practices derived from Context7 documentation.
- **Docusaurus Build Validation**: Ensure the generated Docusaurus site passes all internal build validations, detecting broken links, incorrect configurations, or rendering issues.
- **Context7 Pattern Adherence**: Implement automated checks (e.g., custom linters or scripts) to verify that generated pages explicitly follow patterns returned from Context7 documentation context for configuration, sidebar, MDX, components, and themes.

**Content Quality Checks**:
- **Readability Scores**: Utilize tools (e.g., Flesch-Kincaid) to assess the readability of documentation content, aiming for target scores suitable for beginners.
- **Consistency Checks**: Develop checks for terminology, formatting, and style consistency across the entire documentation.

## V. Decisions Needing Documentation (ADR Suggestions)
**Goal**: Identify architectural decisions made during planning that require formal ADRs.
**Format**: List of potential ADR titles with brief rationale.

-   ### [ADR-0001: Choice of Docusaurus Version/Theme](history/adr/0001-choice-of-docusaurus-version-and-theme.md): Rationale for selecting Docusaurus v3.x with classic theme as the base platform, including considerations for maintainability, features, upgrade path, and custom styling capabilities to support modern UX requirements.
-   ### [Strategy for Managing ROS 2 Environment Dependencies](history/adr/0002-ros-2-environment-dependency-management-strategy.md): Decision on how to ensure consistent and reproducible ROS 2 environments for examples, including the use of `rosdep`, `colcon`, or potential containerization solutions.
-   ### [Integration Approach for Context7 MCP Server](history/adr/0003-context7-mcp-server-integration-approach.md): Details on how Claude CLI will integrate with and consume information from the Context7 MCP Server, including caching strategies, error handling, and frequency of updates.
-   ### [ADR-0004: Custom Homepage Design and Styling Strategy](history/adr/0004-custom-homepage-design-and-styling-strategy.md)
    Decision on implementing custom homepage with hero section and feature cards vs. using default Docusaurus landing page. Includes rationale for:
    - Custom CSS approach vs. theme swizzling vs. complete custom theme
    - Color scheme and dark mode implementation strategy
    - Component architecture for reusable UI elements (hero, feature cards, tabbed code)
    - Performance optimization techniques (lazy loading, image optimization)
    - Responsive design implementation approach
    - Trade-offs between visual appeal and maintainability
-   ### [ADR-0005: Multilingual Support Implementation Strategy](history/adr/0005-multilingual-support-urdu-translation.md)
    Decision on implementing Urdu translation support using Docusaurus i18n plugin. Includes rationale for:
    - Translation approach (human translation vs. machine translation with review vs. hybrid)
    - File structure and organization (i18n directory vs. separate repo)
    - RTL layout handling and CSS adjustments needed
    - Technical terminology translation guidelines (keep English terms vs. transliterate vs. translate)
    - Maintenance strategy for keeping translations in sync with English updates
    - Deployment considerations (single site with locales vs. separate domains)
    - Trade-offs between translation quality, cost, and maintenance burden

## VI. Testing Strategy
**Goal**: Describe how the documentation and its examples will be tested.
**Format**: Test types, scope, and execution.

-   **Unit Tests**:
    *   **Scope**: For any custom Docusaurus components or utility scripts developed as part of the documentation generation process within Claude Code.
    *   **Execution**: Automated through standard testing frameworks (e.g., Jest, Pytest for Claude Code components).
-   **Integration Tests**:
    *   **Scope**: Automated execution of all documentation examples on a clean Ubuntu 22.04 + ROS 2 Humble environment. This includes installation steps, running code snippets, and launching simulation files.
    *   **Execution**: Implemented as part of the CI/CD pipeline, triggering on every documentation update or code change. Success criteria include error-free execution and validated outputs.
-   **End-to-End (E2E) Tests**:
    *   **Scope**: User journey validation on the deployed Docusaurus site. This involves navigating through pages, verifying interactive elements, checking external links, and ensuring the overall user experience aligns with expectations.
    *   **Execution**: Utilize browser automation tools (e.g., Playwright, Cypress) against the GitHub Pages deployment.

## VII. Organization by Phases (Tasks Mapping)
**Goal**: Structure the implementation into logical phases, linking to tasks.md.
**Format**: High-level phases with major milestones.

**Phase 1: Setup & Tooling**
- **Milestone**: Claude CLI configured with Context7 access, basic Docusaurus project structure initialized, implement custom styling and landing page.
- **Major Activities**:
    - Ensure Claude CLI can access and query Context7 MCP Server.
    - Initialize a basic Docusaurus project.
    - Set up version control and initial repository structure.
    - Design and implement custom homepage with hero section
    - Create custom.css with brand colors and typography
    - Build feature cards component for landing page
    - Add project logo and favicon
    - Configure Docusaurus i18n plugin for Urdu language support
    - Set up /i18n/ur/ directory structure
    - Add language switcher to navbar with English/اردو toggle
    - Configure RTL CSS rules for Urdu text direction
    - Create translation guidelines document for maintaining consistency

**Deliverables:**

- [X] *Docusaurus i18n configured with Urdu locale*
- [X] *Language switcher functional in navbar*
- [X] *RTL CSS rules defined*
- [X] *Translation guidelines documented*

**Phase 2: Core Documentation Infrastructure**
- **Milestone**: Docusaurus site capable of building and deploying to GitHub Pages, basic content structure defined.
- **Major Activities**:
    - Configure Docusaurus build process.
    - Set up CI/CD for automated deployment to GitHub Pages.
    - Define initial sidebar structure and content organization based on general curriculum principles.
**Translation Workflow:**
1. Complete English chapter first
2. Extract translatable content (excluding code blocks)
3. Translate chapter content to Urdu
4. Add Urdu comments to complex code examples
5. Review translation with native Urdu technical reviewer
6. Test RTL rendering and fix layout issues
7. Mark chapter as translation-complete

### Phase 2.5: Content Translation (Weeks 6-9) OR (Parallel with Phase 2)

*Goals:*
- Translate all 13 weeks of English content to Urdu
- Ensure technical accuracy and readability
- Maintain consistent terminology across chapters
- Verify RTL layout integrity

*Activities:*

*Translation Workflow:*
1. Complete English chapter first
2. Extract translatable content (excluding code blocks)
3. Translate chapter content to Urdu
4. Add Urdu comments to complex code examples
5. Review translation with native Urdu technical reviewer
6. Test RTL rendering and fix layout issues
7. Mark chapter as translation-complete

*Week-by-Week Translation Schedule:*
- Week 6: Translate Weeks 1-2 content (Introduction to Physical AI)
- Week 7: Translate Weeks 3-5 content (ROS 2 Fundamentals)
- Week 8: Translate Weeks 6-7 content (Gazebo Simulation)
- Week 9: Translate Weeks 8-13 content (Isaac, Humanoid, Conversational AI)

*Deliverables:*
- [ ] All 13 weeks translated to Urdu
- [X] Translation review completed by native speakers
- [X] Technical terminology glossary (English-Urdu)
- [X] RTL layout tested on all pages
- [X] Language switcher tested across all routes

**Phase 3: Content Generation & Integration**
- **Milestone**: Initial documentation content generated, adhering to Context7 Docusaurus guidelines.
- **Major Activities**:
    - Claude CLI uses Context7 to fetch and apply Docusaurus configuration, MDX rules, component usage, and theme customization examples.
    - Generate initial documentation pages, ensuring adherence to Context7 patterns.
    - Integrate initial code examples and textual explanations.

**Phase 4: Reproducibility Implementation**
- **Milestone**: All documentation examples are reproducible on the target environment; automated reproducibility checks are in place.
- **Major Activities**:
    - Set up a repeatable Ubuntu 22.04 + ROS 2 Humble environment for testing.
    - Implement and verify full path structures for all simulation files (URDF, SDF, launch files).
    - Develop and integrate clear dependency installation commands for all examples.
    - Implement automated scripts for executing and validating examples in the target environment.

**Phase 5: Quality Assurance & Refinement**
- **Milestone**: Documentation meets all quality and reproducibility criteria; ready for user consumption.
- **Major Activities**:
    - Conduct comprehensive automated quality checks (Docusaurus build, MDX linting, readability, consistency).
    - Perform E2E tests for user journey validation.
    - Conduct final review cycles and address any identified issues.
    - Iterate and refine content based on testing feedback.
