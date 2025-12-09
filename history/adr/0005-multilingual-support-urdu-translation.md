# ADR-0005: Multilingual Support Implementation Strategy

## Status

Accepted

## Date

2025-12-07

## Context

The Physical AI & Humanoid Robotics documentation needs to support Urdu (اردو) as a secondary language alongside English to make the content accessible to Pakistani and South Asian learners. This requires implementing a robust internationalization (i18n) solution that can handle right-to-left (RTL) text direction, technical terminology translation guidelines, and maintain content synchronization between languages.

The decision impacts how all documentation content will be structured, how the UI will adapt to different languages, and how the build/deployment process will handle multiple locales.

## Decision

We will implement Urdu translation support using Docusaurus' built-in i18n plugin with the following approach:

**Translation approach**: Human translation with technical review - all content will be translated by native Urdu speakers with technical background, not machine translation, to ensure accuracy of complex robotics and AI concepts.

**File structure**: Use Docusaurus standard i18n directory structure with files in `/i18n/ur/docusaurus-plugin-content-docs/current/` to maintain consistency with Docusaurus conventions and enable automatic locale routing.

**RTL layout handling**: Implement RTL CSS rules using `[dir="rtl"]` selectors to ensure proper text alignment, navigation flow, and UI element positioning when Urdu is selected.

**Technical terminology**: Keep English technical terms (ROS 2, URDF, Gazebo, etc.) with Urdu explanations in parentheses on first usage, following the requirement in spec.md FR-015 to maintain technical accuracy while providing accessibility.

**Maintenance strategy**: Implement a workflow where English content is completed first, then translated to Urdu with native reviewer validation, ensuring translations stay synchronized with English updates.

**Deployment**: Single site with locale prefixes (e.g., `/ur/week-01/intro` for Urdu, `/week-01/intro` for English) to maintain unified infrastructure while providing language-specific experiences.

## Consequences

### Positive
- Enables access to Urdu-speaking students and professionals in robotics and AI fields
- Maintains technical accuracy through human translation with technical review
- Follows Docusaurus best practices for i18n, ensuring long-term maintainability
- Single deployment infrastructure reduces operational complexity
- RTL support will benefit other RTL languages in the future
- Clear workflow ensures translations remain synchronized with English content

### Negative
- Translation process requires additional time and human resources
- Maintenance overhead increases as content must be kept synchronized across languages
- RTL CSS implementation requires additional testing across browsers and devices
- Technical terminology consistency requires careful management across all chapters
- Build times may increase due to multiple locale processing

### Risks
- Translation quality depends on availability of native Urdu speakers with technical background
- Keeping translations synchronized with frequent English updates requires discipline
- RTL layout may introduce visual issues that are difficult to catch during development

## Alternatives

**Alternative 1: Machine translation with post-editing**
- Use automated translation services with human review
- Pros: Faster translation process, lower cost
- Cons: Risk of technical inaccuracy, complex concepts may not translate well

**Alternative 2: Separate repository for Urdu content**
- Maintain Urdu translations in a separate repository
- Pros: Isolated development, easier to manage permissions
- Cons: More complex deployment, harder to keep synchronized, increased maintenance

**Alternative 3: Third-party translation platform integration**
- Use services like Crowdin or Transifex for translation management
- Pros: Streamlined translation workflow, collaboration tools
- Cons: Additional dependency, potential cost, complexity of integrating with Docusaurus

## References

- spec.md: Internationalization Requirements (FR-011 through FR-020)
- plan.md: Phase 1 Setup & Tooling (lines 174-178) and Phase 2.5 Content Translation
- tasks.md: User Story 3: Provide Urdu Language Support and i18n Setup and Configuration tasks