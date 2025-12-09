# Translation Guidelines: English to Urdu

## Purpose
This document provides guidelines for translating the Physical AI & Humanoid Robotics documentation from English to Urdu. These guidelines ensure consistency, accuracy, and cultural appropriateness across all translated content.

## Technical Terminology Handling

### English Terms with Urdu Explanations
- Keep all technical terms in English (ROS 2, Gazebo, URDF, SDF, etc.)
- Provide Urdu explanation in parentheses on first usage in each chapter
- Example: "ROS 2 (روبوٹ آپریٹنگ سسٹم ورژن 2) is a middleware framework..."

### Consistent Transliteration
- Use consistent transliteration for technical concepts
- Common terms reference:
  - Node: نوڈ
  - Topic: ٹاپک
  - Service: سروس
  - Action: ایکشن
  - Package: پیکیج
  - Launch file: لانچ فائل
  - Simulation: سیمولیشن
  - Robot: روبوٹ

## Code Comment Standards

### Inline Urdu Comments
- Add Urdu comments to complex code examples where beneficial
- Comments should explain the logic, not just translate variable names
- Use standard Urdu punctuation
- Format: // Urdu explanation of complex logic

### Documentation in Code Examples
- Preserve English variable names and function names
- Add Urdu comments for complex algorithms or non-obvious operations
- Example:
```python
# ROS 2 نوڈ کو شروع کرتا ہے اور اس کا نام 'minimal_publisher' ہے
rclpy.init()
```

## Translation Quality Standards

### Accuracy Requirements
- All technical translations must be reviewed by native Urdu speakers with technical background
- Ensure translated concepts maintain technical accuracy
- Avoid literal translations that may lose technical meaning

### Readability Guidelines
- Maintain Flesch-Kincaid grade 8-10 reading level equivalent in Urdu
- Use simple, clear Urdu that is accessible to beginners
- Avoid overly complex sentence structures

## Review Checklist

### Pre-publication Review
- [ ] Technical terminology consistently handled (English + Urdu explanation)
- [ ] Code examples include appropriate Urdu comments
- [ ] Translations reviewed by native Urdu speaker with technical background
- [ ] All content maintains technical accuracy
- [ ] No cultural insensitivities or inappropriate translations
- [ ] Links and references still work correctly
- [ ] Code syntax remains unchanged and functional

### Consistency Checks
- [ ] Technical terms translated consistently across all chapters
- [ ] Transliteration of common terms remains the same
- [ ] Urdu explanations for technical terms are clear and accurate
- [ ] Tone and style consistent with original documentation

## Workflow

1. Complete English chapter first
2. Translate content while preserving technical accuracy
3. Add Urdu comments to complex code examples
4. Review translation with native Urdu technical reviewer
5. Test RTL rendering and fix layout issues
6. Mark chapter as translation-complete