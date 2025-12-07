import React from 'react';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';

/**
 * Component for displaying reproducible code examples with multiple language/package manager options
 */
export default function CodeExample({title, children, language="bash"}) {
  return (
    <div className="code-example">
      <h4>{title}</h4>
      <CodeBlock language={language} showLineNumbers>
        {children}
      </CodeBlock>
    </div>
  );
}