import React from 'react';

/**
 * Component for displaying system requirements in a consistent format
 */
export default function SystemRequirements({requirements}) {
  return (
    <div className="system-requirements alert alert--info" role="alert">
      <h4>System Requirements</h4>
      <ul>
        {requirements.map((req, index) => (
          <li key={index}>{req}</li>
        ))}
      </ul>
    </div>
  );
}