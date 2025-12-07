import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'ROS 2 Fundamentals',
    description: (
      <>
        Learn the fundamentals of ROS 2, the robotics middleware that powers modern robotics applications.
      </>
    ),
  },
  {
    title: 'Gazebo Simulation',
    description: (
      <>
        Master robot simulation with Gazebo, including URDF/SDF robot descriptions and physics simulation.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac Platform',
    description: (
      <>
        Explore AI-powered perception and manipulation using NVIDIA Isaac SDK and Isaac Sim.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    description: (
      <>
        Understand humanoid robot kinematics, dynamics, and locomotion control techniques.
      </>
    ),
  },
  {
    title: 'Conversational AI',
    description: (
      <>
        Integrate GPT models and voice recognition for conversational robotics applications.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--2')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Why Physical AI?</h2>
            <p className={styles.sectionDescription}>
              Physical AI represents the next frontier in artificial intelligence, combining
              digital intelligence with physical embodiment. This curriculum provides a
              comprehensive pathway from basic concepts to advanced humanoid robotics applications.
            </p>
          </div>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}