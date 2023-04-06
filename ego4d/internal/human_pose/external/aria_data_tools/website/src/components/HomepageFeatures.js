/**
 * (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
 */

import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [

   {
    title: 'C++ and Python Tools ',
    Svg: require('../../static/img/python-brands.svg').default,
    description: (
      <>
        Analyze Project Aria VRS data with Python code and C++ libraries
      </>
    ),
  },

  {
    title: 'A Rich Sensor Suite',
    Svg: require('../../static/img/arrows-to-eye-solid.svg').default,
    description: (
      <>
       Read & visualize Project Aria sequences and sensor data
      </>
    ),
  },
  {
    title: '6DoF Transformations',
    Svg: require('../../static/img/glasses-solid.svg').default,
    description: (
      <>
        Retrieve calibration data and interact with Aria camera models
      </>
    ),
  },

];


function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
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
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
