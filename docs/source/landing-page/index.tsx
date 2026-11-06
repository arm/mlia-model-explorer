// SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import ThemedImage from "@theme/ThemedImage";
import type { ReactElement } from "react";
import { landing } from "./_homepage";
import inkLogo from "./mlia-logo-stacked-ink.png";
import whiteLogo from "./mlia-logo-stacked-white.png";
import styles from "./index.module.css";

export default function Home(): ReactElement {
  return (
    <Layout title={landing.title} description={landing.lead}>
      <main className={styles.surface}>
        <div className={styles.page}>
          <div className={styles.brand}>
            <ThemedImage
              sources={{ light: inkLogo, dark: whiteLogo }}
              alt="Arm machine learning inference advisor"
              width="3300"
              height="432"
            />
          </div>
          <section className={styles.hero} aria-labelledby="landing-title">
            <div className={styles.heroCopy}>
              <p className={styles.eyebrow}>
                {landing.category} / {landing.title}
              </p>
              <h1 id="landing-title">{landing.headline}</h1>
              <p className={styles.lead}>{landing.lead}</p>
              <div className={styles.actions}>
                <Link className={styles.primaryButton} to={landing.start}>
                  Get started →
                </Link>
                <Link className={styles.secondaryButton} to="/documentation">
                  Browse documentation
                </Link>
              </div>
              <Link className={styles.repositoryLink} to={landing.repository}>
                View on GitHub →
              </Link>
            </div>
          </section>
          <section
            className={styles.analysisSection}
            aria-labelledby="focus-title"
          >
            <div className={styles.analysisHeader}>
              <p className={styles.eyebrow}>{landing.category}</p>
              <h2 id="focus-title">Work with {landing.title}.</h2>
            </div>
            <div className={styles.capabilities}>
              {landing.focus.map((item, index) => (
                <Link
                  className={styles.capabilityCard}
                  to={item.to}
                  key={item.to}
                >
                  <span className={styles.number}>0{index + 1}</span>
                  <h2>{item.title}</h2>
                  <p>{item.description}</p>
                  <span className={styles.cardLink}>Read the guide →</span>
                </Link>
              ))}
            </div>
          </section>
          <div className={styles.workspace}>
            <section
              className={styles.startSection}
              aria-labelledby="integration-title"
            >
              <div>
                <p className={styles.eyebrow}>Part of MLIA</p>
                <h2 id="integration-title">Start with the right setup.</h2>
                <p className={styles.sectionLead}>{landing.note}</p>
              </div>
              <div className={styles.textLinks}>
                <Link to={landing.start}>Usage guide →</Link>
                {landing.learning && (
                  <Link to={landing.learning}>Arm Learning Path →</Link>
                )}
                <Link to="https://developer.arm.com/dev-docs/mlia/">
                  MLIA core documentation →
                </Link>
              </div>
            </section>
            <section
              className={styles.pluginSection}
              aria-labelledby="guides-title"
            >
              <div className={styles.sectionHeader}>
                <p className={styles.eyebrow}>Documentation</p>
                <h2 id="guides-title">Find your next step.</h2>
              </div>
              <div className={styles.pluginGrid}>
                {landing.guides.map((guide) => (
                  <Link
                    className={styles.guideRow}
                    key={guide.to}
                    to={guide.to}
                  >
                    <span>{guide.title}</span>
                    <span aria-hidden="true">↗</span>
                  </Link>
                ))}
              </div>
            </section>
          </div>
          <section
            className={styles.contribute}
            aria-labelledby="development-title"
          >
            <div>
              <p className={styles.eyebrow}>Under the hood</p>
              <h2 id="development-title">Explore the implementation.</h2>
              <p>Find integration details and contributor guidance.</p>
            </div>
            <Link className={styles.secondaryButton} to="/development">
              Development →
            </Link>
          </section>
        </div>
      </main>
    </Layout>
  );
}
