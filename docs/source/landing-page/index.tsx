// SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

import SiteLink, { LinkIndicator } from "./_links";
import Layout from "@theme/Layout";
import type { ReactElement } from "react";
import { landing } from "./_homepage";
import styles from "./index.module.css";

export default function Home(): ReactElement {
  return (
    <Layout title={landing.title} description={landing.lead}>
      <main className={styles.surface}>
        <div className={styles.page}>
          <section className={styles.hero} aria-labelledby="landing-title">
            <div className={styles.heroCopy}>
              <p className={styles.eyebrow}>
                {landing.category} / {landing.title}
              </p>
              <h1 id="landing-title">Machine Learning Inference Advisor (MLIA)</h1>
              <p className={styles.heroTagline}>{landing.headline}</p>
              <p className={styles.lead}>{landing.lead}</p>
              <div className={styles.actions}>
                <SiteLink className={styles.primaryButton} to={landing.start}>
                  Get started
                </SiteLink>
                <SiteLink
                  className={styles.secondaryButton}
                  to="/documentation"
                >
                  Browse documentation
                </SiteLink>
              </div>
              <SiteLink
                className={styles.repositoryLink}
                to={landing.repository}
              >
                View on GitHub
              </SiteLink>
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
                <SiteLink
                  className={styles.capabilityCard}
                  indicator={false}
                  to={item.to}
                  key={item.to}
                >
                  <span className={styles.number}>0{index + 1}</span>
                  <h2>{item.title}</h2>
                  <p>{item.description}</p>
                  <span className={styles.cardLink}>
                    Read the guide <LinkIndicator to={item.to} />
                  </span>
                </SiteLink>
              ))}
            </div>
          </section>
          <div className={`${styles.workspace} ${styles.pairedPanels}`}>
            <section
              className={styles.startSection}
              aria-labelledby="integration-title"
            >
              <div>
                <p className={styles.eyebrow}>Part of MLIA</p>
                <h2 id="integration-title">{landing.panels.startTitle}</h2>
                <p className={styles.sectionLead}>
                  {landing.panels.startDescription}
                </p>
              </div>
              <div className={styles.panelLinks}>
                {landing.panels.startLinks.map((item) => (
                  <SiteLink to={item.to} key={item.to}>
                    {item.label}
                  </SiteLink>
                ))}
              </div>
            </section>
            <section
              className={`${styles.pluginSection} ${styles.taskPanel}`}
              aria-labelledby="guides-title"
            >
              <p className={styles.eyebrow}>Understand the view</p>
              <h2 id="guides-title">{landing.panels.resultsTitle}</h2>
              <p className={styles.panelDescription}>
                {landing.panels.resultsDescription}
              </p>
              <div className={styles.panelLinks}>
                {landing.panels.resultsLinks.map((item) => (
                  <SiteLink to={item.to} key={item.to}>
                    {item.label}
                  </SiteLink>
                ))}
              </div>
            </section>
          </div>
          <section
            className={styles.guideDirectory}
            aria-labelledby="all-guides-title"
          >
            <div className={styles.sectionHeader}>
              <p className={styles.eyebrow}>Documentation</p>
              <h2 id="all-guides-title">Explore all guides.</h2>
            </div>
            <div className={styles.pluginGrid}>
              {landing.guides.map((guide) => (
                <SiteLink
                  className={styles.guideRow}
                  to={guide.to}
                  key={guide.to}
                >
                  {guide.title}
                </SiteLink>
              ))}
            </div>
          </section>
          <section
            className={styles.contribute}
            aria-labelledby="development-title"
          >
            <div>
              <p className={styles.eyebrow}>Under the hood</p>
              <h2 id="development-title">Explore the implementation.</h2>
              <p>Find integration details and contributor guidance.</p>
            </div>
            <SiteLink className={styles.secondaryButton} to="/development">
              Development
            </SiteLink>
          </section>
        </div>
      </main>
    </Layout>
  );
}
