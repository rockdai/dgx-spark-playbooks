"use client";

import LiquidGlass from "liquid-glass-react";
import styles from "./top-bar.module.css";

export function TopBar() {
  return (
    <header className={styles.headerWrap}>
      <LiquidGlass
        className={styles.glassShell}
        displacementScale={72}
        blurAmount={0.08}
        saturation={140}
        aberrationIntensity={1.4}
        elasticity={0.18}
        cornerRadius={28}
        padding="0"
        mode="standard"
      >
        <div className={styles.header}>
          <a className={styles.brand} href="/">
            DGX Spark 中文社区
          </a>
          <nav className={styles.nav}>
            <a className={styles.navLink} href="/intro">
              在线文档
            </a>
            <a
              className={styles.buyButton}
              href="https://common-buy.aliyun.com/?commodityCode=datav_spark_public_cn"
              target="_blank"
              rel="noreferrer"
            >
              立即购买
            </a>
          </nav>
        </div>
      </LiquidGlass>
    </header>
  );
}
