/** Partner credits — EPFL, SDSC, and Imaging Plaza.
 *
 *  Attribution copy is explicit per logo:
 *    · developed by         · SDSC
 *    · in collaboration with· EPFL
 *    · a product of         · Imaging Plaza
 */

import { useState } from "react";
import { useTheme } from "../hooks/useTheme";

const EPFL_LOGO_DARK =
  "https://cdn.prod.website-files.com/63ea083f4e797cf53055586b/652d04a587d36e6c661fda44_logo_epfl_footer.svg";
const EPFL_LOGO_LIGHT =
  "https://www.epfl.ch/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg";

const SDSC_LOGO =
  "https://cdn.prod.website-files.com/63ea083f4e797cf53055586b/69df9c31885816580fe4151c_SDSC_Logo_RGB.svg";

const IMAGING_PLAZA_LOGO = "https://imaging-plaza.epfl.ch/logos/imaging_plaza.svg";

function ImagingPlazaFallback({ height = 18 }: { height?: number }) {
  return (
    <svg
      viewBox="0 0 154 28"
      height={height}
      aria-label="imaging-plaza"
      role="img"
      style={{ display: "block" }}
    >
      <rect x="2" y="4" width="20" height="20" rx="3" fill="var(--accent)" />
      <text
        x="12"
        y="19"
        textAnchor="middle"
        fontFamily="JetBrains Mono, ui-monospace, monospace"
        fontSize="11"
        fontWeight="700"
        fill="#02110d"
      >
        ip
      </text>
      <text
        x="30"
        y="19"
        fontFamily="JetBrains Mono, ui-monospace, monospace"
        fontSize="12"
        fontWeight="600"
        fill="currentColor"
        letterSpacing="0.2"
      >
        imaging-plaza
      </text>
    </svg>
  );
}

function ImagingPlazaLogo({ height = 18 }: { height?: number }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <ImagingPlazaFallback height={height} />;
  return (
    <img
      src={IMAGING_PLAZA_LOGO}
      alt="Imaging Plaza"
      height={height}
      style={{ height, display: "block", width: "auto" }}
      onError={() => setFailed(true)}
    />
  );
}

function EpflLogo({ height = 18 }: { height?: number }) {
  const { theme } = useTheme();
  const src = theme === "dark" ? EPFL_LOGO_DARK : EPFL_LOGO_LIGHT;
  return (
    <img
      src={src}
      alt="EPFL"
      height={height}
      style={{ height, display: "block", width: "auto" }}
    />
  );
}

function SdscLogo({ height = 18 }: { height?: number }) {
  return (
    <img
      src={SDSC_LOGO}
      alt="SDSC"
      height={height}
      style={{ height, display: "block", width: "auto" }}
    />
  );
}

/** Compact ribbon used in the login screen. Three labeled credits stacked
 *  horizontally with their respective marks. */
export default function PartnerStrip({ compact = false }: { compact?: boolean }) {
  return (
    <div className={"partner-strip" + (compact ? " compact" : "")}>
      <div className="partner-credit">
        <span className="partner-label">developed by</span>
        <a
          href="https://datascience.ch"
          target="_blank"
          rel="noreferrer"
          className="partner-card"
          title="Swiss Data Science Center"
          data-logo="sdsc"
        >
          <SdscLogo />
        </a>
      </div>
      <div className="partner-credit">
        <span className="partner-label">in collaboration with</span>
        <a
          href="https://www.epfl.ch"
          target="_blank"
          rel="noreferrer"
          className="partner-card"
          title="École polytechnique fédérale de Lausanne"
          data-logo="epfl"
        >
          <EpflLogo />
        </a>
      </div>
      <div className="partner-credit">
        <span className="partner-label">a product of</span>
        <a
          href="https://imaging-plaza.epfl.ch"
          target="_blank"
          rel="noreferrer"
          className="partner-card"
          title="imaging-plaza.epfl.ch"
          data-logo="imaging-plaza"
          style={{ color: "var(--text-soft)" }}
        >
          <ImagingPlazaLogo />
        </a>
      </div>
    </div>
  );
}

/** Vertical credits pinned to the bottom of the sidebar. Three rows of
 *  label + logo, each labeled. */
export function SidebarPartners() {
  return (
    <div className="sidebar-partners">
      <div className="sidebar-partner-row">
        <span className="sidebar-partner-label">developed by</span>
        <a
          href="https://datascience.ch"
          target="_blank"
          rel="noreferrer"
          className="partner-card sidebar-partner"
          title="Swiss Data Science Center"
          data-logo="sdsc"
        >
          <SdscLogo height={14} />
        </a>
      </div>
      <div className="sidebar-partner-row">
        <span className="sidebar-partner-label">in collaboration with</span>
        <a
          href="https://www.epfl.ch"
          target="_blank"
          rel="noreferrer"
          className="partner-card sidebar-partner"
          title="EPFL"
          data-logo="epfl"
        >
          <EpflLogo height={14} />
        </a>
      </div>
      <div className="sidebar-partner-row">
        <span className="sidebar-partner-label">a product of</span>
        <a
          href="https://imaging-plaza.epfl.ch"
          target="_blank"
          rel="noreferrer"
          className="partner-card sidebar-partner sidebar-partner-wide"
          title="imaging-plaza.epfl.ch"
          data-logo="imaging-plaza"
          style={{ color: "var(--text-soft)" }}
        >
          <ImagingPlazaLogo height={14} />
        </a>
      </div>
    </div>
  );
}
