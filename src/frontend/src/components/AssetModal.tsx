/** Asset inspection modal — multi-tab preview that adapts to the asset's
 *  media type. Images & volumes get slice / projection / 3D tabs; video and
 *  audio get an inline player; PDFs render in an iframe. All variants share
 *  a metadata tab.
 */

import { lazy, Suspense, useEffect, useMemo, useState } from "react";

type MediaType = "image" | "volume" | "video" | "audio" | "pdf" | "other";

type AssetInfo = {
  asset_id: string;
  display_name?: string | null;
  original_format?: string | null;
  metadata_text?: string | null;
  media_type: MediaType;
  file_size?: number | null;
  ndim: number;
  shape: number[];
  dtype: string;
  intensity_min: number;
  intensity_max: number;
  is_rgb: boolean;
  is_volume: boolean;
  axes: { z: number | null; y: number | null; x: number | null };
  extra: Record<string, unknown>;
};

type Tab = "preview" | "slices" | "mip" | "3d" | "metadata";

type Props = {
  assetId: string | null;
  fallbackPreviewUrl?: string | null;
  fallbackName?: string | null;
  onClose: () => void;
};

const Volume3D = lazy(() => import("./Volume3D"));

function viewUrl(
  assetId: string,
  kind: "slice" | "mip",
  axis: "x" | "y" | "z",
  index: number,
  gamma: number,
  contrast: number
): string {
  const u = new URLSearchParams({
    kind,
    axis,
    index: String(index),
    gamma: gamma.toFixed(2),
    contrast: contrast.toFixed(2),
  });
  return `/api/files/asset/${assetId}/view?${u.toString()}`;
}

function rawUrl(assetId: string): string {
  return `/api/files/asset/${assetId}/raw`;
}

function fmtSize(b?: number | null): string {
  if (!b || !Number.isFinite(b)) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = b;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 100 ? 0 : 1)} ${u[i]}`;
}

export default function AssetModal({
  assetId,
  fallbackPreviewUrl,
  fallbackName,
  onClose,
}: Props) {
  const [info, setInfo] = useState<AssetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("preview");
  const [axis, setAxis] = useState<"x" | "y" | "z">("z");
  const [slice, setSlice] = useState<number>(0);
  const [gamma, setGamma] = useState<number>(1.0);
  const [contrast, setContrast] = useState<number>(1.0);
  const [threshold, setThreshold] = useState<number>(0.25);

  useEffect(() => {
    if (!assetId) return;
    setInfo(null);
    setError(null);
    setTab("preview");
    setAxis("z");
    setSlice(0);
    setGamma(1.0);
    setContrast(1.0);
    setThreshold(0.25);
    void (async () => {
      try {
        const r = await fetch(`/api/files/asset/${assetId}/info`, {
          credentials: "include",
        });
        if (!r.ok) throw new Error(`info ${r.status}`);
        const data: AssetInfo = await r.json();
        setInfo(data);
        if (data.is_volume) {
          const dim = data.axes.z ?? data.axes.y ?? data.axes.x ?? 1;
          setSlice(Math.floor(dim / 2));
        }
      } catch (e: any) {
        setError(String(e?.message || e));
      }
    })();
  }, [assetId]);

  useEffect(() => {
    if (!assetId) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [assetId, onClose]);

  useEffect(() => {
    if (!info?.is_volume) return;
    const n =
      axis === "z" ? info.axes.z : axis === "y" ? info.axes.y : info.axes.x;
    if (n != null) setSlice(Math.floor(n / 2));
  }, [axis, info]);

  const sliceMax = useMemo(() => {
    if (!info?.is_volume) return 0;
    const n =
      axis === "z" ? info.axes.z : axis === "y" ? info.axes.y : info.axes.x;
    return Math.max(0, (n ?? 1) - 1);
  }, [info, axis]);

  if (!assetId) return null;

  const mt: MediaType = info?.media_type ?? "image";
  const isVolume = mt === "volume" || !!info?.is_volume;
  const isImagery = mt === "image" || mt === "volume";

  const sliceSrc =
    info && tab === "slices"
      ? viewUrl(assetId, "slice", axis, slice, gamma, contrast)
      : null;
  const mipSrc =
    info && tab === "mip"
      ? viewUrl(assetId, "mip", axis, 0, gamma, contrast)
      : null;

  const displayName = info?.display_name ?? fallbackName ?? "asset";

  // Tab ordering depends on media type.
  const tabs: { id: Tab; label: string; enabled: boolean }[] = (() => {
    if (mt === "video" || mt === "audio") {
      return [
        { id: "preview", label: "player", enabled: true },
        { id: "metadata", label: "metadata", enabled: true },
      ];
    }
    if (mt === "pdf") {
      return [
        { id: "preview", label: "viewer", enabled: true },
        { id: "metadata", label: "metadata", enabled: true },
      ];
    }
    return [
      { id: "preview", label: "preview", enabled: true },
      { id: "slices", label: "slices", enabled: isVolume },
      { id: "mip", label: "projections", enabled: isVolume },
      { id: "3d", label: "3d", enabled: isVolume },
      { id: "metadata", label: "metadata", enabled: true },
    ];
  })();

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal asset-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <header className="modal-head">
          <div className="modal-tag mono">
            {mt} · {info?.original_format || "—"}
          </div>
          <button className="modal-x" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <h2 className="modal-title asset-title" title={displayName}>
          {displayName}
        </h2>
        {info && (
          <p className="modal-sub mono">
            {isImagery && info.ndim > 0
              ? `${info.ndim}D · ${info.shape.join(" × ")} · ${info.dtype} · range [${info.intensity_min.toFixed(1)}, ${info.intensity_max.toFixed(1)}]`
              : `${(mt || "file").toUpperCase()} · ${fmtSize(info.file_size)}`}
          </p>
        )}

        <nav className="asset-tabs mono" role="tablist">
          {tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              className={
                "tab " +
                (tab === t.id ? "active" : "") +
                (!t.enabled ? " disabled" : "")
              }
              disabled={!t.enabled}
              onClick={() => t.enabled && setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {!info && !error && (
          <div className="asset-loading mono">loading info…</div>
        )}
        {error && (
          <div className="error-banner" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}

        {info && (
          <div className="asset-body">
            {/* Preview / player / pdf */}
            {tab === "preview" && (
              <div className="asset-view">
                {mt === "video" ? (
                  <video
                    controls
                    src={rawUrl(assetId)}
                    className="asset-media"
                  />
                ) : mt === "audio" ? (
                  <audio
                    controls
                    src={rawUrl(assetId)}
                    className="asset-audio"
                  />
                ) : mt === "pdf" ? (
                  <iframe
                    src={rawUrl(assetId)}
                    className="asset-pdf"
                    title={displayName}
                  />
                ) : fallbackPreviewUrl ? (
                  <img
                    className="asset-img"
                    src={fallbackPreviewUrl}
                    alt={displayName}
                  />
                ) : (
                  <div className="asset-empty mono">no preview available</div>
                )}
              </div>
            )}

            {tab === "slices" && isVolume && (
              <>
                <div className="asset-view">
                  <img
                    className="asset-img"
                    src={sliceSrc ?? ""}
                    alt="slice"
                    loading="eager"
                  />
                </div>
                <div className="asset-controls">
                  <div className="control-row">
                    <label className="ctrl-label mono">axis</label>
                    <div className="seg">
                      {(["z", "y", "x"] as const).map((a) => (
                        <button
                          key={a}
                          type="button"
                          className={"seg-btn " + (axis === a ? "active" : "")}
                          onClick={() => setAxis(a)}
                        >
                          {a}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="control-row">
                    <label className="ctrl-label mono">
                      slice · {slice + 1} / {sliceMax + 1}
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={sliceMax}
                      value={slice}
                      onChange={(e) => setSlice(parseInt(e.target.value, 10))}
                    />
                  </div>
                  <GammaContrast
                    gamma={gamma}
                    setGamma={setGamma}
                    contrast={contrast}
                    setContrast={setContrast}
                  />
                </div>
              </>
            )}

            {tab === "mip" && isVolume && (
              <>
                <div className="asset-view">
                  <img
                    className="asset-img"
                    src={mipSrc ?? ""}
                    alt="maximum intensity projection"
                  />
                </div>
                <div className="asset-controls">
                  <div className="control-row">
                    <label className="ctrl-label mono">project along</label>
                    <div className="seg">
                      {(["z", "y", "x"] as const).map((a) => (
                        <button
                          key={a}
                          type="button"
                          className={"seg-btn " + (axis === a ? "active" : "")}
                          onClick={() => setAxis(a)}
                        >
                          {a}
                        </button>
                      ))}
                    </div>
                  </div>
                  <GammaContrast
                    gamma={gamma}
                    setGamma={setGamma}
                    contrast={contrast}
                    setContrast={setContrast}
                  />
                </div>
              </>
            )}

            {tab === "3d" && isVolume && (
              <>
                <div className="asset-view volume-host">
                  <Suspense
                    fallback={
                      <div className="asset-loading mono">loading three.js…</div>
                    }
                  >
                    <Volume3D
                      assetId={assetId}
                      gamma={gamma}
                      contrast={contrast}
                      threshold={threshold}
                    />
                  </Suspense>
                </div>
                <div className="asset-controls">
                  <div className="control-row">
                    <label className="ctrl-label mono">
                      threshold · {threshold.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min={0.0}
                      max={0.95}
                      step={0.01}
                      value={threshold}
                      onChange={(e) =>
                        setThreshold(parseFloat(e.target.value))
                      }
                    />
                  </div>
                  <GammaContrast
                    gamma={gamma}
                    setGamma={setGamma}
                    contrast={contrast}
                    setContrast={setContrast}
                  />
                </div>
              </>
            )}

            {tab === "metadata" && (
              <div className="asset-meta">
                <KV label="display name" value={displayName} />
                <KV label="format" value={info.original_format || "—"} />
                <KV label="media type" value={mt} />
                <KV label="file size" value={fmtSize(info.file_size)} />
                {isImagery && info.ndim > 0 && (
                  <>
                    <KV label="ndim" value={String(info.ndim)} />
                    <KV label="shape" value={info.shape.join(" × ")} />
                    <KV label="dtype" value={info.dtype} />
                    <KV label="rgb" value={info.is_rgb ? "yes" : "no"} />
                    <KV label="volume" value={info.is_volume ? "yes" : "no"} />
                    <KV
                      label="intensity range"
                      value={`${info.intensity_min.toFixed(2)} … ${info.intensity_max.toFixed(2)}`}
                    />
                  </>
                )}
                {info.metadata_text && (
                  <KV label="agent summary" value={info.metadata_text} multiline />
                )}
                {Object.entries(info.extra).length > 0 && (
                  <details className="asset-extra-details">
                    <summary className="mono">extra metadata</summary>
                    <pre className="asset-extra">
                      {JSON.stringify(info.extra, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function GammaContrast({
  gamma,
  setGamma,
  contrast,
  setContrast,
}: {
  gamma: number;
  setGamma: (n: number) => void;
  contrast: number;
  setContrast: (n: number) => void;
}) {
  return (
    <>
      <div className="control-row">
        <label className="ctrl-label mono">gamma · {gamma.toFixed(2)}</label>
        <input
          type="range"
          min={0.3}
          max={3.0}
          step={0.05}
          value={gamma}
          onChange={(e) => setGamma(parseFloat(e.target.value))}
        />
      </div>
      <div className="control-row">
        <label className="ctrl-label mono">
          contrast · {contrast.toFixed(2)}
        </label>
        <input
          type="range"
          min={0.3}
          max={3.0}
          step={0.05}
          value={contrast}
          onChange={(e) => setContrast(parseFloat(e.target.value))}
        />
      </div>
      <div className="control-row">
        <button
          type="button"
          className="btn-ghost"
          onClick={() => {
            setGamma(1.0);
            setContrast(1.0);
          }}
        >
          reset
        </button>
      </div>
    </>
  );
}

function KV({
  label,
  value,
  multiline = false,
}: {
  label: string;
  value: string;
  multiline?: boolean;
}) {
  return (
    <div className="kv">
      <span className="kv-key mono">{label}</span>
      <span className={"kv-val" + (multiline ? " multiline" : "")}>
        {value}
      </span>
    </div>
  );
}
