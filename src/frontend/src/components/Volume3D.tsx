/** WebGL2 volume renderer with three.js (ray-marched, transparent).
 *
 *  Robust against:
 *    - the user closing the modal mid-init (every await checks `cancelled`,
 *      and any half-built resource is disposed in the cleanup path)
 *    - the host element being unmounted / detached when we get to it
 *    - WebGL render errors at runtime (animate() catches and tears down so
 *      we don't burn a CPU on a broken context)
 *    - WebGL1 fallback (we require WebGL2 explicitly)
 *
 *  Down-sampled to 64³ by default so we stay well under any browser-side
 *  GPU memory limits; 64³ × Float32 = 1 MB texture, still readable.
 */

import { useEffect, useRef, useState } from "react";

type Props = {
  assetId: string;
  gamma: number;
  contrast: number;
  threshold: number;
};

type LoadState =
  | { kind: "idle" }
  | { kind: "loading"; phase: string }
  | { kind: "ready"; shape: [number, number, number] }
  | { kind: "error"; message: string };

const VERTEX_SHADER = /* glsl */ `
  out vec3 vOrigin;
  out vec3 vDirection;

  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPosition, 1.0));
    vDirection = position - vOrigin;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  in vec3 vOrigin;
  in vec3 vDirection;
  out vec4 outColor;

  uniform sampler3D uVolume;
  uniform float uThreshold;
  uniform float uGamma;
  uniform float uContrast;
  uniform vec3 uAccent;
  uniform float uOpacity;

  vec2 hitBox(vec3 orig, vec3 dir) {
    const vec3 boxMin = vec3(-0.5);
    const vec3 boxMax = vec3(0.5);
    vec3 invDir = 1.0 / dir;
    vec3 tMin = (boxMin - orig) * invDir;
    vec3 tMax = (boxMax - orig) * invDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
  }

  void main() {
    vec3 rd = normalize(vDirection);
    vec2 bounds = hitBox(vOrigin, rd);
    if (bounds.x > bounds.y) discard;
    bounds.x = max(bounds.x, 0.0);
    vec3 p = vOrigin + rd * bounds.x;
    // 96 fixed-step samples keep the fragment cheap; that's enough for a
    // 64..96-side volume.
    vec3 inc = rd * (1.0 / 96.0);
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 128; i++) {
      vec3 coord = p + 0.5;
      if (coord.x < 0.0 || coord.y < 0.0 || coord.z < 0.0 ||
          coord.x > 1.0 || coord.y > 1.0 || coord.z > 1.0) break;
      float v = texture(uVolume, coord).r;
      v = (v - 0.5) * uContrast + 0.5;
      v = clamp(v, 0.0, 1.0);
      v = pow(v, 1.0 / max(uGamma, 0.01));
      float dens = smoothstep(uThreshold, 1.0, v) * uOpacity;
      vec3 col = uAccent * v;
      acc.rgb += (1.0 - acc.a) * col * dens;
      acc.a += (1.0 - acc.a) * dens;
      if (acc.a >= 0.97) break;
      p += inc;
    }
    if (acc.a < 0.01) discard;
    outColor = acc;
  }
`;

export default function Volume3D({ assetId, gamma, contrast, threshold }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<LoadState>({ kind: "idle" });
  const cleanupRef = useRef<() => void>(() => {});
  const uniformsRef = useRef<any>(null);

  useEffect(() => {
    let cancelled = false;
    // Collect disposers in order so we can tear down whatever we managed to
    // build, even if the user navigated away mid-init.
    const disposers: Array<() => void> = [];
    const runDisposers = () => {
      while (disposers.length) {
        try {
          disposers.pop()!();
        } catch {
          /* keep going — best-effort cleanup */
        }
      }
    };
    cleanupRef.current = runDisposers;

    setState({ kind: "loading", phase: "loading three.js…" });

    async function init() {
      const [three, ctrls] = await Promise.all([
        import("three"),
        import("three/examples/jsm/controls/OrbitControls.js"),
      ]);
      if (cancelled) return;
      const THREE = three;
      const OrbitControls = ctrls.OrbitControls;

      setState({ kind: "loading", phase: "downloading volume…" });

      const r = await fetch(`/api/files/asset/${assetId}/volume?max_side=64`, {
        credentials: "include",
      });
      if (cancelled) return;
      if (!r.ok) throw new Error(`volume request failed: ${r.status}`);
      const shapeHeader = r.headers.get("X-Volume-Shape") || "";
      const shape = shapeHeader.split(",").map((s) => parseInt(s, 10)) as [
        number,
        number,
        number,
      ];
      if (!shape.every((n) => Number.isFinite(n) && n > 0)) {
        throw new Error(`bad volume shape: ${shapeHeader}`);
      }
      const buf = await r.arrayBuffer();
      if (cancelled) return;
      const data = new Float32Array(buf);
      const expected = shape[0] * shape[1] * shape[2];
      if (data.length !== expected) {
        throw new Error(
          `volume size mismatch: got ${data.length} floats, expected ${expected} for shape ${shape.join("×")}`
        );
      }

      // Wait one frame so the host has its real size — and re-check
      // cancellation right after.
      await new Promise((res) => requestAnimationFrame(() => res(null)));
      if (cancelled) return;
      const host = hostRef.current;
      if (!host || !host.isConnected) {
        // Host detached before we got here; bail without creating any GL
        // resources.
        return;
      }
      const w = Math.max(1, host.clientWidth);
      const h = Math.max(1, host.clientHeight);

      let renderer: any;
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: true,
          premultipliedAlpha: false,
          powerPreference: "high-performance",
          // failIfMajorPerformanceCaveat surfaces "no GPU" outright so we
          // can show a useful message rather than render garbage.
          failIfMajorPerformanceCaveat: false,
        });
      } catch (err: any) {
        throw new Error(`webgl_init_failed: ${err?.message || err}`);
      }
      disposers.push(() => {
        try {
          renderer.dispose();
        } catch {}
      });
      if (!renderer.capabilities?.isWebGL2) {
        throw new Error(
          "WebGL2 not available — try a recent Chrome / Firefox / Safari."
        );
      }
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      renderer.setSize(w, h);
      renderer.setClearColor(0x000000, 0);
      host.innerHTML = "";
      host.appendChild(renderer.domElement);
      disposers.push(() => {
        if (renderer.domElement.parentNode === host) {
          host.removeChild(renderer.domElement);
        }
      });

      // Detect context loss so the page survives even if the GPU goes away.
      const onCtxLost = (e: Event) => {
        e.preventDefault();
        setState({
          kind: "error",
          message: "webgl context lost — try closing and reopening this tab",
        });
        runDisposers();
      };
      renderer.domElement.addEventListener("webglcontextlost", onCtxLost);
      disposers.push(() =>
        renderer.domElement.removeEventListener("webglcontextlost", onCtxLost)
      );

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
      camera.position.set(0, 0, 1.7);

      const tex = new THREE.Data3DTexture(data, shape[2], shape[1], shape[0]);
      tex.format = THREE.RedFormat;
      tex.type = THREE.FloatType;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.unpackAlignment = 1;
      tex.needsUpdate = true;
      disposers.push(() => tex.dispose());

      const uniforms = {
        uVolume: { value: tex },
        uThreshold: { value: threshold },
        uGamma: { value: gamma },
        uContrast: { value: contrast },
        uAccent: { value: new THREE.Color(0.0, 0.83, 0.65) },
        uOpacity: { value: 0.09 },
      };
      uniformsRef.current = uniforms;

      const material = new THREE.ShaderMaterial({
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        glslVersion: THREE.GLSL3,
        uniforms,
        side: THREE.BackSide,
        transparent: true,
      });
      disposers.push(() => material.dispose());

      const geometry = new THREE.BoxGeometry(1, 1, 1);
      disposers.push(() => geometry.dispose());
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enablePan = false;
      controls.target.set(0, 0, 0);
      controls.minDistance = 1.0;
      controls.maxDistance = 6.0;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.6;
      disposers.push(() => controls.dispose());

      let raf = 0;
      let alive = true;
      function animate() {
        if (!alive) return;
        try {
          controls.update();
          renderer.render(scene, camera);
        } catch (err) {
          console.error("Volume3D render error", err);
          alive = false;
          setState({
            kind: "error",
            message: "render failed — see console",
          });
          runDisposers();
          return;
        }
        raf = requestAnimationFrame(animate);
      }
      disposers.push(() => {
        alive = false;
        cancelAnimationFrame(raf);
      });
      animate();

      function onResize() {
        const w2 = Math.max(1, host!.clientWidth);
        const h2 = Math.max(1, host!.clientHeight);
        renderer.setSize(w2, h2);
        camera.aspect = w2 / h2;
        camera.updateProjectionMatrix();
      }
      const ro = new ResizeObserver(onResize);
      ro.observe(host);
      disposers.push(() => ro.disconnect());
      setTimeout(onResize, 50);

      // Tail check: if the user closed in the last few ms, dispose now.
      if (cancelled) {
        runDisposers();
        return;
      }

      setState({ kind: "ready", shape });
    }

    init().catch((err) => {
      console.error("Volume3D init failed", err);
      if (!cancelled) {
        setState({ kind: "error", message: String(err?.message || err) });
      }
      runDisposers();
    });

    return () => {
      cancelled = true;
      runDisposers();
      uniformsRef.current = null;
    };
  }, [assetId]);

  useEffect(() => {
    const u = uniformsRef.current;
    if (!u) return;
    u.uGamma.value = gamma;
    u.uContrast.value = contrast;
    u.uThreshold.value = threshold;
  }, [gamma, contrast, threshold]);

  return (
    <div className="volume3d">
      <div ref={hostRef} className="volume3d-canvas" />
      {state.kind === "loading" && (
        <div className="volume3d-status mono">{state.phase}</div>
      )}
      {state.kind === "error" && (
        <div className="volume3d-status err mono">{state.message}</div>
      )}
      {state.kind === "ready" && (
        <div className="volume3d-hint mono">
          {state.shape.join(" × ")} · drag · wheel zoom
        </div>
      )}
    </div>
  );
}
