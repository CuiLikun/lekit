import * as THREE from "./three.module.min.js";
import { OrbitControls } from "./OrbitControls.js";
import { EngagementTrajectory } from "./teleop_node_monitor_trajectory.js";

const SIDES = ["left", "right"];
const COLORS = { left: 0x5bc9f5, right: 0xf5b84b, inactive: 0x56616e };
const MAX_REFRESH_HZ = 60;
const FRAME_INTERVAL_MS = 1000 / MAX_REFRESH_HZ;
const STREAM_RECONNECT_DELAY_MS = 1000;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let pendingStatus = null;
let statusSocket = null;
let reconnectTimer = null;
let lastDashboardFrameAt = null;

for (const side of SIDES) {
  const labels = side === "left" ? ["X", "Y", "≡"] : ["A", "B", "≡"];
  document.querySelector(`[data-details-for="${side}"]`).innerHTML = `
    <div class="input-grid">
      <div class="input-tile" data-tile="trigger">
        <div class="label">Trigger</div><div class="input-icon"><strong>☝</strong><span>index</span></div>
        <div class="meter" data-meter="trigger"><span></span></div>
      </div>
      <div class="input-tile" data-tile="squeeze">
        <div class="label">Squeeze</div><div class="input-icon"><strong>✊</strong><span>grip</span></div>
        <div class="meter" data-meter="squeeze"><span></span></div>
      </div>
      <div class="input-tile">
        <div class="label">Thumbstick</div>
        <div class="button-cluster"><span class="button-dot" data-detail-control="thumbstick_click">●</span><span data-stick-direction>•</span></div>
      </div>
      <div class="input-tile">
        <div class="label">Buttons</div>
        <div class="button-cluster">
          <span class="button-dot" data-detail-control="primary_button">${labels[0]}</span>
          <span class="button-dot" data-detail-control="secondary_button">${labels[1]}</span>
          <span class="button-dot" data-detail-control="menu_button">${labels[2]}</span>
        </div>
      </div>
    </div>
    <div class="exact-poses">
      <div class="exact-row"><span>grip position</span><span data-value="translation">—</span></div>
      <div class="exact-row"><span>grip xyzw</span><span data-value="rotation">—</span></div>
      <div class="exact-row"><span>aim position</span><span data-value="aim_translation">—</span></div>
      <div class="exact-row"><span>aim xyzw</span><span data-value="aim_rotation">—</span></div>
    </div>`;
}

function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, finiteNumber(value)));
}

function vector(action, key, size, identity = false) {
  const value = action[key];
  if (!Array.isArray(value) || value.length !== size) {
    return identity ? [0, 0, 0, 1] : Array(size).fill(0);
  }
  return value.map((item, index) => finiteNumber(item, identity && index === 3 ? 1 : 0));
}

function formatVector(value) {
  return value.map(item => finiteNumber(item).toFixed(3)).join(", ");
}

function setActive(element, active) {
  element?.classList.toggle("is-active", Boolean(active));
}

function updateController(side, action) {
  const panel = document.getElementById(`${side}-controller`);
  const key = field => `${side}.${field}`;
  const tracking = Boolean(action[key("is_tracking")]);
  const aimTracking = Boolean(action[key("is_aim_tracking")]);
  const engaged = Boolean(action[key("is_engaged")]);
  const squeeze = clamp(action[key("squeeze")], 0, 1);
  const trigger = clamp(action[key("trigger")], 0, 1);
  const thumbstick = vector(action, key("thumbstick"), 2);

  panel.dataset.tracking = String(tracking);
  setActive(panel.querySelector('[data-indicator="tracking"]'), tracking);
  setActive(panel.querySelector('[data-indicator="aim-tracking"]'), aimTracking);
  setActive(panel.querySelector('[data-indicator="engaged"]'), engaged);
  setActive(panel.querySelector('[data-indicator="aim"]'), aimTracking);

  for (const [field, level] of [["trigger", trigger], ["squeeze", squeeze]]) {
    const control = panel.querySelector(`[data-control="${field}"]`);
    control.style.setProperty("--level", level.toFixed(3));
    setActive(control, level >= 0.5);
    control.setAttribute("aria-label", `${field} ${Math.round(level * 100)} percent`);
    const meter = panel.querySelector(`[data-meter="${field}"]`);
    meter.style.setProperty("--level", level.toFixed(3));
    meter.setAttribute("aria-label", `${field} ${Math.round(level * 100)} percent`);
  }

  for (const field of ["primary_button", "secondary_button", "menu_button"]) {
    const pressed = finiteNumber(action[key(field)]) >= 0.5;
    const elements = panel.querySelectorAll(`[data-control="${field}"], [data-detail-control="${field}"]`);
    for (const element of elements) {
      setActive(element, pressed);
      element.setAttribute("aria-label", `${field} ${pressed ? "pressed" : "released"}`);
    }
  }

  const stickPressed = finiteNumber(action[key("thumbstick_click")]) >= 0.5;
  const knob = panel.querySelector('[data-part="stick-knob"]');
  knob.style.transform = `translate(${(thumbstick[0] * 9).toFixed(2)}px, ${(-thumbstick[1] * 9).toFixed(2)}px)`;
  setActive(knob, stickPressed);
  const detailStick = panel.querySelector('[data-detail-control="thumbstick_click"]');
  setActive(detailStick, stickPressed);
  detailStick.setAttribute("aria-label", `thumbstick ${stickPressed ? "pressed" : "released"}`);
  panel.querySelector("[data-stick-direction]").textContent = `↔ ${thumbstick[0].toFixed(2)}  ↕ ${thumbstick[1].toFixed(2)}`;

  for (const field of ["translation", "rotation", "aim_translation", "aim_rotation"]) {
    const size = field.endsWith("rotation") ? 4 : 3;
    const value = vector(action, key(field), size, size === 4);
    panel.querySelector(`[data-value="${field}"]`).textContent = formatVector(value);
  }

  poseScene?.update(side, action);
}

function serviceColor(state) {
  if (state === "streaming") return "var(--ok)";
  if (state === "fault" || state === "stopped") return "var(--bad)";
  return "var(--warn)";
}

function applyStatus(status) {
  const values = {
    rate: `${finiteNumber(status.publish_rate_hz).toFixed(1)} Hz`,
    age: status.last_frame_age_ms == null ? "—" : `${finiteNumber(status.last_frame_age_ms).toFixed(0)} ms`,
    sequence: status.sequence ?? "—",
    sampled: status.sampled_frames,
    published: status.published_frames,
    dropped: status.dropped_frames,
    uptime: status.uptime,
  };
  for (const [id, value] of Object.entries(values)) {
    document.getElementById(id).textContent = value;
  }
  const badge = document.getElementById("badge");
  badge.textContent = status.state;
  badge.style.color = serviceColor(status.state);
  badge.style.borderColor = serviceColor(status.state);
  const error = document.getElementById("error");
  error.style.display = status.last_error ? "block" : "none";
  error.textContent = status.last_error || "";
  for (const side of SIDES) updateController(side, status.action);
  document.getElementById("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

function showStreamDisconnected(message) {
  const badge = document.getElementById("badge");
  badge.textContent = "disconnected";
  badge.style.color = "var(--bad)";
  badge.style.borderColor = "var(--bad)";
  document.getElementById("updated").textContent = message;
}

function streamUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/stream`;
}

function connectStatusStream() {
  const socket = new WebSocket(streamUrl());
  statusSocket = socket;

  socket.addEventListener("message", event => {
    if (statusSocket !== socket) return;
    try {
      pendingStatus = JSON.parse(event.data);
    } catch (error) {
      showStreamDisconnected(`Invalid monitor update: ${error.message}`);
      socket.close(1003, "Invalid status payload");
    }
  });
  socket.addEventListener("error", () => socket.close());
  socket.addEventListener("close", () => {
    if (statusSocket !== socket) return;
    statusSocket = null;
    showStreamDisconnected("Live stream disconnected; reconnecting…");
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectStatusStream, STREAM_RECONNECT_DELAY_MS);
  });
}

function renderDashboardStatus(timestamp) {
  requestAnimationFrame(renderDashboardStatus);
  if (pendingStatus == null) return;
  if (lastDashboardFrameAt != null && timestamp - lastDashboardFrameAt < FRAME_INTERVAL_MS) return;
  const status = pendingStatus;
  pendingStatus = null;
  lastDashboardFrameAt = timestamp;
  applyStatus(status);
}

class RelativePoseScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 20);
    this.camera.position.set(0, 0.62, 1.45);
    this.camera.lookAt(0, 0.08, 0);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.08, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 3;
    this.controls.minPolarAngle = 0.15;
    this.controls.maxPolarAngle = Math.PI - 0.15;
    this.controls.maxTargetRadius = 0.8;
    this.controls.update();
    this.controls.saveState();
    this.renderer.domElement.addEventListener("dblclick", () => this.controls.reset());

    this.scene.add(new THREE.HemisphereLight(0xd8edff, 0x18202a, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(0.8, 1.4, 1.2);
    this.scene.add(keyLight);

    const grid = new THREE.GridHelper(1.7, 17, 0x51657a, 0x263545);
    grid.position.y = -0.22;
    this.scene.add(grid);

    this.basis = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    this.inverseBasis = this.basis.clone().invert();
    this.controllers = {
      left: this.createController("left", -0.34),
      right: this.createController("right", 0.34),
    };
    this.renderRateElement = document.getElementById("render-rate");
    this.renderFrames = 0;
    this.renderWindowStartedAt = performance.now();
    this.lastRenderedAt = null;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.animate(this.renderWindowStartedAt);
  }

  createController(side, originX) {
    const color = COLORS[side];
    const origin = new THREE.Group();
    origin.position.set(originX, 0.02, 0);
    this.scene.add(origin);
    origin.add(new THREE.AxesHelper(0.13));
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 12, 8),
      new THREE.MeshBasicMaterial({ color }),
    );
    origin.add(marker);

    const pose = new THREE.Group();
    origin.add(pose);
    const materials = [];
    const material = (roughness = 0.6) => {
      const value = new THREE.MeshStandardMaterial({ color: COLORS.inactive, roughness, metalness: 0.08 });
      materials.push(value);
      return value;
    };

    const handle = new THREE.Mesh(new THREE.CapsuleGeometry(0.052, 0.14, 6, 14), material(0.72));
    handle.position.y = -0.075;
    handle.rotation.z = side === "left" ? -0.16 : 0.16;
    pose.add(handle);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.077, 24, 14), material(0.46));
    head.scale.set(1.0, 0.48, 1.12);
    head.position.y = 0.085;
    pose.add(head);
    const face = new THREE.Mesh(new THREE.CylinderGeometry(0.061, 0.069, 0.019, 28), material(0.38));
    face.position.y = 0.126;
    pose.add(face);
    for (const [x, z] of [[-0.022, -0.012], [0.024, 0.014]]) {
      const button = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.007, 16), material(0.32));
      button.position.set(side === "left" ? x : -x, 0.14, z);
      pose.add(button);
    }

    const aimPose = new THREE.Group();
    origin.add(aimPose);
    const aimGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -0.36),
    ]);
    const aimLine = new THREE.Line(aimGeometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
    aimPose.add(aimLine);
    aimPose.visible = false;

    const trajectory = new EngagementTrajectory();
    const traceGeometry = new THREE.BufferGeometry();
    const tracePositions = new THREE.BufferAttribute(new Float32Array(trajectory.maximumPoints * 3), 3);
    traceGeometry.setAttribute("position", tracePositions);
    traceGeometry.setDrawRange(0, 0);
    const traceMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 });
    const trace = new THREE.Line(traceGeometry, traceMaterial);
    const tracePointMaterial = new THREE.PointsMaterial({
      color,
      size: 0.012,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
    });
    const tracePoints = new THREE.Points(traceGeometry, tracePointMaterial);
    const traceEndpoint = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 16, 10),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.92 }),
    );
    traceEndpoint.visible = false;
    origin.add(trace, tracePoints, traceEndpoint);

    return {
      color,
      pose,
      aimPose,
      materials,
      trajectory,
      trace,
      traceGeometry,
      tracePositions,
      traceMaterial,
      tracePointMaterial,
      traceEndpoint,
      targetPosition: new THREE.Vector3(),
      targetQuaternion: new THREE.Quaternion(),
    };
  }

  updateTraceGeometry(controller) {
    const points = controller.trajectory.points;
    points.forEach((point, index) => controller.tracePositions.setXYZ(index, point[0], point[1], point[2]));
    controller.tracePositions.needsUpdate = true;
    controller.traceGeometry.setDrawRange(0, points.length);
    controller.traceGeometry.computeBoundingSphere();
  }

  mapPosition(values, destination) {
    destination.set(values[0], values[2], -values[1]);
  }

  mapQuaternion(values, destination) {
    const input = new THREE.Quaternion(values[0], values[1], values[2], values[3]).normalize();
    destination.copy(this.basis).multiply(input).multiply(this.inverseBasis).normalize();
  }

  update(side, action) {
    const controller = this.controllers[side];
    const key = field => `${side}.${field}`;
    const tracking = Boolean(action[key("is_tracking")]);
    const engaged = tracking && Boolean(action[key("is_engaged")]);
    const aimTracking = engaged && Boolean(action[key("is_aim_tracking")]);

    for (const material of controller.materials) {
      material.color.setHex(tracking ? controller.color : COLORS.inactive);
      material.emissive.setHex(engaged ? controller.color : 0x000000);
      material.emissiveIntensity = engaged ? 0.16 : 0;
    }

    if (engaged) {
      this.mapPosition(vector(action, key("translation"), 3), controller.targetPosition);
      this.mapQuaternion(vector(action, key("rotation"), 4, true), controller.targetQuaternion);
    } else {
      controller.targetPosition.set(0, 0, 0);
      controller.targetQuaternion.identity();
    }

    const traceChanged = controller.trajectory.update(engaged, controller.targetPosition.toArray());
    if (traceChanged) this.updateTraceGeometry(controller);
    controller.traceMaterial.opacity = engaged ? 0.88 : 0.3;
    controller.tracePointMaterial.opacity = engaged ? 0.82 : 0.24;
    controller.traceEndpoint.visible = engaged;
    controller.traceEndpoint.position.copy(controller.targetPosition);

    controller.aimPose.visible = aimTracking;
    if (aimTracking) {
      this.mapPosition(vector(action, key("aim_translation"), 3), controller.aimPose.position);
      this.mapQuaternion(vector(action, key("aim_rotation"), 4, true), controller.aimPose.quaternion);
    }
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  animate(timestamp) {
    requestAnimationFrame(nextTimestamp => this.animate(nextTimestamp));
    if (this.lastRenderedAt != null && timestamp - this.lastRenderedAt < FRAME_INTERVAL_MS) return;
    this.lastRenderedAt = timestamp;
    for (const controller of Object.values(this.controllers)) {
      if (reducedMotion) {
        controller.pose.position.copy(controller.targetPosition);
        controller.pose.quaternion.copy(controller.targetQuaternion);
      } else {
        controller.pose.position.lerp(controller.targetPosition, 0.22);
        controller.pose.quaternion.slerp(controller.targetQuaternion, 0.22);
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.renderFrames += 1;
    const elapsedMs = timestamp - this.renderWindowStartedAt;
    if (elapsedMs >= 500) {
      const rateHz = this.renderFrames * 1_000 / elapsedMs;
      this.renderRateElement.textContent = `${rateHz.toFixed(1)} Hz`;
      this.renderFrames = 0;
      this.renderWindowStartedAt = timestamp;
    }
  }
}

let poseScene = null;
try {
  poseScene = new RelativePoseScene(document.getElementById("pose-scene"));
} catch (error) {
  document.getElementById("three-fallback").style.display = "grid";
  console.error("Three.js pose view failed to initialize", error);
}

requestAnimationFrame(renderDashboardStatus);
connectStatusStream();
