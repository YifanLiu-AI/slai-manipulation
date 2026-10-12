const cameraFigures = new Map();

function cameraLabel(camera) {
  const datasetKey = String(camera.dataset_key || "");
  if (datasetKey.includes("d435_primary")) return "D435 Primary RGB";
  if (datasetKey.includes("d435_secondary")) return "D435 Secondary RGB";
  if (datasetKey.includes("d405")) return "D405 RGB";
  return camera.label || camera.role || camera.key;
}

function ensureCameras(cameras) {
  const grid = document.getElementById("camera-grid");
  const template = document.getElementById("camera-template");
  for (const camera of cameras) {
    if (cameraFigures.has(camera.key)) continue;
    const figure = template.content.firstElementChild.cloneNode(true);
    const image = figure.querySelector("img");
    figure.dataset.cameraFigure = camera.key;
    figure.querySelector(".camera-label").textContent = cameraLabel(camera);
    figure.querySelector(".camera-age").dataset.cameraAge = camera.key;
    image.dataset.camera = camera.key;
    image.alt = `${cameraLabel(camera)} 实时画面`;
    grid.appendChild(figure);
    cameraFigures.set(camera.key, figure);
    refreshFrame(image);
  }
  for (const [key, figure] of cameraFigures) {
    if (!cameras.some((camera) => camera.key === key)) {
      figure.remove();
      cameraFigures.delete(key);
    }
  }
}

function refreshFrame(image) {
  const refresh = () => window.setTimeout(() => {
    image.src = `/frame/${encodeURIComponent(image.dataset.camera)}.jpg?t=${Date.now()}`;
  }, 50);
  image.addEventListener("load", refresh);
  image.addEventListener("error", () => window.setTimeout(refresh, 500));
  refresh();
}

async function refreshCameraStatus() {
  try {
    const response = await fetch("/api/cameras", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const cameras = await response.json();
    const values = Object.values(cameras);
    ensureCameras(values);
    for (const camera of values) {
      const label = document.querySelector(`[data-camera-age="${CSS.escape(camera.key)}"]`);
      if (!label) continue;
      const online = Boolean(camera.connected && camera.age_ms != null);
      label.textContent = online ? `${Number(camera.age_ms).toFixed(0)} ms` : "OFFLINE";
      label.classList.toggle("online", online);
      label.classList.toggle("slow", online && camera.age_ms >= 100);
    }
  } catch (_) {
    for (const label of document.querySelectorAll("[data-camera-age]")) {
      label.textContent = "OFFLINE";
      label.classList.remove("online", "slow");
    }
  } finally {
    window.setTimeout(refreshCameraStatus, 100);
  }
}

function eventTime(value) {
  const text = String(value || "--:--:--");
  return text.includes("T") ? text.split("T")[1].slice(0, 8) : text.slice(-8);
}

async function refreshRecordingStatus() {
  try {
    const response = await fetch("/api/recording", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const badge = document.querySelector(".record-badge");
    badge.dataset.state = payload.state.code;
    document.querySelector(".record-label").textContent = payload.state.label;
    document.querySelector(".record-detail").textContent = payload.state.detail || "";
    const temperature = payload.temperature || {};
    const temperatureBox = document.querySelector(".temperature-status");
    const temperatureValue = document.querySelector(".temperature-value");
    const temperatureDetail = document.querySelector(".temperature-detail");
    const temperatureLevel = String(temperature.level || "unknown");
    temperatureBox.dataset.temperatureLevel = temperatureLevel;
    temperatureValue.textContent = temperature.max_c == null ? "--" : `${Number(temperature.max_c).toFixed(1)} °C`;
    temperatureDetail.textContent = temperatureLevel === "critical"
      ? `临界温度，限制 ${Number(temperature.limit_c || 80).toFixed(0)} °C`
      : temperatureLevel === "warning"
        ? `温度警告，临界 ${Number(temperature.critical_c || 75).toFixed(0)} °C`
        : temperatureLevel === "normal"
          ? `正常，警告线 ${Number(temperature.warning_c || 70).toFixed(0)} °C`
          : "监测尚未就绪";
    const list = document.querySelector(".event-list");
    const shouldFollow = list.scrollHeight - list.scrollTop - list.clientHeight < 20;
    list.replaceChildren(...payload.events.map((event) => {
      const row = document.createElement("div");
      row.className = "event";
      row.dataset.level = event.level;
      const timestamp = document.createElement("span");
      timestamp.className = "event-time";
      timestamp.textContent = eventTime(event.time);
      const mark = document.createElement("span");
      mark.className = "event-mark";
      mark.textContent = "●";
      const message = document.createElement("span");
      message.textContent = event.message;
      row.append(timestamp, mark, message);
      return row;
    }));
    if (shouldFollow) list.scrollTop = list.scrollHeight;
  } catch (_) {
    const badge = document.querySelector(".record-badge");
    badge.dataset.state = "blocked";
    document.querySelector(".record-label").textContent = "状态断开";
  } finally {
    window.setTimeout(refreshRecordingStatus, 100);
  }
}

async function refreshDevices() {
  try {
    const response = await fetch("/api/devices", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    document.querySelector(".capture-mode").textContent = String(payload.mode || "combined").toUpperCase();
    for (const name of ["ur5", "wuji"]) {
      const item = document.querySelector(`[data-device="${name}"]`);
      const state = payload.devices?.[name] || {state: "error"};
      const label = state.state === "active" ? "ACTIVE" : state.state === "inactive" ? "NOT ENABLED" : state.state === "starting" ? "STARTING" : "ERROR";
      item.textContent = `${name === "ur5" ? "UR5" : "Wuji"} ${label}`;
      item.classList.toggle("starting", state.state === "starting");
      item.classList.toggle("inactive", state.state === "inactive");
      item.classList.toggle("error", state.state === "error");
    }
  } catch (_) {
    for (const item of document.querySelectorAll("[data-device]")) item.classList.add("error");
  } finally {
    window.setTimeout(refreshDevices, 250);
  }
}

async function refreshMouse() {
  try {
    const response = await fetch("/api/spacemouse", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    document.querySelector(".status").classList.toggle("online", state.connected);
    document.querySelector(".age").textContent = state.age_ms == null ? "--" : `${Number(state.age_ms).toFixed(0)} ms`;
    const motion = Array.from({length: 6}, (_, index) =>
      Math.max(-1, Math.min(1, Number(state.motion?.[index] || 0))));
    const puck = document.querySelector(".puck");
    puck.style.transform = `translate(${motion[0] * 16}px, ${-motion[1] * 14}px)
      scale(${1 + motion[2] * 0.13}) rotateX(${-motion[3] * 18}deg)
      rotateY(${motion[4] * 18}deg) rotateZ(${motion[5] * 24}deg)`;
    puck.classList.toggle("active", motion.some((value) => Math.abs(value) > 0.01));
    document.querySelector(".motion-values").textContent =
      `T ${motion.slice(0, 3).map((value) => value.toFixed(2)).join(" ")} · ` +
      `R ${motion.slice(3).map((value) => value.toFixed(2)).join(" ")}`;
    for (const button of document.querySelectorAll("[data-button]")) {
      const aliases = {rear: "r", front: "f"};
      button.classList.toggle("pressed", Boolean(
        state.buttons?.[button.dataset.button] || state.buttons?.[aliases[button.dataset.button]],
      ));
    }
  } catch (_) {
    document.querySelector(".status").classList.remove("online");
  } finally {
    window.setTimeout(refreshMouse, 50);
  }
}

refreshCameraStatus();
refreshRecordingStatus();
refreshDevices();
refreshMouse();
