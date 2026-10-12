const DASHBOARD_LAN_HOST = "192.168.1.102";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

if (LOOPBACK_HOSTS.has(window.location.hostname)) {
  const host = window.location.port
    ? `${DASHBOARD_LAN_HOST}:${window.location.port}`
    : DASHBOARD_LAN_HOST;
  window.location.replace(
    `${window.location.protocol}//${host}${window.location.pathname}${window.location.search}${window.location.hash}`,
  );
}

const CAMERA_ORDER = ["secondary", "primary", "wrist"];
const CAMERA_LABELS = {
  secondary: "CAM 01 - LEFT",
  primary: "CAM 00 - CENTER MAIN",
  wrist: "CAM 02 - RIGHT",
};
const cameraFigures = new Map();
const cameraPreviews = new Map();

function sortedCameras(cameras) {
  return [...cameras].sort((left, right) => {
    const leftIndex = CAMERA_ORDER.indexOf(left.key);
    const rightIndex = CAMERA_ORDER.indexOf(right.key);
    return (leftIndex < 0 ? 99 : leftIndex) - (rightIndex < 0 ? 99 : rightIndex);
  });
}

function ensureCameras(cameras) {
  const grid = document.getElementById("camera-grid");
  const template = document.getElementById("camera-template");
  for (const camera of sortedCameras(cameras)) {
    let figure = cameraFigures.get(camera.key);
    if (!figure) {
      figure = template.content.firstElementChild.cloneNode(true);
      figure.dataset.camera = camera.key;
      const image = figure.querySelector("img");
      image.dataset.camera = camera.key;
      image.alt = `${CAMERA_LABELS[camera.key] || camera.label || camera.key} live RGB feed`;
      grid.appendChild(figure);
      cameraFigures.set(camera.key, figure);
      cameraPreviews.set(camera.key, createCameraPreview(figure));
    }
    figure.querySelector(".camera-label").textContent =
      CAMERA_LABELS[camera.key] || camera.label || camera.role || camera.key;
    const resolution = Array.isArray(camera.resolution) ? camera.resolution.join("x") : "--";
    const fps = Number(camera.fps || 0);
    figure.querySelector(".camera-format").textContent = `${resolution} / ${fps.toFixed(0)}fps`;
  }
  for (const [key, figure] of cameraFigures) {
    if (!cameras.some((camera) => camera.key === key)) {
      cameraPreviews.get(key)?.stop();
      cameraPreviews.delete(key);
      figure.remove();
      cameraFigures.delete(key);
    }
  }
  const orderedFigures = sortedCameras(cameras)
    .map((camera) => cameraFigures.get(camera.key))
    .filter(Boolean);
  const currentFigures = [...grid.children];
  if (orderedFigures.some((figure, index) => currentFigures[index] !== figure)) {
    grid.replaceChildren(...orderedFigures);
  }
}

function whepUrl(role) {
  const encoded = encodeURIComponent(role);
  if (window.location.hostname === "record.leai.me") {
    return `/${encoded}/whep`;
  }
  const host = window.location.hostname.includes(":")
    ? `[${window.location.hostname}]`
    : window.location.hostname;
  return `${window.location.protocol}//${host}:8889/${encoded}/whep`;
}

function isLanPage() {
  const host = window.location.hostname;
  if (host === "localhost" || host === "::1" || host.startsWith("127.")) return true;
  if (host.startsWith("10.") || host.startsWith("192.168.")) return true;
  const match = host.match(/^172\.(\d+)\./);
  return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
}

function createCameraPreview(figure) {
  const image = figure.querySelector("img");
  const video = figure.querySelector("video");
  const role = image.dataset.camera;
  let peer = null;
  let request = null;
  let retryTimer = null;
  let snapshotTimer = null;
  let snapshotProbe = null;
  let retryDelay = 250;
  let stopped = false;
  let generation = 0;

  const setFrame = (visible, webrtc) => {
    figure.dataset.frame = String(visible);
    figure.dataset.webrtc = String(webrtc);
  };
  const showFallback = () => {
    setFrame(Boolean(image.complete && image.naturalWidth > 0), false);
  };
  const loadSnapshot = () => {
    if (stopped || document.hidden || snapshotProbe != null) return;
    const probe = new Image();
    snapshotProbe = probe;
    probe.addEventListener("load", () => {
      if (snapshotProbe !== probe) return;
      snapshotProbe = null;
      image.src = probe.src;
    });
    probe.addEventListener("error", () => {
      if (snapshotProbe !== probe) return;
      snapshotProbe = null;
      showFallback();
      scheduleSnapshot();
    });
    probe.src = `/api/cameras/${encodeURIComponent(role)}/frame.jpg?t=${Date.now()}`;
  };
  const scheduleRetry = () => {
    if (stopped || document.hidden || retryTimer != null) return;
    const delay = retryDelay;
    retryDelay = Math.min(retryDelay * 2, 2000);
    retryTimer = window.setTimeout(() => {
      retryTimer = null;
      connectWebRTC();
    }, delay);
  };
  const scheduleSnapshot = () => {
    if (stopped || document.hidden || snapshotTimer != null) return;
    snapshotTimer = window.setTimeout(() => {
      snapshotTimer = null;
      loadSnapshot();
    }, 250);
  };
  const closePeer = () => {
    generation += 1;
    request?.abort();
    request = null;
    if (peer) {
      peer.onconnectionstatechange = null;
      peer.close();
      peer = null;
    }
    video.srcObject = null;
  };
  const connectWebRTC = async () => {
    if (stopped || document.hidden) return;
    closePeer();
    const currentGeneration = generation;
    if (!("RTCPeerConnection" in window)) {
      loadSnapshot();
      return;
    }
    try {
      const connection = new RTCPeerConnection({
        iceServers: isLanPage() ? [] : [{urls: "stun:stun.l.google.com:19302"}],
      });
      peer = connection;
      const showDecodedFrame = () => {
        const showVideo = () => {
          retryDelay = 250;
          setFrame(true, true);
        };
        if ("requestVideoFrameCallback" in video) {
          video.requestVideoFrameCallback(showVideo);
        } else {
          showVideo();
        }
      };
      const startPlayback = () => {
        video.play().then(showDecodedFrame).catch(() => {});
      };
      video.onloadedmetadata = startPlayback;
      video.onplaying = showDecodedFrame;
      connection.addTransceiver("video", {direction: "recvonly"});
      connection.ontrack = (event) => {
        if (event.streams[0]) video.srcObject = event.streams[0];
      };
      connection.onconnectionstatechange = () => {
        if (peer === connection && ["failed", "disconnected", "closed"].includes(connection.connectionState)) {
          showFallback();
          loadSnapshot();
          scheduleRetry();
        }
      };
      const offer = await connection.createOffer();
      await connection.setLocalDescription(offer);
      await new Promise((resolve) => {
        if (connection.iceGatheringState === "complete") return resolve();
        const done = () => {
          if (connection.iceGatheringState === "complete") {
            connection.removeEventListener("icegatheringstatechange", done);
            resolve();
          }
        };
        connection.addEventListener("icegatheringstatechange", done);
        window.setTimeout(resolve, isLanPage() ? 350 : 1500);
      });
      if (currentGeneration !== generation) return;
      const controller = new AbortController();
      request = controller;
      const response = await fetch(whepUrl(role), {
        method: "POST",
        headers: {"Content-Type": "application/sdp"},
        body: connection.localDescription.sdp,
        signal: controller.signal,
      });
      if (request === controller) request = null;
      if (currentGeneration !== generation) return;
      if (!response.ok) throw new Error(`WHEP HTTP ${response.status}`);
      await connection.setRemoteDescription({type: "answer", sdp: await response.text()});
      if (video.readyState >= 1) startPlayback();
    } catch (_) {
      if (stopped || document.hidden || currentGeneration !== generation) return;
      closePeer();
      showFallback();
      loadSnapshot();
      scheduleRetry();
    }
  };
  image.addEventListener("load", () => {
    if (snapshotTimer != null) {
      window.clearTimeout(snapshotTimer);
      snapshotTimer = null;
    }
    if (figure.dataset.webrtc !== "true") setFrame(true, false);
  });
  video.addEventListener("error", () => {
    showFallback();
    loadSnapshot();
    scheduleRetry();
  });
  loadSnapshot();
  connectWebRTC();

  return {
    start() {
      if (!stopped) return;
      stopped = false;
      retryDelay = 250;
      loadSnapshot();
      connectWebRTC();
    },
    stop() {
      stopped = true;
      if (retryTimer != null) window.clearTimeout(retryTimer);
      if (snapshotTimer != null) window.clearTimeout(snapshotTimer);
      retryTimer = null;
      snapshotTimer = null;
      snapshotProbe = null;
      closePeer();
      setFrame(false, false);
    },
  };
}

function renderCameraStatus(cameras) {
  ensureCameras(cameras);
  for (const camera of cameras) {
    const figure = cameraFigures.get(camera.key);
    if (!figure) continue;
    const online = Boolean(camera.connected && camera.valid && camera.age_ms != null);
    const slow = online && Number(camera.age_ms) >= 100;
    figure.dataset.online = String(online);
    figure.dataset.slow = String(slow);
    figure.querySelector(".camera-latency").textContent = online
      ? `Lat: ${Number(camera.age_ms).toFixed(0)}ms`
      : "Lat: OFFLINE";
    const resolution = Array.isArray(camera.resolution) ? camera.resolution.join("x") : "--";
    figure.querySelector(".camera-format").textContent =
      `${resolution} / ${Number(camera.fps || 0).toFixed(0)}fps`;
  }
  const onlineCount = cameras.filter((camera) => camera.connected && camera.valid).length;
  document.getElementById("system-label").textContent =
    `Sys: ${onlineCount === cameras.length && cameras.length ? "Nominal" : `${onlineCount}/${cameras.length} Cameras`}`;
}

function eventTime(value) {
  const text = String(value || "--:--:--");
  if (text.includes("T")) return text.split("T")[1].slice(0, 12);
  return text.slice(-12);
}

function renderEvents(events) {
  const list = document.querySelector(".event-list");
  const shouldFollow = list.scrollHeight - list.scrollTop - list.clientHeight < 24;
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "event empty";
    empty.textContent = "Waiting for collection events.";
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(...events.map((event) => {
    const row = document.createElement("div");
    row.className = "event";
    row.dataset.level = event.level || "info";
    const timestamp = document.createElement("span");
    timestamp.className = "event-time";
    timestamp.textContent = eventTime(event.time);
    const level = document.createElement("span");
    level.className = "event-level";
    level.textContent = event.code || event.level || "info";
    const message = document.createElement("span");
    message.className = "event-message";
    message.textContent = event.message || "";
    row.append(timestamp, level, message);
    return row;
  }));
  if (shouldFollow) list.scrollTop = list.scrollHeight;
}

function metricValue(value) {
  return value == null ? "—" : String(Number(value));
}

function historyTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || "—");
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function renderCollectionHistory(history) {
  const summary = history.summary || {};
  const taskId = String(summary.task_id || summary.task || "DATASET");
  document.getElementById("collection-task").textContent = taskId.toUpperCase();
  document.getElementById("collection-task").title = String(summary.task || taskId);
  document.getElementById("collection-dataset").textContent = summary.dataset || "data/lerobot";
  document.getElementById("metric-saved").textContent = metricValue(summary.saved);
  document.getElementById("metric-attempts").textContent = metricValue(summary.attempts);
  document.getElementById("metric-discarded").textContent = metricValue(summary.discarded);
  document.getElementById("metric-history-saved").textContent = metricValue(summary.history_saved);

  const list = document.querySelector(".history-list");
  const sessions = Array.isArray(history.sessions) ? history.sessions : [];
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "history-row empty";
    empty.textContent = "data/lerobot 中还没有可显示的数据集";
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(...sessions.map((session) => {
    const row = document.createElement("div");
    row.className = "history-row";
    row.title = session.dataset_path || session.dataset || "";
    const values = [
      historyTime(session.time),
      session.task || "—",
      metricValue(session.attempts),
      metricValue(session.saved),
      metricValue(session.discarded),
      session.dataset || "—",
    ];
    for (const value of values) {
      const cell = document.createElement("span");
      cell.textContent = value;
      row.appendChild(cell);
    }
    return row;
  }));
}

let historyRequest = null;
let historyRetryTimer = null;

async function refreshCollectionHistory() {
  if (historyRequest) return historyRequest;
  if (historyRetryTimer != null) {
    window.clearTimeout(historyRetryTimer);
    historyRetryTimer = null;
  }
  historyRequest = fetch("/api/collection-history", {cache: "no-store"})
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(renderCollectionHistory)
    .catch(() => {
      document.getElementById("collection-dataset").textContent = "data/lerobot 读取失败";
      const list = document.querySelector(".history-list");
      const error = document.createElement("div");
      error.className = "history-row empty";
      error.textContent = "历史接口暂时不可用，请刷新页面";
      list.replaceChildren(error);
      historyRetryTimer = window.setTimeout(refreshCollectionHistory, 2000);
    })
    .finally(() => { historyRequest = null; });
  return historyRequest;
}

let lastEventSignature = "";
let lastHistorySignature = "";

function renderRecordingStatus(status) {
  const indicator = document.querySelector(".record-indicator");
  indicator.dataset.state = status.phase || "starting";
  const episode = status.episode || {};
  const detail = status.recording
    ? `${episode.valid_frames || 0} 帧 · ${Number(episode.elapsed_s || 0).toFixed(1)} 秒`
    : (status.can_record ? "等待 MENU" : "");
  document.getElementById("record-label").textContent = detail
    ? `${status.phase_label || "准备中"} · ${detail}`
    : (status.phase_label || "准备中");
  renderTemperature(status.temperature || {});
  const events = Array.isArray(status.events) ? [...status.events].reverse() : [];
  renderEvents(events);
}

function renderTemperature(temperature) {
  const box = document.querySelector(".temperature-status");
  const maximum = Number(temperature.max_c);
  const available = Boolean(temperature.available && Number.isFinite(maximum));
  const level = available ? String(temperature.level || "normal") : "unknown";
  box.dataset.available = String(available);
  box.dataset.level = level;
  box.querySelector("strong").textContent = available ? `${maximum.toFixed(1)}°C` : "--";

  const values = Array.isArray(temperature.values)
    ? temperature.values.map(Number).filter(Number.isFinite)
    : [];
  if (!available) {
    box.title = "WujiHand temperature is waiting for the collection process";
    return;
  }
  const minimum = values.length ? Math.min(...values) : maximum;
  const warning = Number(temperature.warning_c);
  const threshold = Number.isFinite(warning) ? ` · warning ${warning.toFixed(0)}°C` : "";
  box.title = `${values.length || 1} joints · ${minimum.toFixed(1)}–${maximum.toFixed(1)}°C${threshold}`;
}

function renderDevices(devices) {
  const names = {ur5: "UR5", wuji: "WujiHand", wrist: "Wrist"};
  const labels = {active: "ON", inactive: "OFF", starting: "STARTING", error: "ERROR"};
  document.getElementById("device-summary").textContent = ["ur5", "wuji", "wrist"]
    .map((name) => `${names[name]} ${labels[devices?.[name]?.state] || "ERROR"}`)
    .join(" · ");
}

async function loadSpaceMouseMap() {
  const host = document.getElementById("spacemouse-container");
  try {
    const response = await fetch("/spacemouse-input-map.svg?v=20260823-5");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    host.innerHTML = await response.text();
    const svg = host.querySelector("svg");
    if (svg) {
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.setAttribute("focusable", "false");
      requestAnimationFrame(positionSpaceMousePuck);
    }
  } catch (_) {
    host.textContent = "Input map unavailable";
  }
}

function positionSpaceMousePuck() {
  const wrap = document.querySelector(".spacemouse-wrap");
  const host = document.getElementById("spacemouse-container");
  const puck = document.getElementById("spacemouse-puck");
  const svg = host?.querySelector("svg");
  if (!wrap || !puck || !svg) return;
  const wrapRect = wrap.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();
  // center-ring is cx=252, cy=268 in a 508x502 viewBox.
  puck.style.left = `${svgRect.left - wrapRect.left + svgRect.width * (252 / 508)}px`;
  puck.style.top = `${svgRect.top - wrapRect.top + svgRect.height * (268 / 502)}px`;
}

function setSpaceMouseButtons(buttons) {
  const aliases = {
    rear: ["r"],
    front: ["f"],
    rotation_lock: ["lock"],
    roll_cw: ["roll"],
  };
  for (const button of document.querySelectorAll("#spacemouse-container [data-button]")) {
    const name = button.dataset.button;
    const names = [name, ...(aliases[name] || [])];
    button.dataset.pressed = String(names.some((candidate) => Boolean(buttons?.[candidate])));
  }
}

function setMotion(motion) {
  const values = Array.from({length: 6}, (_, index) =>
    Math.max(-1, Math.min(1, Number(motion?.[index] || 0))));
  const puck = document.getElementById("spacemouse-puck");
  // Keep input values unchanged, but make small real movements visible
  // inside the center ring.
  const travel = 44;
  puck.style.transform = `translate(calc(-50% + ${values[1] * travel}px), calc(-50% + ${values[0] * travel}px)) ` +
    `scale(${1 + values[2] * 0.1}) rotateX(${-values[3] * 18}deg) ` +
    `rotateY(${values[4] * 18}deg) rotateZ(${values[5] * 22}deg)`;
  puck.classList.toggle("active", values.some((value) => Math.abs(value) > 0.01));
  for (const [index, value] of values.entries()) {
    document.querySelector(`[data-axis="${index}"] output`).value = value.toFixed(2);
  }
}

function renderMouse(state) {
  const stateBox = document.querySelector(".spacemouse-state");
  stateBox.dataset.connected = String(Boolean(state.connected && state.valid));
  stateBox.title = state.error || "Read-only status from the SLAI collection process";
  document.getElementById("spacemouse-connection-state").textContent = state.connected
    ? (state.active ? "SpaceMouse Pro · ACTIVE" : "SpaceMouse Pro")
    : "SpaceMouse Pro · WAITING";
  document.getElementById("spacemouse-age").textContent = state.age_ms == null
    ? "--"
    : `${Number(state.age_ms).toFixed(0)}ms`;
  setMotion(state.motion);
  setSpaceMouseButtons(state.buttons);
}

function renderStatus(status) {
  const collectionActive = Boolean(status.dataset_path);
  const eventPanel = document.querySelector(".event-panel");
  eventPanel.dataset.collectionActive = String(collectionActive);
  document.getElementById("event-panel-title").textContent = collectionActive
    ? "System Events Log"
    : "Dataset Overview";
  renderCameraStatus(Array.isArray(status.cameras) ? status.cameras : []);
  const eventSignature = JSON.stringify([
    status.phase,
    status.recording,
    status.episode,
    status.temperature,
    status.events,
  ]);
  if (eventSignature !== lastEventSignature) {
    lastEventSignature = eventSignature;
    renderRecordingStatus(status);
  }
  const historySignature = collectionActive ? "active" : "idle";
  if (!collectionActive && historySignature !== lastHistorySignature) {
    lastHistorySignature = historySignature;
    refreshCollectionHistory();
  }
  renderDevices(status.devices || {});
}

window.addEventListener("resize", positionSpaceMousePuck, {passive: true});
document.addEventListener("visibilitychange", () => {
  for (const preview of cameraPreviews.values()) {
    if (document.hidden) preview.stop();
    else preview.start();
  }
});
window.addEventListener("pagehide", () => {
  for (const preview of cameraPreviews.values()) preview.stop();
});

let pollingStarted = false;

function startStatusPolling() {
  if (pollingStarted) return;
  pollingStarted = true;
  const poll = async () => {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      renderStatus(await response.json());
    } catch (_) {
      document.getElementById("system-label").textContent = "Sys: Offline";
    } finally {
      window.setTimeout(poll, 250);
    }
  };
  poll();
}

function connectStatusStream() {
  if (!("EventSource" in window)) {
    startStatusPolling();
    return;
  }
  const stream = new EventSource("/api/events");
  let received = false;
  const fallbackTimer = window.setTimeout(() => {
    if (!received) {
      stream.close();
      startStatusPolling();
    }
  }, 4000);
  stream.onmessage = (event) => {
    received = true;
    renderStatus(JSON.parse(event.data));
  };
  stream.onerror = () => {
    if (received) document.getElementById("system-label").textContent = "Sys: Reconnecting";
  };
}

function connectSpaceMouseStream() {
  if (!("EventSource" in window)) return;
  const stream = new EventSource("/api/spacemouse/events");
  stream.onmessage = (event) => {
    renderMouse(JSON.parse(event.data));
  };
  stream.onerror = () => {};
}

loadSpaceMouseMap();
refreshCollectionHistory();
connectStatusStream();
connectSpaceMouseStream();
window.setInterval(() => {
  const panel = document.querySelector(".event-panel");
  if (panel?.dataset.collectionActive === "false") refreshCollectionHistory();
}, 5000);
