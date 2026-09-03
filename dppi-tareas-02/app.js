import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const GESTURE_MEMES = {
  default: ["memes/basketball.jpeg"],
  basketball: ["memes/basketball.jpeg"],
  timido: ["memes/timido.jpeg"],
  nose: ["memes/nose.jpeg"],
  mandarinas: ["memes/mandarinas.jpeg"],
  helly: ["memes/helly.jpeg"],
  strong: ["memes/strong.jpeg"],
  corazon: ["memes/corazon.jpeg"],
};

const STABLE_FRAMES_REQUIRED = 4;
const DEFAULT_FALLBACK_MS = 500;
const FACE_STALE_MS = 1200;
const SIDE_EYE_YAW_DEG = 15.0;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null;
let lastYawDebug = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function fingerExtended(lm, mcp, pip, tip) {
  const v1 = { x: lm[pip].x - lm[mcp].x, y: lm[pip].y - lm[mcp].y, z: (lm[pip].z || 0) - (lm[mcp].z || 0) };
  const v2 = { x: lm[tip].x - lm[pip].x, y: lm[tip].y - lm[pip].y, z: (lm[tip].z || 0) - (lm[pip].z || 0) };
  return angleDeg(v1, v2) < 45;
}

function yawFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6;

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    thumbTip: lm[4],
    wrist: lm[0],
    palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);

    let yawDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      yawDeg = yawFromTransformMatrix(faceResult.facialTransformationMatrixes[0].data);
    }

    lastFace = { mouthCenter, faceWidth, yawDeg, t: now };
    lastYawDebug = yawDeg;
  }
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;
  const yawDeg = faceIsFresh ? lastFace.yawDeg : 0;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    if (faceIsFresh && Math.abs(yawDeg) > SIDE_EYE_YAW_DEG) {
      return "basketball";
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  // 2 hands
  if (hands.length === 2) {
    const [h1, h2] = hands;
    const avgScale = (h1.handScale + h2.handScale) / 2;
    const handsDist = dist(h1.palmCenter, h2.palmCenter);

    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const handsCenter = {
        x: (h1.palmCenter.x + h2.palmCenter.x) / 2,
        y: (h1.palmCenter.y + h2.palmCenter.y) / 2,
        z: ((h1.palmCenter.z || 0) + (h2.palmCenter.z || 0)) / 2,
      };

      // helly.jpeg
      const oneFist = (h1.curledCount >= 2 && h2.curledCount <= 2) || (h2.curledCount >= 2 && h1.curledCount <= 2);
      const handsClasped = handsDist / avgScale < 2.0;
      const atChestNeck =
        handsCenter.y > mouthCenter.y + faceWidth * 0.05 &&
        handsCenter.y < mouthCenter.y + faceWidth * 2.2;
      if (handsClasped && atChestNeck && (oneFist || handsDist / avgScale < 1.4)) {
        return "helly";
      }

      // mandarinas.jpeg
      const bothFists = h1.curledCount >= 3 && h2.curledCount >= 3;
      const atNeckChin =
        h1.palmCenter.y >= mouthCenter.y - faceWidth * 0.2 &&
        h2.palmCenter.y >= mouthCenter.y - faceWidth * 0.2 &&
        h1.palmCenter.y <= mouthCenter.y + faceWidth * 1.9 &&
        h2.palmCenter.y <= mouthCenter.y + faceWidth * 1.9;
      const handsClose = handsDist / faceWidth < 2.2;
      if (bothFists && atNeckChin && handsClose) {
        return "mandarinas";
      }

      // nose.jpeg
      const bothOpen = h1.curledCount <= 1 && h2.curledCount <= 1;
      const belowShoulders =
        h1.palmCenter.y > mouthCenter.y - faceWidth * 0.3 &&
        h2.palmCenter.y > mouthCenter.y - faceWidth * 0.3;
      const handsWideSpread = handsDist / faceWidth > 1.2;
      if (bothOpen && belowShoulders && handsWideSpread) {
        return "nose";
      }
    }
  }

  // 1 hand
  const h = hands[0];

  if (faceIsFresh) {
    const { mouthCenter, faceWidth } = lastFace;
    // corazon.jpeg
    const onChest =
      h.palmCenter.y > mouthCenter.y + faceWidth * 0.3 &&
      Math.abs(h.palmCenter.x - mouthCenter.x) < faceWidth * 1.5;
    const thumbUp = h.thumbOut && h.thumbTip.y < h.wrist.y;
    if (onChest && thumbUp) {
      return "corazon";
    }
  }

  // strong.jpeg
  return "strong";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture] || GESTURE_MEMES["default"];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent =
    `gesture: ${currentGesture}\n` +
    `yaw: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)} deg  (profile thr +/-${SIDE_EYE_YAW_DEG.toFixed(1)})`;
}

init().catch((err) => console.error(err));
